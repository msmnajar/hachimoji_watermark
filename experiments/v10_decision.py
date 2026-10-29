# experiments/v10_decision.py  (revision 10, draft for protocol review)
"""Exact verification decision for every code family, with classification of every attempt.

All families (Hachimoji GF(64) / GF(8) RS, four-letter GF(4) RS, binary BCH) share:
  * bit selection S (q most reliable bits of the registered image),
  * bit order: keyed (HMAC(k_h, "hrs-stream") key stream) if il, else index order,
  * message M = content || tag, tag = first T bits of HMAC(k_h, "hrs-tag" || content),
  * code offset h = c XOR sigma (symbols; bits for BCH),
  * erasures: the s least reliable received symbols (bits for BCH) of every block,
  * BOUNDED-DISTANCE decision: a block is decoded only to a code word c' with
        2 e' + s <= n - K   (RS)      or   2 e' + s <= 2 t   (BCH),
    where e' = number of NON-erased positions in which c' differs from the received word.
    (reedsolo can return words beyond this radius; they are rejected - see bench_v10.py.)
  * acceptance: every block decodes within the radius and the tag of the decoded content matches.

Classification of an attempt against the registrant's genuine symbols sigma_reg:
  cond   : 2 e + s <= bound in every block, e counted against sigma_reg  (decodes to the genuine c)
  status : 'accept_genuine'  - accepted, decoded message == registered message   (mechanism a)
           'accept_other'    - accepted, decoded message != registered message   (mechanism b: tag collision)
           'reject_fail'     - some block has no code word within the radius
           'reject_tag'      - all blocks decoded within the radius, tag mismatch
"""
import hashlib
import hmac

import os

import numpy as np

if os.environ.get("V10_PURE_RS") == "1":
    import reedsolo                     # pure-Python reference implementation
else:
    import creedsolo as reedsolo        # Cython build of the same reedsolo 1.7.0 source (identical outputs, bench_v10)

from core import hachimoji_dna as H
from core.chaotic_system import HyperchaoticSystem6D, seed_from_digest

PRIM = {2: 0x7, 3: 0xB, 6: 0x43}


def sel_idx(z, q):
    k = int(round(q * len(z)))
    return np.sort(np.argsort(-np.abs(z), kind="stable")[:k])


def _bits(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def _int(b):
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v


_RAW = {}
_CODECS = {}


def _bch(n, k):
    if ("bch", n, k) not in _CODECS:
        import galois
        _CODECS[("bch", n, k)] = (galois.BCH(n, k), FastBCH(n, k))
    return _CODECS[("bch", n, k)]


_ACTIVE_FIELD = [None]


def activate_field(m):
    """creedsolo (unlike pure-Python reedsolo) keeps ONE set of module-global GF log/antilog tables, set when an
    RSCodec is constructed and NOT restored by encode/decode.  Every RS encode/decode in this module therefore
    first (re)activates the tables of its own field GF(2^m)."""
    if _ACTIVE_FIELD[0] != m:
        reedsolo.init_tables(PRIM[m], 2, m)
        _ACTIVE_FIELD[0] = m


def _rs(m, nsym):
    if ("rs", m, nsym) not in _CODECS:
        _CODECS[("rs", m, nsym)] = reedsolo.RSCodec(nsym=nsym, nsize=2 ** m - 1, c_exp=m, prim=PRIM[m], generator=2)
        _ACTIVE_FIELD[0] = m
    return _CODECS[("rs", m, nsym)]


def raw_stream(key, n):
    """First n raw samples of the key stream HyperchaoticSystem6D(HMAC(key, "hrs-stream")); the sampler is
    sequential (buffered RK4 steps), so ints(a) followed by ints(b) equals a split of raw(a + b)."""
    r = _RAW.get(key)
    if r is None or len(r) < n:
        d = hmac.new(key, b"hrs-stream", hashlib.sha256).digest()
        ch = HyperchaoticSystem6D(**seed_from_digest(d))
        r = ch.raw(max(n, 1024))
        if len(_RAW) > 4096:
            _RAW.clear()
        _RAW[key] = r
    return r[:n]


def _ints(raw, m):
    from core.chaotic_system import SCALE
    frac = np.mod(np.abs(raw) * SCALE, 1.0)
    return np.minimum((frac * m).astype(np.int64), m - 1)


class Coder:
    """cfg: fam, q, il, m (1 = binary BCH), nblk, nc (n per block), k (K per block), s, t (BCH)."""

    def __init__(self, key, cfg, tag_bits, check_content=True):
        self.key, self.c, self.T = key, cfg, tag_bits
        self.m, self.nblk, self.nc, self.K, self.s = cfg["m"], cfg["nblk"], cfg["nc"], cfg["k"], cfg["s"]
        self.msg_bits = self.nblk * self.K * self.m
        self.content_bits = self.msg_bits - tag_bits
        if check_content:
            assert self.content_bits >= 64, "infeasible: fewer than 64 content bits"
        if self.m == 1:
            full_n = 255 if cfg["nc"] > 63 else 63
            full_k = cfg["k"] + (full_n - cfg["nc"])
            self.bch, self.fbch = _bch(full_n, full_k)
            assert self.bch.t == cfg["t"], (self.bch.t, cfg["t"])
            self.bound = 2 * cfg["t"]
        else:
            self.rs = _rs(self.m, self.nc - self.K)
            self.bound = self.nc - self.K
        self._cache = None

    # ------------------------------------------------------------------ key stream
    def stream(self, n_sel):
        if self._cache and self._cache[0] == n_sel:
            return self._cache[1]
        nb = self.nblk * self.nc * (2 if self.m == 6 else 1)
        il = self.c["il"]
        raw = raw_stream(self.key, (n_sel if il else 0) + nb)
        order = np.argsort(_ints(raw[:n_sel], 1 << 30), kind="stable") if il else np.arange(n_sel)
        off = n_sel if il else 0
        rules = _ints(raw[off:off + nb], H.N_RULES) if self.m in (3, 6) else None
        self._cache = (n_sel, (order, rules))
        return order, rules

    def symbols(self, Zs, order, rules):
        """Zs: (P, |S|) logits at S -> symbols (P, nblk*nc) and reliabilities."""
        b = (Zs[:, order] > 0).astype(np.int64); a = np.abs(Zs[:, order])
        n = self.nblk * self.nc
        if self.m == 1:
            if b.shape[1] < n:
                raise ValueError("not enough bits")
            return b[:, :n], a[:, :n]
        if self.m == 2:
            L = 2 * n
            if b.shape[1] < L:
                raise ValueError("not enough bits")
            u = b[:, :L].reshape(len(b), n, 2)
            return 2 * u[..., 0] + u[..., 1], a[:, :L].reshape(len(b), n, 2).min(-1)
        nb = n * (2 if self.m == 6 else 1)
        if b.shape[1] < 3 * nb:
            raise ValueError("not enough bits")
        t = b[:, :3 * nb].reshape(len(b), nb, 3) @ np.array([4, 2, 1])
        beta = H.RULES[rules[None, :nb], t]
        rel3 = a[:, :3 * nb].reshape(len(b), nb, 3).min(-1)
        if self.m == 3:
            return beta, rel3
        return 8 * beta[:, 0::2] + beta[:, 1::2], np.minimum(rel3[:, 0::2], rel3[:, 1::2])

    # ------------------------------------------------------------------ message
    def tag(self, content):
        d = hmac.new(self.key, b"hrs-tag" + bytes(content), hashlib.sha256).digest()
        return [int(x) for byte in d for x in format(byte, "08b")][:self.T]

    def message(self, content):
        return list(content) + self.tag(content)

    def encode(self, msg_bits):
        if self.m != 1:
            activate_field(self.m)
        out = []
        for b in range(self.nblk):
            blk = msg_bits[b * self.K * self.m:(b + 1) * self.K * self.m]
            if self.m == 1:
                import galois
                out += list(np.array(self.bch.encode(galois.GF2(np.array(blk, np.uint8)))))
            else:
                sy = [_int(blk[i:i + self.m]) for i in range(0, len(blk), self.m)]
                out += list(self.rs.encode(bytearray(sy)))
        return np.array(out, np.int64)

    # ------------------------------------------------------------------ protocol
    def register(self, z0, content):
        S = sel_idx(z0, self.c["q"])
        order, rules = self.stream(len(S))
        sig, _ = self.symbols(z0[S][None], order, rules)
        msg = self.message(content)
        c = self.encode(msg)
        return dict(S=S, helper=c ^ sig[0], msg=msg, sig=sig[0])

    def erasures(self, rel):
        """rel: (P, nblk*nc) -> boolean erasure mask (s least reliable per block, stable order)."""
        R = rel.reshape(len(rel), self.nblk, self.nc)
        E = np.zeros(R.shape, bool)
        if self.s:
            o = np.argsort(R, -1, kind="stable")[..., :self.s]
            np.put_along_axis(E, o, True, -1)
        return E.reshape(len(rel), -1)

    def condition(self, sig_recv, er, sig_reg):
        e = (sig_recv != sig_reg[None]) & ~er
        e = e.reshape(len(e), self.nblk, self.nc).sum(-1)
        return (2 * e + self.s <= self.bound).all(-1)

    def decode_block(self, r, er_pos):
        """returns the message symbols/bits of the unique code word within the radius, or None."""
        if self.m == 1:
            raise RuntimeError("BCH blocks are decoded in batch")
        try:
            msg, msgecc, _ = self.rs.decode(bytearray(r.tolist()), erase_pos=er_pos if er_pos else None)
        except reedsolo.ReedSolomonError:
            return None
        cw = np.array(list(msgecc), np.int64)
        mask = np.ones(len(r), bool); mask[er_pos] = False
        if 2 * int(((cw != r) & mask).sum()) + self.s > self.bound:
            return None                                  # beyond the bounded-distance radius
        return list(msg)

    def verify_batch(self, Z, rec):
        """Z: (P, 512) received logits -> cond (P,), status list, timing-free."""
        order, rules = self.stream(len(rec["S"]))
        sig, rel = self.symbols(Z[:, rec["S"]], order, rules)
        er = self.erasures(rel)
        cond = self.condition(sig, er, rec["sig"])
        R = rec["helper"][None] ^ sig
        if self.m == 1:
            return cond, self._verify_bch(R, er, rec)
        activate_field(self.m)
        status = []
        for p in range(len(Z)):
            dec = []
            for b in range(self.nblk):
                sl = slice(b * self.nc, (b + 1) * self.nc)
                out = self.decode_block(R[p, sl], np.where(er[p, sl])[0].tolist())
                if out is None:
                    dec = None; break
                dec += out
            status.append(self._classify(dec, rec))
        return cond, status

    def _verify_bch(self, R, er, rec):
        P = len(R); ok = np.ones(P, bool); bits = [[] for _ in range(P)]
        for b in range(self.nblk):
            sl = slice(b * self.nc, (b + 1) * self.nc)
            r = R[:, sl]; e = er[:, sl]; found = np.zeros(P, bool); cwb = np.zeros_like(r)
            for fill in (0, 1):
                x = np.where(e, fill, r).astype(np.uint8)
                idx = np.where(ok & ~found)[0]
                if len(idx) == 0:
                    break
                cw, ne = self.fbch.decode(x[idx])
                dist = ((cw != r[idx]) & ~e[idx]).sum(-1)
                good = (np.array(ne) >= 0) & (2 * dist + self.s <= self.bound)
                found[idx[good]] = True; cwb[idx[good]] = cw[good]
            ok &= found
            for p in np.where(ok)[0]:
                bits[p] += cwb[p, :self.K].tolist()        # systematic: message bits first
        return [self._classify(bits[p] if ok[p] else None, rec) for p in range(P)]

    def _classify(self, dec, rec):
        if dec is None:
            return "reject_fail"
        if self.m == 1:
            mb = [int(x) for x in dec]
        else:
            mb = [x for v in dec for x in _bits(int(v), self.m)]
        content = mb[:self.content_bits]
        if mb[self.content_bits:] != self.tag(content):
            return "reject_tag"
        return "accept_genuine" if mb == rec["msg"] else "accept_other"


# ---------------------------------------------------------------------- NC rule, vectorized
def nc_batch_fn(zw, e, rec):
    """Vectorized, exact equivalent of eval_cached.fast_nc: returns G(F) -> NC (rounded to 4 decimals)
    for a batch F (P, n_bits) of feature-bit vectors.  The parity of every watermark bit depends on one
    feature bit (Proposition 3.1), so par(f) = where(f[p], par1, par0) with par0/par1 the parities of the
    all-zero and all-one feature vectors; the inverse Arnold map is a fixed permutation."""
    import sys as _s
    from eval_cached import WM
    from watermarking.zero_watermark import arnold, WM_SIZE
    blk, grp = e["blk"], e["grp"]; cls = rec["block_class"]
    full_fn = lambda p: np.where(blk[p] < 0, True, cls[np.maximum(blk[p], 0)])
    nb = len(e["f"][0])
    ks, _ = zw._keystream(nb, int(grp.max()) + 1, full_fn, rec["uniform_full"], rec.get("sel"))
    par0 = zw._code_parity(np.zeros(nb, np.uint8), grp, ks); par1 = zw._code_parity(np.ones(nb, np.uint8), grp, ks)
    p = ks["p"]
    perm = arnold(np.arange(WM_SIZE * WM_SIZE).reshape(WM_SIZE, WM_SIZE), *zw.ap, inverse=True).ravel()
    a = WM.ravel().astype(float); aa = a @ a
    mwk = rec["mwk"].astype(np.uint8)

    def G(F):
        F = np.asarray(F)
        ok = F.max(1) <= 1
        par = np.where(F[:, p] == 1, par1[None], par0[None])
        w = (par ^ mwk[None])[:, perm].astype(float)
        num = w @ a; den = np.sqrt(aa * (w * w).sum(1)) + 1e-15
        out = np.round(num / den, 4)
        return np.where(ok, out, np.nan)
    return G


# ---------------------------------------------------------------------- vectorized binary BCH decoder
class FastBCH:
    """Bounded-distance (t-error) Berlekamp-Massey decoder for the narrow-sense binary BCH codes of
    galois.BCH(n, k) (same field, primitive element and roots), vectorized over words; shortened words
    (length L < n) are the last L coefficients (galois convention: index 0 = highest degree).
    decode(X) -> (codewords, n_corrected) with n_corrected = -1 on failure, as galois.BCH.decode."""

    def __init__(self, n, k):
        import galois
        self.ref = galois.BCH(n, k)
        self.n, self.t = n, self.ref.t
        GF = self.ref.extension_field
        self.q = GF.order
        prim = int(GF.primitive_element)
        exp = np.zeros(2 * self.q, np.int64); log = np.zeros(self.q, np.int64); x = 1
        for i in range(self.q - 1):
            exp[i] = x; log[x] = i
            x = int(GF(x) * GF(prim))
        exp[self.q - 1:2 * self.q - 2] = exp[:self.q - 1]
        self.exp, self.log = exp, log
        assert int(self.ref.alpha) == prim and self.ref.c == 1

    def mul(self, a, b):
        z = (a == 0) | (b == 0)
        r = self.exp[(self.log[a] + self.log[b]) % (self.q - 1)]
        return np.where(z, 0, r)

    def decode(self, X):
        X = np.asarray(X, np.int64); W, L = X.shape; t = self.t; Q = self.q - 1
        deg = np.arange(L - 1, -1, -1)                      # degree of every array position
        S = np.zeros((W, 2 * t), np.int64)
        for j in range(1, 2 * t + 1):
            pw = self.exp[(deg * j) % Q]                    # alpha^(j*deg)
            S[:, j - 1] = np.bitwise_xor.reduce(np.where(X == 1, pw[None], 0), axis=1)
        # Berlekamp-Massey (vectorized)
        Lam = np.zeros((W, 2 * t + 2), np.int64); Lam[:, 0] = 1
        Bp = Lam.copy(); Lr = np.zeros(W, np.int64); mshift = np.ones(W, np.int64); b = np.ones(W, np.int64)
        for r in range(2 * t):
            d = S[:, r].copy()
            for i in range(1, t + 1):
                d ^= self.mul(Lam[:, i], S[:, r - i] if r - i >= 0 else np.zeros(W, np.int64)) * (r - i >= 0)
            nz = d != 0
            coef = np.where(nz, self.exp[(self.log[np.maximum(d, 1)] - self.log[b]) % Q], 0)
            shifted = np.zeros_like(Bp)
            for s_ in range(1, 2 * t + 1):
                sel = mshift == s_
                if sel.any():
                    shifted[sel, s_:] = Bp[sel, :-s_]
            T_ = Lam ^ self.mul(coef[:, None], shifted)
            upd = nz & (2 * Lr <= r)
            Bp = np.where(upd[:, None], Lam, Bp); b = np.where(upd, d, b)
            Lr = np.where(upd, r + 1 - Lr, Lr)
            mshift = np.where(upd, 1, mshift + 1)
            Lam = np.where(nz[:, None], T_, Lam)
        degL = np.array([int(np.max(np.nonzero(l)[0])) for l in Lam])
        # Chien search over the L transmitted positions: error at degree p iff Lam(alpha^-p) = 0
        val = np.zeros((W, L), np.int64)
        for i in range(2 * t + 2):
            pw = self.exp[((-deg * i) % Q)]
            val ^= self.mul(Lam[:, i][:, None], np.broadcast_to(pw[None], (W, L)))
        err = val == 0
        nerr = err.sum(1)
        ok = (degL == Lr) & (nerr == degL) & (degL <= t)
        C = np.where(err, 1 - X, X)
        if ok.any():                                        # corrected word must be a code word
            S2 = np.zeros((W, 2 * t), np.int64)
            for j in range(1, 2 * t + 1):
                pw = self.exp[(deg * j) % Q]
                S2[:, j - 1] = np.bitwise_xor.reduce(np.where(C == 1, pw[None], 0), axis=1)
            ok &= (S2 == 0).all(1)
        return np.where(ok[:, None], C, X), np.where(ok, nerr, -1)
