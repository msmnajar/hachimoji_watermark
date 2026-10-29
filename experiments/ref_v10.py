# experiments/ref_v10.py  (revision 10; INDEPENDENT reference implementation for conformance tests)
"""Independent reference for the complete verification pipeline of v10_decision.Coder.

Shares no decoding, erasure, reconstruction or tag code with v10_decision.py:
  * symbols, key-stream order and Hachimoji rules: watermarking/hachimoji_rs.HachimojiRS (revision-7 code)
    for the RS families; a fresh implementation for binary BCH;
  * registration (encoding, tag, helper data): HachimojiRS.register (pure-Python reedsolo encoder) for RS;
    galois.BCH.encode for BCH;
  * erasures: re-implemented here;
  * RS decoding: Berlekamp-Welch (linear algebra over galois.GF) on the generalized-RS form of the
    reedsolo code (c_j = v_j f(beta_j), beta_j = alpha^-j, v_j = alpha^(-j*nsym) prod_{l=n}^{N-1}(beta_j - beta_l);
    j = degree, bytearray index = n-1-j; checked against reedsolo encodings in conformance_v10.py);
    a different algorithm from the Berlekamp-Massey decoder used by creedsolo;
  * BCH decoding: galois.BCH.decode (independent of FastBCH), fill-0 / fill-1 erasure rule re-implemented;
  * content extraction and HMAC-SHA256 tag check: re-implemented here.
Exhaustive nearest-code-word search (brute_block) is a third, assumption-free check for GF(4) and GF(8) blocks.
"""
import hashlib
import hmac
import itertools

import galois
import numpy as np

from core.chaotic_system import HyperchaoticSystem6D, seed_from_digest
from watermarking.hachimoji_rs import HachimojiRS

PRIM = {2: 0x7, 3: 0xB, 6: 0x43}


def tag_bits(key, content, T):
    d = hmac.new(key, b"hrs-tag" + bytes(int(b) for b in content), hashlib.sha256).digest()
    return [(byte >> (7 - i)) & 1 for byte in d for i in range(8)][:T]


class GRS:
    """The reedsolo RS code (nsym, n, GF(2^m), prim, generator 2, fcr 0) in evaluation form."""

    def __init__(self, m, n, K):
        self.GF = galois.GF(2 ** m, irreducible_poly=PRIM[m], primitive_element=2)
        GF = self.GF; a = GF(2); N = 2 ** m - 1; nsym = n - K
        self.m, self.n, self.K = m, n, K
        self.beta = GF([int(a ** ((-j) % N)) for j in range(n)])
        v = []
        for j in range(n):
            z = GF(1)
            for l in range(n, N):
                z = z * (a ** ((-j) % N) - a ** ((-l) % N))
            v.append(int(a ** ((-(j * nsym)) % N) * z))
        self.v = GF(v)

    def decode(self, r, erased):
        """r: n symbols in bytearray order; erased: set of bytearray indices.
        Returns the unique code word (bytearray order) with 2 e' + s <= n - K, or None."""
        GF, n, K = self.GF, self.n, self.K
        idx = [a for a in range(n) if a not in erased]              # bytearray indices kept
        js = [n - 1 - a for a in idx]
        P = len(idx); e = (P - K) // 2
        if e < 0:
            return None
        y = GF([int(r[a]) for a in idx]) / self.v[js]
        B = self.beta[js]
        # unknowns: Q_0..Q_{e+K-1}, E_0..E_e ;  Q(B) - y E(B) = 0
        cols = [B ** i for i in range(e + K)] + [y * (B ** i) for i in range(e + 1)]
        A = np.stack(cols, axis=1)
        ns = A.null_space()
        if ns.shape[0] == 0:
            return None
        x = ns[0]
        Q = galois.Poly(x[:e + K][::-1], field=GF)
        E = galois.Poly(x[e + K:][::-1], field=GF)
        if E == galois.Poly([0], field=GF):
            return None
        f, rem = divmod(Q, E)
        if rem != galois.Poly([0], field=GF) or f.degree >= K:
            return None
        cw_j = self.v * f(self.beta)                                # by degree j
        cw = [int(cw_j[n - 1 - a]) for a in range(n)]               # bytearray order
        err = sum(1 for a in idx if cw[a] != int(r[a]))
        if 2 * err + len(erased) > n - K:
            return None
        return cw


def brute_block(m, n, K, r, erased, codec):
    """Exhaustive search over all (2^m)^K code words (small codes only)."""
    best = []
    for msg in itertools.product(range(2 ** m), repeat=K):
        cw = list(codec.encode(bytearray(msg)))
        err = sum(1 for a in range(n) if a not in erased and cw[a] != int(r[a]))
        if 2 * err + len(erased) <= n - K:
            best.append(cw)
    assert len(best) <= 1, "radius not unique"
    return best[0] if best else None


class RefRS:
    def __init__(self, key, cfg, T):
        self.key, self.c, self.T = key, cfg, T
        self.m, self.nblk, self.nc, self.K, self.s = cfg["m"], cfg["nblk"], cfg["nc"], cfg["k"], cfg["s"]
        self.h = HachimojiRS(key, q=cfg["q"], m=self.m, nsym=self.nc - self.K, k_info=self.K, erase=self.s,
                             interleave=cfg["il"], blocks=self.nblk)
        self.grs = GRS(self.m, self.nc, self.K)

    def register(self, z0, content):
        rec = self.h.register(z0, list(content))
        assert self.h.payload_bits - len(content) == self.T
        return rec

    def verify(self, z, rec, brute=False):
        order, rules = self.h._stream(len(rec["sel"]), self.h.n)
        s, rel = self.h._symbols(z[rec["sel"]], order, rules)
        r = rec["helper"].astype(np.int64) ^ s
        msg = []
        for b in range(self.nblk):
            sl = slice(b * self.nc, (b + 1) * self.nc)
            rb, relb = r[sl], rel[sl]
            erased = set(sorted(range(self.nc), key=lambda a: (relb[a], a))[:self.s])
            cw = (brute_block(self.m, self.nc, self.K, rb, erased, self.h.codec) if brute
                  else self.grs.decode(rb, erased))
            if cw is None:
                return False, None
            msg += cw[:self.K]
        bits = [(v >> (self.m - 1 - i)) & 1 for v in msg for i in range(self.m)]
        nct = rec["n_content"]
        content = bits[:nct]
        if bits[nct:] != tag_bits(self.key, content, self.T):
            return False, None
        return True, content


class RefBCH:
    def __init__(self, key, cfg, T):
        self.key, self.c, self.T = key, cfg, T
        self.nblk, self.nc, self.K, self.s = cfg["nblk"], cfg["nc"], cfg["k"], cfg["s"]
        full_n = 255 if self.nc > 63 else 63
        self.code = galois.BCH(full_n, self.K + (full_n - self.nc))
        self.t = self.code.t

    def _bits(self, z, S):
        zs = z[S]
        if self.c["il"]:
            d = hmac.new(self.key, b"hrs-stream", hashlib.sha256).digest()
            ch = HyperchaoticSystem6D(**seed_from_digest(d))
            order = np.argsort(ch.ints(len(S), 1 << 30), kind="stable")
        else:
            order = np.arange(len(S))
        L = self.nblk * self.nc
        return (zs[order] > 0).astype(np.int64)[:L], np.abs(zs[order])[:L]

    def register(self, z0, content):
        k = int(round(self.c["q"] * len(z0)))
        S = np.sort(np.argsort(-np.abs(z0), kind="stable")[:k])
        b, _ = self._bits(z0, S)
        mb = list(content) + tag_bits(self.key, content, self.T)
        cw = []
        for i in range(self.nblk):
            cw += [int(x) for x in self.code.encode(galois.GF2(np.array(mb[i * self.K:(i + 1) * self.K], np.uint8)))]
        return dict(S=S, helper=np.array(cw) ^ b, n_content=len(content))

    def prepare(self, z, rec):
        """Received word r = h XOR bits and its erasure mask (per block), for global batched decoding."""
        b, rel = self._bits(z, rec["S"]); r = rec["helper"] ^ b
        er = np.zeros(len(r), bool)
        for i in range(self.nblk):
            relb = rel[i * self.nc:(i + 1) * self.nc]
            for a in sorted(range(self.nc), key=lambda a: (relb[a], a))[:self.s]:
                er[i * self.nc + a] = True
        return r, er

    def finish(self, msg, rec):
        if msg is None:
            return False, None
        content = msg[:rec["n_content"]]
        return (True, content) if msg[rec["n_content"]:] == tag_bits(self.key, content, self.T) else (False, None)

    def verify_many(self, Zs, rec):
        """Batched form of verify (galois decodes many words per call); same rules, same results."""
        Zs = np.asarray(Zs); P = len(Zs)
        Bs, Rel = zip(*[self._bits(z, rec["S"]) for z in Zs]); Bs = np.stack(Bs); Rel = np.stack(Rel)
        R = rec["helper"][None] ^ Bs
        alive = np.ones(P, bool); msgs = [[] for _ in range(P)]
        for i in range(self.nblk):
            rb = R[:, i * self.nc:(i + 1) * self.nc]; relb = Rel[:, i * self.nc:(i + 1) * self.nc]
            er = np.zeros_like(rb, bool)
            for p in range(P):
                er[p, sorted(range(self.nc), key=lambda a: (relb[p, a], a))[:self.s]] = True
            found = np.zeros(P, bool); cwb = np.zeros_like(rb)
            for fill in (0, 1):
                idx = np.where(alive & ~found)[0]
                if len(idx) == 0:
                    break
                x = np.where(er[idx], fill, rb[idx]).astype(np.uint8)
                cw, ne = self.code.decode(galois.GF2(x), output="codeword", errors=True)
                cw = np.array(cw, np.int64).reshape(len(idx), -1); ne = np.array(ne).reshape(-1)
                d = ((cw != rb[idx]) & ~er[idx]).sum(1)
                good = (ne >= 0) & (2 * d + self.s <= 2 * self.t)
                found[idx[good]] = True; cwb[idx[good]] = cw[good]
            alive &= found
            for p in np.where(alive)[0]:
                msgs[p] += cwb[p, :self.K].tolist()
        out = []
        for p in range(P):
            if not alive[p]:
                out.append((False, None)); continue
            content = msgs[p][:rec["n_content"]]
            ok = msgs[p][rec["n_content"]:] == tag_bits(self.key, content, self.T)
            out.append((True, content) if ok else (False, None))
        return out

    def verify(self, z, rec):
        b, rel = self._bits(z, rec["S"])
        r = rec["helper"] ^ b
        msg = []
        for i in range(self.nblk):
            rb = r[i * self.nc:(i + 1) * self.nc]; relb = rel[i * self.nc:(i + 1) * self.nc]
            er = sorted(range(self.nc), key=lambda a: (relb[a], a))[:self.s]
            found = None
            for fill in (0, 1):
                x = rb.copy(); x[er] = fill
                cw, ne = self.code.decode(galois.GF2(x.astype(np.uint8)), output="codeword", errors=True)
                if int(ne) < 0:
                    continue
                cw = np.array(cw, np.int64)
                mask = np.ones(self.nc, bool); mask[er] = False
                if 2 * int(((cw != rb) & mask).sum()) + self.s <= 2 * self.t:
                    found = cw; break
            if found is None:
                return False, None
            msg += found[:self.K].tolist()
        content = msg[:rec["n_content"]]
        if msg[rec["n_content"]:] != tag_bits(self.key, content, self.T):
            return False, None
        return True, content


def bch_decode_global(code, t, s, nblk, nc, K, R, ER, chunk=4000):
    """galois-batched bounded-distance decoding of many received words (rows of R, erasure masks ER) of the same
    BCH configuration; returns per row the decoded message bits (list) or None."""
    P = len(R); alive = np.ones(P, bool); msgs = [[] for _ in range(P)]
    for i in range(nblk):
        rb = R[:, i * nc:(i + 1) * nc]; er = ER[:, i * nc:(i + 1) * nc]
        found = np.zeros(P, bool); cwb = np.zeros_like(rb)
        for fill in (0, 1):
            idx = np.where(alive & ~found)[0]
            for c0 in range(0, len(idx), chunk):
                ii = idx[c0:c0 + chunk]
                x = np.where(er[ii], fill, rb[ii]).astype(np.uint8)
                cw, ne = code.decode(galois.GF2(x), output="codeword", errors=True)
                cw = np.array(cw, np.int64).reshape(len(ii), -1); ne = np.array(ne).reshape(-1)
                d = ((cw != rb[ii]) & ~er[ii]).sum(1)
                good = (ne >= 0) & (2 * d + s <= 2 * t)
                found[ii[good]] = True; cwb[ii[good]] = cw[good]
        alive &= found
        for p in np.where(alive)[0]:
            msgs[p] += cwb[p, :K].tolist()
    return [msgs[p] if alive[p] else None for p in range(P)]
