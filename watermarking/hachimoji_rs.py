# watermarking/hachimoji_rs.py  (revision 7)
"""Hachimoji Reed-Solomon code-offset layer (HRS).

The parity zero-watermark keeps one bit of every Hachimoji base, so the encoding
reduces to a keyed bit substitution (Proposition 1).  HRS uses the full base:

Registration (cover image never modified)
 1. S: the indices of the q*512 most reliable AIF-Net bits (stored in the record).
 2. The bits f_S are read in index order (or in a keyed order, `interleave`) and
    grouped into tribits; tribit u_i becomes the Hachimoji base
        beta_i = R_{r_i}(u_i),   r_i in [0, 384) from the key stream,
    i.e. every base has its own dynamic coding rule.  With `m = 6` two consecutive
    bases form a dinucleotide, the GF(64) symbol 8*beta_{2i} + beta_{2i+1}; with
    `m = 3` every base is a GF(8) symbol.
 3. The message M = content || tag, with tag = HMAC(k, "hrs-tag" || content)
    truncated to the remaining bits, is encoded by a Reed-Solomon code over
    GF(2^m), shortened to n = k + nsym <= 2^m - 1 symbols:  c = RS(M).
 4. Helper data h = c XOR s (symbol-wise; s = feature symbols) are stored.
Verification
 5. The received image gives s'; the `erase` least reliable symbols (reliability =
    smallest |logit| among the symbol's bits) are declared erasures; c' = h XOR s' is
    decoded with errors and erasures (2e + erase <= nsym); the image is accepted iff
    decoding succeeds and the tag matches; the content is then returned exactly.

Because every keyed rule R_r is a bijection of the eight bases, decodability depends
only on which symbols changed, not on the key; the rules and the key-stream
interleaving make h independent of c for an adversary without the key (for any
fixed tribit, the 384 rules map it to each of the 8 bases exactly 48 times).
"""
import hashlib
import hmac

import numpy as np
import reedsolo

from core import hachimoji_dna as H
from core.chaotic_system import HyperchaoticSystem6D, seed_from_digest


def _bits_to_int(b):
    v = 0
    for x in b:
        v = (v << 1) | int(x)
    return v


def _int_to_bits(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


class HachimojiRS:
    def __init__(self, key: bytes, q=0.9, m=6, nsym=39, k_info=24, erase=12, interleave=False, blocks=1):
        """blocks: number of RS codewords (consecutive groups of n symbols); erase is per block."""
        assert m in (2, 3, 6)      # m = 2: four-letter comparison only (no Hachimoji rule)
        self.key, self.q, self.m, self.nsym, self.k_info, self.erase, self.il = key, q, m, nsym, k_info, erase, interleave
        self.nblk = blocks
        self.nc = k_info + nsym                      # symbols per codeword
        assert self.nc <= 2 ** m - 1
        self.n = self.nc * blocks                    # symbols used in total
        prim = {2: 0x7, 3: 0xB, 6: 0x43}[m]           # x^2+x+1, x^3+x+1, x^6+x+1 (primitive)
        self.codec = reedsolo.RSCodec(nsym=nsym, nsize=2 ** m - 1, c_exp=m, prim=prim, generator=2)
        self.payload_bits = k_info * m * blocks

    # ---------------------------------------------------------- key material
    def _stream(self, n_sel, n_sym):
        """Key-stream order and rules; depend only on the key and the sizes, so they are cached."""
        ck = (n_sel, n_sym)
        if getattr(self, "_cache", None) and self._cache[0] == ck:
            return self._cache[1]
        out = self._stream_compute(n_sel, n_sym)
        self._cache = (ck, out)
        return out

    def _stream_compute(self, n_sel, n_sym):
        d = hmac.new(self.key, b"hrs-stream", hashlib.sha256).digest()
        ch = HyperchaoticSystem6D(**seed_from_digest(d))
        order = np.argsort(ch.ints(n_sel, 1 << 30), kind="stable") if self.il else np.arange(n_sel)
        rules = ch.ints(n_sym * (2 if self.m == 6 else 1), H.N_RULES)
        return order, rules

    def _symbols(self, z_sel, order, rules):
        """z_sel: logits at S -> (symbol values, reliabilities) for the first n symbols."""
        b = (z_sel[order] > 0).astype(np.int64)
        a = np.abs(z_sel[order])
        if self.m == 2:                         # four-letter DNA: two bits per base, no Hachimoji rule
            L = self.n * 2
            if len(b) < L:
                raise ValueError("not enough selected bits for this code")
            u = b[:L].reshape(-1, 2); sym = 2 * u[:, 0] + u[:, 1]
            return sym, a[:L].reshape(-1, 2).min(1)
        L3 = (len(b) // 3) * 3
        t = b[:L3].reshape(-1, 3) @ np.array([4, 2, 1])
        nb = self.n * (2 if self.m == 6 else 1)
        if len(t) < nb:
            raise ValueError("not enough selected bits for this code")
        beta = H.RULES[rules[:nb], t[:nb]]            # dynamic Hachimoji coding: one rule per base
        rel3 = a[:3 * nb].reshape(-1, 3).min(1)
        if self.m == 3:
            return beta, rel3
        return 8 * beta[0::2] + beta[1::2], np.minimum(rel3[0::2], rel3[1::2])

    def _message(self, content_bits):
        cb = len(content_bits)
        tb = self.payload_bits - cb
        assert tb >= 32, "tag must have at least 32 bits"
        tag = hmac.new(self.key, b"hrs-tag" + bytes(content_bits), hashlib.sha256).digest()
        tag_bits = [int(x) for byte in tag for x in format(byte, "08b")][:tb]
        bits = list(content_bits) + tag_bits
        return [_bits_to_int(bits[i:i + self.m]) for i in range(0, self.payload_bits, self.m)], tag_bits

    # ---------------------------------------------------------- protocol
    def register(self, z, content_bits):
        k = int(round(self.q * len(z)))
        S = np.sort(np.argsort(-np.abs(z), kind="stable")[:k]).astype(np.int32)
        order, rules = self._stream(len(S), self.n)
        s, _ = self._symbols(z[S], order, rules)
        msg, _ = self._message(content_bits)
        K = self.k_info
        c = np.concatenate([np.array(self.codec.encode(bytearray(msg[b * K:(b + 1) * K])), np.int64)
                            for b in range(self.nblk)])
        return dict(sel=S, helper=(c ^ s).astype(np.uint8), n_content=len(content_bits))

    def verify(self, z, rec):
        order, rules = self._stream(len(rec["sel"]), self.n)
        s, rel = self._symbols(z[rec["sel"]], order, rules)
        r = rec["helper"].astype(np.int64) ^ s
        dec = []
        for b in range(self.nblk):
            sl = slice(b * self.nc, (b + 1) * self.nc)
            er = sorted(np.argsort(rel[sl], kind="stable")[:self.erase].tolist()) if self.erase else None
            try:
                dec += list(self.codec.decode(bytearray(r[sl].tolist()), erase_pos=er)[0])
            except reedsolo.ReedSolomonError:
                return False, None
        bits = [x for v in dec for x in _int_to_bits(int(v), self.m)]
        content = bits[:rec["n_content"]]
        _, tag_bits = self._message(content)
        if bits[rec["n_content"]:] != tag_bits:
            return False, None
        return True, content
