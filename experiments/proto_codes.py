"""Prototype (VALIDATION data only): genuine/impostor acceptance of code-offset
decoders over the selected AIF-Net bits, computed exactly from bounded-distance
decoding conditions (accept iff 2e + s <= n - k in every block).
"""
import itertools
import json
import sys

import numpy as np

D = np.load("results_v4/q_selection_logits.npz")
clean, att, vid, depth = D["clean"], D["att"], D["vid"], D["depth"]
NA, NI, NB = att.shape
rng = np.random.RandomState(0)


def sel_idx(z, q):
    k = int(round(q * len(z)))
    return np.sort(np.argsort(-np.abs(z), kind="stable")[:k])


def symbols(bits, m, order):
    """bits (..., L) -> symbols of m bits after keyed reordering, truncating."""
    b = bits[..., order]
    L = (b.shape[-1] // m) * m
    b = b[..., :L].reshape(*b.shape[:-1], -1, m)
    return b


def block_ok(err_sym, rel_sym, n, k, s, T=None):
    """err_sym: bool (..., N) symbol errors; rel_sym: reliability; returns accept bool.
    Blocks of n consecutive symbols; in each block erase the s least reliable symbols."""
    N = (err_sym.shape[-1] // n) * n
    e = err_sym[..., :N].reshape(*err_sym.shape[:-1], -1, n)
    r = rel_sym[..., :N].reshape(*rel_sym.shape[:-1], -1, n)
    if s > 0:
        o = np.argsort(r, axis=-1)
        erased = np.zeros_like(e)
        np.put_along_axis(erased, o[..., :s], True, axis=-1)
        ee = (e & ~erased).sum(-1)
    else:
        ee = e.sum(-1)
    if T is not None:
        return (ee <= T).all(-1), N // n
    return (2 * ee + s <= n - k).all(-1), N // n


def evaluate(q, m, n, k, s, interleave, T=None):
    gen, imp = [], []
    for i in range(NI):
        S = sel_idx(clean[i], q)
        order = rng.permutation(len(S)) if interleave else np.arange(len(S))
        f0 = (clean[i, S] > 0)
        sym0 = symbols(f0, m, order)
        # genuine: every attack
        za = att[:, i, S]
        sa = symbols(za > 0, m, order)
        err = (sa != sym0).any(-1)
        rel = symbols(np.abs(za), m, order).min(-1)
        ok, nblk = block_ok(err, rel, n, k, s, T)
        gen.append(ok)
        if vid[i] >= 0:      # impostor: same depth, other BraTS volume (clean)
            for j in range(NI):
                if j != i and vid[j] >= 0 and depth[j] == depth[i]:
                    zj = clean[j, S]
                    sj = symbols(zj > 0, m, order)
                    e2 = (sj != sym0).any(-1)
                    r2 = symbols(np.abs(zj), m, order).min(-1)
                    imp.append(block_ok(e2[None], r2[None], n, k, s, T)[0][0])
    gen = np.stack(gen, 1)  # (NA, NI)
    payload = nblk * k * m
    return dict(q=q, m=m, n=n, k=k, s=s, il=interleave, payload=payload,
                GAR=round(float(gen.mean()), 4), GAR_brats=round(float(gen[:, vid >= 0].mean()), 4),
                worst_attack=round(float(gen.mean(1).min()), 3), FAR=round(float(np.mean(imp)), 4), n_imp=len(imp))


rows = []
# exact-match reference (current scheme at tau = 1.00 ~ no flipped selected bit)
for q in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    rows.append(evaluate(q, 1, 1, 1, 0, False) | {"name": "exact"})
# GF(8) RS (Hachimoji symbol), n=7 (and extended 8, 9 for comparison), GF(4) RS n<=5, binary BCH bounded distance
for q in (0.7, 0.8, 0.9, 1.0):
    for il in (False, True):
        for n, k in [(7, 1), (7, 2), (7, 3), (9, 3), (9, 4)]:
            for s in (0, 1, 2):
                if s <= n - k:
                    rows.append(evaluate(q, 3, n, k, s, il) | {"name": f"RS8({n},{k})"})
        for n, k in [(5, 2), (5, 1), (3, 1)]:
            for s in (0, 1, 2):
                if s <= n - k:
                    rows.append(evaluate(q, 2, n, k, s, il) | {"name": f"RS4({n},{k})"})
        for n, k, T in [(31, 16, 3), (31, 11, 5), (63, 36, 5), (63, 30, 6), (63, 24, 7), (127, 64, 10), (127, 50, 13)]:
            rows.append(evaluate(q, 1, n, k, 0, il, T) | {"name": f"BCH({n},{k},{T})"})
json.dump(rows, open("/tmp/claude-0/proto_rows.json", "w"))
for r in rows:
    if r["payload"] >= 96:
        print(r)
