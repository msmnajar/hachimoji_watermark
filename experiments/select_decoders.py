# experiments/select_decoders.py  (revision 7)
"""Chooses the parameters of every decision rule on VALIDATION data only.

Genuine attempts : the 52 validation images (32 BraTS slices of the 8 validation volumes,
                   20 held-out NIH/ISIC images) after each of the 21 standard attacks
                   (cached logits of select_q_v4.py).
Impostor attempts: all slices of the 8 validation volumes (val_impostors.py):
                   (a) other patient, nearest relative depth  (slice replacement),
                   (b) same volume, 1 or 2 slice steps away (3 or 6 slices: duplication).
Rule families (each tuned in its own favour, then frozen for the test data):
  NC   : parity zero-watermark of the paper, accept iff NC >= tau (exact pipeline);
  HRS64: Hachimoji dinucleotide Reed-Solomon over GF(64) (proposed);
  HRS8 : Hachimoji base Reed-Solomon over GF(8), blocks of 7;
  DNA4 : four-letter DNA Reed-Solomon over GF(4), blocks of 3;
  BCH  : binary BCH codes (n = 63 blocks, or two shortened n = 255 blocks), errors and erasures.
Code rules are evaluated by the exact bounded-distance condition 2e + s <= n - k per block
(errors-and-erasures RS / BCH decoding succeeds iff it holds; a wrong codeword is rejected by
the tag up to probability 2^-tag_bits).  Selection: zero impostor acceptance on validation,
payload >= 96 bits (64 content + 32 tag), maximum mean genuine acceptance; ties -> larger payload.
Usage: python experiments/select_decoders.py <val_logits.npz> <val_impostors.npz> <train_set.npz> <out.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))

BCH63 = [(36, 5), (30, 6), (24, 7), (18, 10), (16, 11)]
BCH255 = [(131, 18), (115, 21), (99, 23), (87, 26), (71, 29)]


def sel_idx(z, q):
    k = int(round(q * len(z)))
    return np.sort(np.argsort(-np.abs(z), kind="stable")[:k])


def sym_err(b0, z, m, n_sym, order):
    """symbol errors and reliabilities of the received logits z against bits b0 (both at S)."""
    b = (z > 0)[..., order][..., :n_sym * m]
    a = np.abs(z)[..., order][..., :n_sym * m]
    e = (b.reshape(*b.shape[:-1], n_sym, m) != b0[order][:n_sym * m].reshape(n_sym, m)).any(-1)
    return e, a.reshape(*a.shape[:-1], n_sym, m).min(-1)


def accept(e, r, nblk, nc, red, s, bch_t=None):
    e = e[..., :nblk * nc].reshape(*e.shape[:-1], nblk, nc)
    r = r[..., :nblk * nc].reshape(*r.shape[:-1], nblk, nc)
    if s:
        o = np.argsort(r, -1, kind="stable")
        er = np.zeros_like(e); np.put_along_axis(er, o[..., :s], True, -1)
        ee = (e & ~er).sum(-1)
    else:
        ee = e.sum(-1)
    lim = 2 * bch_t if bch_t is not None else red
    return (2 * ee + s <= lim).all(-1)


def code_configs():
    out = []
    for q in (0.6, 0.7, 0.8, 0.9, 1.0):
        L = int(round(q * 512))
        for il in (False, True):
            ns = min(63, L // 6)                               # HRS64: one shortened codeword
            for k in (16, 20, 24, 28):
                if ns - k >= 2:
                    for s in range(0, ns - k + 1, 4):
                        out.append(dict(fam="HRS64", q=q, il=il, m=6, nblk=1, nc=ns, k=k, s=s))
            nb = (L // 3) // 7                                  # HRS8: blocks of 7 bases
            for k in (1, 2, 3, 4):
                for s in range(0, 7 - k):
                    out.append(dict(fam="HRS8", q=q, il=il, m=3, nblk=nb, nc=7, k=k, s=s))
            nb = (L // 2) // 3                                  # DNA4: blocks of 3 bases
            for k in (1, 2):
                for s in range(0, 3 - k):
                    out.append(dict(fam="DNA4", q=q, il=il, m=2, nblk=nb, nc=3, k=k, s=s))
            for k, t in BCH63:
                nb = L // 63
                for s in (0, 2, 4, 6):
                    if s <= 2 * t:
                        out.append(dict(fam="BCH", q=q, il=il, m=1, nblk=nb, nc=63, k=k, s=s, t=t))
            for k, t in BCH255:
                nb = 1 if L <= 255 else 2
                Lb = min(255, L // nb)
                for s in (0, 4, 8, 12, 16):
                    out.append(dict(fam="BCH", q=q, il=il, m=1, nblk=nb, nc=Lb, k=k - (255 - Lb), s=s, t=t))
    for c in out:
        c["payload"] = c["nblk"] * c["k"] * c["m"]
    return [c for c in out if c["payload"] >= 96 and c["k"] > 0]


def eval_code(c, gen_items, imp_items):
    """gen_items: list of (z_clean, z_att (A,512)); imp_items: list of (z_reg, z_other)."""
    rng = np.random.RandomState(0)
    n_sym = c["nblk"] * c["nc"]
    g = []
    for z0, za in gen_items:
        S = sel_idx(z0, c["q"]); order = rng.permutation(len(S)) if c["il"] else np.arange(len(S))
        e, r = sym_err(z0[S] > 0, za[:, S], c["m"], n_sym, order)
        g.append(accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t")))
    g = np.stack(g, 1)                                        # (A, I)
    fa = {}
    for (z0, zj), kind in imp_items:
        S = sel_idx(z0, c["q"]); order = rng.permutation(len(S)) if c["il"] else np.arange(len(S))
        e, r = sym_err(z0[S] > 0, zj[S][None], c["m"], n_sym, order)
        fa[kind] = fa.get(kind, 0) + int(accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t"))[0])
    return float(g.mean()), float(g.mean(1).min()), fa


def impostor_pairs(V):
    vid, rel = V["vid"], V["reldepth"]
    pairs = []
    for i in range(len(vid)):
        for u in np.unique(vid):
            if u == vid[i]:
                continue
            js = np.where(vid == u)[0]
            pairs.append((i, js[np.argmin(np.abs(rel[js] - rel[i]))], "other_patient"))
        same = np.where(vid == vid[i])[0]
        pos = int(np.where(same == i)[0][0])
        for d in (-2, -1, 1, 2):
            if 0 <= pos + d < len(same):
                pairs.append((i, same[pos + d], f"neighbour_{abs(d)}"))
    return pairs


if __name__ == "__main__":
    vl, vi, npz, out = sys.argv[1:5]
    L = np.load(vl); V = np.load(vi)
    gen_items = [(L["clean"][i], L["att"][:, i]) for i in range(L["clean"].shape[0])]
    pairs = impostor_pairs(V)
    imp_items = [((V["z"][i], V["z"][j]), kind) for i, j, kind in pairs]
    res = dict(n_genuine=int(L["att"].shape[0] * L["att"].shape[1]), n_impostor=len(pairs),
               impostor_kinds={k: sum(1 for p in pairs if p[2] == k) for k in set(p[2] for p in pairs)},
               families={})
    rows = []
    for c in code_configs():
        gar, worst, fa = eval_code(c, gen_items, imp_items)
        rows.append(dict(c, GAR=round(gar, 4), worst=round(worst, 3), FA=fa))
    # threat model A (2-D record binding): impostors = other patients; B (volume integrity): + neighbours
    for tm, kinds in (("A_other_patient", ("other_patient",)), ("B_all", ("other_patient", "neighbour_1", "neighbour_2"))):
        res["families"][tm] = {}
        for fam in ("HRS64", "HRS8", "DNA4", "BCH"):
            cand = [r for r in rows if r["fam"] == fam and sum(r["FA"].get(k, 0) for k in kinds) == 0]
            cand.sort(key=lambda r: (-r["GAR"], -r["payload"], -r["worst"]))
            res["families"][tm][fam] = dict(selected=cand[0] if cand else None, top5=cand[:5],
                                            n_configs=sum(1 for r in rows if r["fam"] == fam))
    json.dump(dict(res, all=rows), open(out, "w"), indent=1)
    for tm, F in res["families"].items():
        for fam, v in F.items():
            print(tm, fam, v["selected"])
    print(res["n_genuine"], "genuine,", res["n_impostor"], "impostor", res["impostor_kinds"])
