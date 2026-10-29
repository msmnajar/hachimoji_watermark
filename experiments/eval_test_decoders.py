# experiments/eval_test_decoders.py  (revision 7)
"""Applies the decision rules FROZEN on validation (select_decoders.py, select_nc_tau.py,
threat model A: impostor = image of another patient) to the expanded test sets.

Genuine : every test image after each operation (24 standard + clinical, and all 34).
Impostor: record of image i verified with the clean image of every other patient j of the
          same set (BraTS-100: middle slices, i.e. the same relative depth).
NC rules use the exact parity pipeline (CachedZW); code rules use the bounded-distance
condition, and the proposed HRS64 rule is additionally run with the real decoder
(watermarking/hachimoji_rs.py) on all genuine attempts to confirm the condition.
Usage: python experiments/eval_test_decoders.py <cache.npy> <set> <sel_decoders.json> <sel_nc.json> <out.json> [prefix]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from eval_cached import zw_for, meta_2d, register, extract_nc, fast_nc      # noqa: E402
from select_decoders import sel_idx, sym_err, accept              # noqa: E402

STD = ['Gaussian_5', 'Gaussian_10', 'Gaussian_20', 'Gaussian_40', 'SaltPepper_1%', 'SaltPepper_5%',
       'SaltPepper_10%', 'JPEG_90', 'JPEG_70', 'JPEG_50', 'JPEG_30', 'Median_3', 'Median_5', 'Rotation_5',
       'Rotation_10', 'Rotation_30', 'Scale_0.5', 'Scale_0.75', 'Crop_10%', 'Crop_25%', 'HistEq']
CLIN = ['PACS_PNG', 'PACS_JPEG95', 'Downsample_x2']


def z_of(e):
    """signed reliability: bit = z > 0, reliability = |z| (AIF-Net: the standardized logits)."""
    return np.where(e["f"] == 1, 1.0, -1.0) * e["rel"]


def code_rule(c, Z0, Za, Zimp_pairs):
    rng = np.random.RandomState(0)
    n_sym = c["nblk"] * c["nc"]
    gen = []
    for z0, za in zip(Z0, Za):
        S = sel_idx(z0, c["q"]); order = rng.permutation(len(S)) if c["il"] else np.arange(len(S))
        e, r = sym_err(z0[S] > 0, za[:, S], c["m"], n_sym, order)
        gen.append(accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t")))
    fa = 0
    for i, j in Zimp_pairs:
        z0 = Z0[i]; S = sel_idx(z0, c["q"]); order = rng.permutation(len(S)) if c["il"] else np.arange(len(S))
        e, r = sym_err(z0[S] > 0, Z0[j][S][None], c["m"], n_sym, order)
        fa += int(accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t"))[0])
    return np.stack(gen), fa                      # (I, C), count


if __name__ == "__main__":
    cache_f, set_name, sd, sn, out = sys.argv[1:6]
    prefix = sys.argv[6] if len(sys.argv) > 6 else ""
    C = np.load(cache_f, allow_pickle=True).item()
    E = C["sets"][set_name]
    cond = C["conditions"][1:]
    SD = json.load(open(sd))["families"]["A_other_patient"]
    SN = json.load(open(sn))
    tau_t = SN["A_other_patient"]["tau_far0"]
    I = len(E)
    pairs = [(i, j) for i in range(I) for j in range(I) if i != j]
    Z0 = [z_of(e)[0] for e in E]
    Za = [z_of(e)[1:] for e in E]
    res = dict(set=set_name, n_images=I, n_impostor=len(pairs), conditions=cond, rules={})
    # ---- NC rules (exact pipeline)
    G = np.zeros((I, len(cond))); fns = []
    for i, e in enumerate(E):
        zw = zw_for(meta_2d(e["name"], prefix=prefix), "learned_v4", 0.7)
        rec = register(zw, e); g = fast_nc(zw, e, rec); fns.append(g)
        G[i] = [g(e["f"][c]) for c in range(1, len(cond) + 1)]
        if i < 2:                                   # check against the full extraction
            assert all(extract_nc(zw, e, c, rec) == G[i][c - 1] for c in range(1, len(cond) + 1))
    imp_nc = [fns[i](E[j]["f"][0]) for i, j in pairs]
    imp_nc = np.array(imp_nc)
    for nm, t in (("NC_tau1.00", 1.0), (f"NC_tau{tau_t:.3f}", tau_t)):
        res["rules"][nm] = dict(acc=(G >= t - 1e-9).tolist(), FA=int((imp_nc >= t - 1e-9).sum()))
    res["nc_genuine"] = G.tolist(); res["nc_impostor_max"] = float(imp_nc.max())
    # ---- code rules (frozen configs)
    for fam in ("HRS64", "HRS8", "DNA4", "BCH"):
        c = SD[fam]["selected"]
        g, fa = code_rule(c, Z0, Za, pairs)
        res["rules"][fam] = dict(config={k: v for k, v in c.items() if k not in ("GAR", "worst", "FA")},
                                 acc=g.tolist(), FA=fa)
    # ---- real decoder for HRS64 (genuine attempts)
    from watermarking.hachimoji_rs import HachimojiRS
    from core.key_generator import PatientKey
    c = SD["HRS64"]["selected"]
    agree = 0; tot = 0
    rs = np.random.RandomState(1)
    for i, e in enumerate(E):
        key = PatientKey(meta_2d(e["name"], prefix=prefix)).sub("hrs")
        h = HachimojiRS(key, q=c["q"], m=6, nsym=c["nc"] - c["k"], k_info=c["k"], erase=c["s"], interleave=c["il"])
        content = list(rs.randint(0, 2, h.payload_bits - 32))
        rec = h.register(Z0[i], content)
        for ci in range(len(cond)):
            ok, got = h.verify(Za[i][ci], rec)
            res.setdefault("hrs64_real", []).append(bool(ok and got == content))
    json.dump(res, open(out, "w"))
    def summ(acc, names):
        a = np.array(acc, float); idx = [cond.index(n) for n in names]
        return round(float(a[:, idx].mean()), 4)
    print(set_name, "impostor pairs", len(pairs))
    for nm, r in res["rules"].items():
        print(f"{nm:14s} GAR std+clin {summ(r['acc'], STD + CLIN):.4f}  all34 {summ(r['acc'], cond):.4f}  FA {r['FA']}")
    print("HRS64 real decoder genuine acceptance (all34):", round(float(np.mean(res["hrs64_real"])), 4))
