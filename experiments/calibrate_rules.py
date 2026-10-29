# experiments/calibrate_rules.py  (revision 9)
"""Per-model, per-modality calibration of the decision rules on CALIBRATION data only.

Genuine : BraTS validation slices (32) / 50 NIH / 50 ISIC calibration images x 24 operations.
Impostor: BraTS: 1736 same-depth pairs of different validation patients;
          NIH, ISIC: all 39,800 ordered pairs of the 200 calibration patients.
Budget  : false-acceptance rate <= 1e-4 (BraTS: 0 of 1736; NIH/ISIC: <= 3 of 39,800).
For every family (NC threshold of the parity zero-watermark; Hachimoji GF(64) RS; Hachimoji GF(8) RS;
four-letter GF(4) RS; binary BCH), the configuration with the highest genuine acceptance within
the budget is kept, per modality ("modality-aware") and as one configuration for all ("global").
Usage: python experiments/calibrate_rules.py <calib.npz> <val_impostors.npz> <out.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from select_decoders import code_configs, sel_idx, sym_err, accept, impostor_pairs   # noqa: E402
from select_nc_tau import entry                                                      # noqa: E402
from eval_cached import zw_for, meta_2d, register, fast_nc                           # noqa: E402

FAR_TARGET = 1e-4
TAUS = np.round(np.arange(0.90, 1.0001, 0.001), 3)


def modality_data(C, V):
    D = {}
    for m in ("nih", "isic"):
        z = C[f"{m}_z"]; att = C[f"{m}_att"]
        n = len(z)
        D[m] = dict(gen_clean=z[:len(att)], gen_att=att, gen_cls=C[f"{m}_cls"][:len(att)],
                    imp_reg=z, imp_cls=C[f"{m}_cls"], pairs=[(i, j) for i in range(n) for j in range(n) if i != j],
                    imp_other=z)
    P = [(i, j) for i, j, k in impostor_pairs(V) if k == "other_patient"]
    D["brats"] = dict(gen_clean=C["brats_z"], gen_att=C["brats_att"], gen_cls=C["brats_cls"],
                      imp_reg=V["z"], imp_cls=V["cls"], pairs=P, imp_other=V["z"])
    return D


def code_eval(c, d, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    n_sym = c["nblk"] * c["nc"]
    g = []
    for z0, za in zip(d["gen_clean"], d["gen_att"]):
        S = sel_idx(z0, c["q"]); order = rng.permutation(len(S)) if c["il"] else np.arange(len(S))
        e, r = sym_err(z0[S] > 0, za[:, S], c["m"], n_sym, order)
        g.append(accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t")))
    gar = float(np.mean(g))
    by_i = {}
    for i, j in d["pairs"]:
        by_i.setdefault(i, []).append(j)
    fa = 0
    for i, js in by_i.items():
        z0 = d["imp_reg"][i]
        S = sel_idx(z0, c["q"]); order = rng.permutation(len(S)) if c["il"] else np.arange(len(S))
        e, r = sym_err(z0[S] > 0, d["imp_other"][js][:, S], c["m"], n_sym, order)
        fa += int(accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t")).sum())
    return gar, fa


def nc_values(d):
    G = []
    for k, (z0, za) in enumerate(zip(d["gen_clean"], d["gen_att"])):
        e = entry([z0] + list(za), d["gen_cls"][k])
        zw = zw_for(meta_2d(f"CALG_{k}"), "learned_v4", 0.7); rec = register(zw, e); g = fast_nc(zw, e, rec)
        G.append([g(e["f"][c]) for c in range(1, e["f"].shape[0])])
    by_i = {}
    for i, j in d["pairs"]:
        by_i.setdefault(i, []).append(j)
    I = []
    for i, js in by_i.items():
        e = entry([d["imp_reg"][i]] + [d["imp_other"][j] for j in js], d["imp_cls"][i])
        zw = zw_for(meta_2d(f"CALI_{i}"), "learned_v4", 0.7); rec = register(zw, e); g = fast_nc(zw, e, rec)
        I += [g(e["f"][c]) for c in range(1, e["f"].shape[0])]
    return np.array(G, float), np.array(I, float)


if __name__ == "__main__":
    cal, vi, out = sys.argv[1:4]
    C = np.load(cal); V = np.load(vi)
    D = modality_data(C, V)
    res = dict(far_target=FAR_TARGET, n_pairs={m: len(d["pairs"]) for m, d in D.items()},
               n_genuine={m: int(d["gen_att"].shape[0] * d["gen_att"].shape[1]) for m, d in D.items()}, rows=[])
    # NC threshold rule
    for m, d in D.items():
        G, I = nc_values(d)
        for t in TAUS:
            res["rows"].append(dict(fam="NC", cfg=dict(tau=float(t)), mod=m, GAR=float((G >= t - 1e-9).mean()),
                                    FA=int((I >= t - 1e-9).sum())))
        print("NC", m, "max impostor NC", I.max(), flush=True)
    for c in code_configs():
        cfg = {k: c[k] for k in ("fam", "q", "il", "m", "nblk", "nc", "k", "s", "payload") if k in c}
        if "t" in c:
            cfg["t"] = c["t"]
        for m, d in D.items():
            gar, fa = code_eval(c, d)
            res["rows"].append(dict(fam=c["fam"], cfg=cfg, mod=m, GAR=gar, FA=fa))
    # selection
    sel = {"modality_aware": {}, "global": {}}
    fams = sorted(set(r["fam"] for r in res["rows"]))
    for f in fams:
        sel["modality_aware"][f] = {}
        for m in D:
            budget = int(np.floor(FAR_TARGET * res["n_pairs"][m] + 1e-9))
            cand = [r for r in res["rows"] if r["fam"] == f and r["mod"] == m and r["FA"] <= budget]
            cand.sort(key=lambda r: (-r["GAR"], r["FA"], -r["cfg"].get("payload", 0)))
            sel["modality_aware"][f][m] = cand[0] if cand else None
        # global: same cfg in all modalities
        keyed = {}
        for r in res["rows"]:
            if r["fam"] == f:
                keyed.setdefault(json.dumps(r["cfg"], sort_keys=True), {})[r["mod"]] = r
        best = None
        for k, rr in keyed.items():
            if len(rr) == len(D) and all(rr[m]["FA"] <= int(np.floor(FAR_TARGET * res["n_pairs"][m] + 1e-9)) for m in D):
                g = np.mean([rr[m]["GAR"] for m in D])
                if best is None or g > best[0]:
                    best = (g, json.loads(k))
        sel["global"][f] = None if best is None else dict(cfg=best[1], mean_GAR=best[0])
    res["selected"] = sel
    json.dump(res, open(out, "w"))
    for mode in sel:
        for f, v in sel[mode].items():
            print(mode, f, json.dumps(v)[:300])
