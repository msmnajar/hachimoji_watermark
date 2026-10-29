# experiments/eval_calibrated.py  (revision 9)
"""Applies the rules calibrated per model (calibrate_rules.py, calibration data only) to the
100-patient TEST sets: genuine acceptance over the 24 operations and false acceptances among the
9900 pairs of different patients, for the modality-aware and the global operating points.
Code rules: exact errors-and-erasures decoding condition; NC rule: exact parity pipeline.
Usage: python experiments/eval_calibrated.py <cache.npy> <calibrated.json> <out.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from select_decoders import sel_idx, sym_err, accept                 # noqa: E402
from eval_test_decoders import z_of, STD, CLIN                       # noqa: E402
from eval_cached import zw_for, meta_2d, register, fast_nc           # noqa: E402

MOD = {"brats100": "brats", "nih100": "nih", "isic100": "isic"}


def code_test(c, Z0, Za):
    rng = np.random.RandomState(0)
    n_sym = c["nblk"] * c["nc"]
    I = len(Z0); acc = np.zeros((I, Za.shape[1]), bool); fa = 0
    for i in range(I):
        S = sel_idx(Z0[i], c["q"]); order = rng.permutation(len(S)) if c["il"] else np.arange(len(S))
        b0 = Z0[i, S] > 0
        e, r = sym_err(b0, Za[i][:, S], c["m"], n_sym, order)
        acc[i] = accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t"))
        e, r = sym_err(b0, np.delete(Z0, i, 0)[:, S], c["m"], n_sym, order)
        fa += int(accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t")).sum())
    return acc, fa


if __name__ == "__main__":
    cache, calj, out = sys.argv[1:4]
    C = np.load(cache, allow_pickle=True).item()
    CAL = json.load(open(calj))["selected"]
    res = {}
    for s, E in C["sets"].items():
        if s not in MOD:
            continue
        m = MOD[s]
        cond = C["conditions"][1:]; idx = [cond.index(n) for n in STD + CLIN]
        Z0 = np.stack([z_of(e)[0] for e in E]); Za = np.stack([z_of(e)[1:] for e in E])[:, idx]
        # NC rule: genuine NC and impostor NC (exact pipeline)
        G = []; Iv = []
        fns = []
        for e in E:
            zw = zw_for(meta_2d(e["name"]), "learned_v4", 0.7); rec = register(zw, e); g = fast_nc(zw, e, rec)
            fns.append(g); G.append([g(e["f"][c + 1]) for c in idx])
        for i in range(len(E)):
            for j in range(len(E)):
                if i != j:
                    Iv.append(fns[i](E[j]["f"][0]))
        G = np.array(G); Iv = np.array(Iv)
        out_s = {"NC_tau1.00": dict(acc=(G >= 1 - 1e-9).tolist(), FA=int((Iv >= 1 - 1e-9).sum()))}
        for mode in ("modality_aware", "global"):
            for f, v in CAL[mode].items():
                if v is None or (mode == "modality_aware" and v.get(m) is None):
                    out_s[f"{mode}|{f}"] = None
                    continue
                cfg = v["cfg"] if mode == "global" else v[m]["cfg"]
                if f == "NC":
                    t = cfg["tau"]
                    out_s[f"{mode}|{f}"] = dict(cfg=cfg, acc=(G >= t - 1e-9).tolist(), FA=int((Iv >= t - 1e-9).sum()))
                else:
                    acc, fa = code_test(cfg, Z0, Za)
                    out_s[f"{mode}|{f}"] = dict(cfg=cfg, acc=acc.tolist(), FA=fa)
        res[s] = out_s
        for k, v in out_s.items():
            print(s, k, None if v is None else (round(float(np.mean(v["acc"])), 4), v["FA"]), flush=True)
    json.dump(res, open(out, "w"))
