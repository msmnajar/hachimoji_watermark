# experiments/operating_curves.py  (revision 7)
"""Descriptive trade-off between false acceptance and genuine acceptance for every rule family
on a TEST set, over each family's whole parameter grid (select_decoders.code_configs) and,
for the NC rule, over tau.  Nothing is selected here; the frozen operating points come from
validation (select_decoders.py / select_nc_tau.py).  Pareto fronts per family are reported.
Impostor: record of image i verified with the clean image of another patient j.
Usage: python experiments/operating_curves.py <cache.npy> <set> <test_decoders.json> <out.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from select_decoders import code_configs, sel_idx, sym_err, accept      # noqa: E402
from eval_test_decoders import z_of, STD, CLIN                          # noqa: E402


def pareto(points):
    """points: list of (FA, GAR, cfg). Keeps points not dominated (lower FA and higher GAR)."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    out, best = [], -1
    for fa, g, c in pts:
        if g > best:
            out.append((fa, g, c)); best = g
    return out


if __name__ == "__main__":
    cache_f, set_name, td, out = sys.argv[1:5]
    C = np.load(cache_f, allow_pickle=True).item()
    E = C["sets"][set_name]; cond = C["conditions"][1:]
    idx = [cond.index(n) for n in STD + CLIN]
    Z0 = np.stack([z_of(e)[0] for e in E]); Za = np.stack([z_of(e)[1:] for e in E])[:, idx]
    I = len(E)
    res = dict(set=set_name, n_impostor=I * (I - 1), families={})
    rng = np.random.RandomState(0)
    for c in code_configs():
        n_sym = c["nblk"] * c["nc"]
        g = []; fa = 0
        for i in range(I):
            S = sel_idx(Z0[i], c["q"]); order = rng.permutation(len(S)) if c["il"] else np.arange(len(S))
            b0 = Z0[i, S] > 0
            e, r = sym_err(b0, Za[i][:, S], c["m"], n_sym, order)
            g.append(accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t")).mean())
            others = np.delete(Z0, i, 0)[:, S]
            e, r = sym_err(b0, others, c["m"], n_sym, order)
            fa += int(accept(e, r, c["nblk"], c["nc"], c["nc"] - c["k"], c["s"], c.get("t")).sum())
        res["families"].setdefault(c["fam"], []).append((fa, float(np.mean(g)), {k: c[k] for k in ("q", "il", "nblk", "nc", "k", "s", "payload")}))
    T = json.load(open(td))
    G = np.array(T["nc_genuine"])[:, idx]
    # NC impostor values are not stored per pair; recompute the curve from the stored genuine NC and
    # the impostor counts at the two frozen taus only (exact); see test_decoders json.
    res["NC_points"] = [(T["rules"][k]["FA"], float(np.array(T["rules"][k]["acc"])[:, idx].mean()), k)
                        for k in T["rules"] if k.startswith("NC")]
    res["pareto"] = {f: pareto(v) for f, v in res["families"].items()}
    json.dump(res, open(out, "w"))
    for f, p in res["pareto"].items():
        print(f, [(fa, round(g, 4)) for fa, g, _ in p][:12])
    print("NC", res["NC_points"])
