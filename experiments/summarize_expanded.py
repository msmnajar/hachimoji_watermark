# experiments/summarize_expanded.py  (revision 7)
"""NC results of the parity zero-watermark on the expanded patient-level test sets for every
feature extractor (cached features, exact pipeline via CachedZW):
  mean NC over the 24 standard + clinical operations, over the 6 unseen ones, the worst operation,
  NC between different patients (record of i, clean image of j), and the patient-level paired
  bootstrap 95% CI (10,000 resamples, seed 0) of the advantage of a reference configuration.
Every test image is from a different patient, so image-level resampling is patient-level.
Usage: python experiments/summarize_expanded.py <out.json> name=cache.npy:feature:q [...]
       (q = 'all' for no selection); the first entry is the reference.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from eval_cached import zw_for, meta_2d, register, fast_nc               # noqa: E402
from eval_test_decoders import STD, CLIN                                 # noqa: E402

UNSEEN = ['Unseen_Rotation_45', 'Unseen_Gaussian_60', 'Unseen_JPEG_10', 'Unseen_Unsharp',
          'Unseen_Speckle_0.05', 'Unseen_MotionBlur_7']


def run(cache, feat, q, set_name, n_imp_per=99):
    C = np.load(cache, allow_pickle=True).item()
    E = C["sets"][set_name]; cond = C["conditions"][1:]
    G = np.full((len(E), len(cond)), np.nan); fns = []
    for i, e in enumerate(E):
        zw = zw_for(meta_2d(e["name"]), feat, q)
        rec = register(zw, e); g = fast_nc(zw, e, rec); fns.append(g)
        for c in range(len(cond)):
            v = g(e["f"][c + 1]); G[i, c] = np.nan if v is None else v
    imp = [fns[i](E[j]["f"][0]) for i in range(len(E)) for j in range(len(E)) if i != j][: len(E) * n_imp_per]
    return dict(names=[e["name"] for e in E], cond=cond, G=G, imp=np.array(imp, float))


def summ(r):
    c = r["cond"]; G = r["G"]
    i24 = [c.index(n) for n in STD + CLIN]; iu = [c.index(n) for n in UNSEEN]
    per = np.nanmean(G, 0)
    w = int(np.argmin(per[i24]))
    return dict(mean24=round(float(np.nanmean(G[:, i24])), 4), unseen=round(float(np.nanmean(G[:, iu])), 4),
                worst=(STD + CLIN)[w], worst_nc=round(float(per[i24][w]), 4),
                rot10=round(float(per[c.index("Rotation_10")]), 4), rot30=round(float(per[c.index("Rotation_30")]), 4),
                distinct_mean=round(float(r["imp"].mean()), 3), distinct_sd=round(float(r["imp"].std()), 3),
                distinct_max=round(float(r["imp"].max()), 4), per_image24=np.nanmean(G[:, i24], 1).tolist())


def boot(a, b, n=10000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rs = np.random.RandomState(seed)
    bs = [d[rs.randint(len(d), size=len(d))].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], round(float(d.mean()), 4)


if __name__ == "__main__":
    out = sys.argv[1]
    specs = []
    for a in sys.argv[2:]:
        nm, rest = a.split("=")
        cache, feat, q = rest.split(":")
        specs.append((nm, cache, feat, None if q == "all" else float(q)))
    res = json.load(open(out)) if os.path.exists(out) else {}
    for nm, cache, feat, q in specs:
        C = np.load(cache, allow_pickle=True).item()
        for s in C["sets"]:
            key = f"{nm}|{s}"
            if key in res:
                continue
            res[key] = summ(run(cache, feat, q, s))
            print(key, {k: v for k, v in res[key].items() if k != "per_image24"}, flush=True)
            json.dump(res, open(out, "w"))
    ref = specs[0][0]
    for k in list(res):
        nm, s = k.split("|")
        if nm != ref and f"{ref}|{s}" in res and "|" in k:
            ci, m = boot(res[f"{ref}|{s}"]["per_image24"], res[k]["per_image24"])
            res[k]["adv_of_ref"] = dict(mean=m, ci95=ci)
            print(f"{ref} - {nm} on {s}: {m} {ci}")
    json.dump(res, open(out, "w"))
