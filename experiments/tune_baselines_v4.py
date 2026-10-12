# experiments/tune_baselines_v4.py  (v4)
"""
Chooses, for each re-implemented baseline, the binarization rule that is most
favorable to it, on VALIDATION images only (the same images as select_q_v4.py).
Candidate rules for a real-valued feature part v (clean image) and v' (attacked):
  median : bit = v > median(v)                      (per-image median; default)
  pop    : bit = v > m_pop, m_pop = per-dimension median over 200 training images
  dct    : bit = sign of the 2nd..(n+1)th DCT-II coefficients of v (perceptual hash)
The rule with the highest validation score (mean benign agreement over the
standard attacks minus mean agreement between different BraTS validation slices)
is used in the test runs. Usage: python experiments/tune_baselines_v4.py <train_set.npz> <out.json>
"""
import json
import os
import sys

import numpy as np
from scipy.fft import dct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import baselines as B
from evaluation.attacks import STANDARD_ATTACKS

RULES = ("median", "pop", "dct")


def binarize(parts, rule, pop=None):
    out = []
    for i, v in enumerate(parts):
        if rule == "median":
            out.append(v > np.median(v))
        elif rule == "pop":
            out.append(v > pop[i])
        else:
            c = dct(v - v.mean(), norm="ortho")
            out.append(c[1:] > 0)
    return np.concatenate(out)


def main(npz, out):
    _D = np.load(npz)
    D = {k: _D[k] for k in ("brats", "vol_id", "big", "n_volumes")}
    vol, nv = D["vol_id"], int(D["n_volumes"])
    imgs, vid = [], []
    for v in range(nv - 8, nv):
        ids = [i for i in range(len(vol)) if vol[i] == v]
        for r in (0.3, 0.45, 0.6, 0.75):
            imgs.append(D["brats"][ids[int(r * (len(ids) - 1))]].copy()); vid.append(v)
    big_va = list(range(90, 100)) + list(range(190, 200))
    imgs += [D["big"][i].copy() for i in big_va]; vid += [-1] * len(big_va)
    rs0 = np.random.RandomState(0)
    tr = [i for i in range(len(vol)) if vol[i] < nv - 8]
    pop_imgs = [D["brats"][i].copy() for i in rs0.choice(tr, 100, replace=False)] + \
               [D["big"][i].copy() for i in rs0.choice([i for i in range(200) if i not in big_va], 100, replace=False)]
    res = {}
    for name, fn in B.BASELINES.items():
        popf = [fn(im) for im in pop_imgs]
        pop = [np.median(np.stack([p[i] for p in popf]), 0) for i in range(len(popf[0]))]
        clean = [fn(im) for im in imgs]
        att = []
        rs = np.random.RandomState(11)
        for an, af in STANDARD_ATTACKS.items():
            np.random.seed(rs.randint(1 << 30))
            att.append([fn(af(im)) for im in imgs])
        res[name] = {}
        for rule in RULES:
            b0 = [binarize(c, rule, pop) for c in clean]
            ben = np.mean([[np.mean(binarize(a[i], rule, pop) == b0[i]) for i in range(len(imgs))] for a in att])
            nb = sum(1 for v in vid if v >= 0)
            imp = np.mean([np.mean(b0[i] == b0[j]) for i in range(nb) for j in range(nb) if vid[i] != vid[j]])
            res[name][rule] = dict(benign=float(ben), impostor=float(imp), score=float(ben - imp))
            print(name, rule, res[name][rule], flush=True)
        res[name]["selected"] = max(RULES, key=lambda r: res[name][r]["score"])
        res[name]["pop_thresholds"] = [p.tolist() for p in pop]
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
