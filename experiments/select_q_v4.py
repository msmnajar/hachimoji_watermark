# experiments/select_q_v4.py  (v4)
"""
Chooses the fraction q of most reliable feature bits (reliability-aware
selection) on VALIDATION data only (the last 8 BraTS training-pool volumes,
cases 183-190, and the 20 held-out NIH/ISIC training-pool images); test images
and the unseen attacks are never used.

For each validation image, the AIF-Net v4 logits (with test-time augmentation)
are computed for the clean image and after every standard attack. For each q,
  benign   = agreement of the selected bits after each standard attack,
  impostor = agreement of the selected bits of a BraTS slice with the slice at the
             same position of every other validation volume (different patient,
             same anatomy: the hardest negative of the volume experiment).
q minimizes the equal error rate (EER) between the benign and the impostor
agreements; ties are broken by the higher mean benign agreement. (A first pass
used the tail separation P5(benign) - P99(impostor), which is reported as well;
it varied by less than 0.03 over q and selected q = 1.0.)

Usage: python experiments/select_q_v4.py <train_set.npz> <out.json>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import feature_net_v4 as FN4
from evaluation.attacks import STANDARD_ATTACKS

QS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)


def main(npz, out):
    _D = np.load(npz)
    D = {k: _D[k] for k in ("brats", "vol_id", "big", "n_volumes")}
    vol, nv = D["vol_id"], int(D["n_volumes"])
    imgs, vid = [], []
    for v in range(nv - 8, nv):
        ids = [i for i in range(len(vol)) if vol[i] == v]
        for r in (0.3, 0.45, 0.6, 0.75):                    # same relative depths in every volume
            imgs.append(D["brats"][ids[int(r * (len(ids) - 1))]].copy()); vid.append((v, r))
    big_va = list(range(90, 100)) + list(range(190, 200))    # held-out NIH and ISIC training-pool images
    imgs += [D["big"][i].copy() for i in big_va]; vid += [(-1, i) for i in big_va]
    cache = os.path.splitext(out)[0] + "_logits.npz"
    if os.path.exists(cache):
        z = np.load(cache); clean, att = z["clean"], z["att"]
    else:
        clean = np.stack([FN4.logits(im) for im in imgs])
        att = []
        rs = np.random.RandomState(11)
        for name, fn in STANDARD_ATTACKS.items():
            np.random.seed(rs.randint(1 << 30))
            att.append(np.stack([FN4.logits(fn(im)) for im in imgs]))
            print(name, flush=True)
        att = np.stack(att)
    np.savez(os.path.splitext(out)[0] + "_logits.npz", clean=clean, att=att, names=np.array(list(STANDARD_ATTACKS)),
             vid=np.array([v[0] for v in vid]), depth=np.array([float(v[1]) for v in vid]))
    res = {"q": {}}
    nb = sum(1 for v in vid if v[0] >= 0)
    for q in QS:
        k = int(round(q * clean.shape[1]))
        sel = np.argsort(-np.abs(clean), 1)[:, :k]
        b0 = np.take_along_axis(clean, sel, 1) > 0
        benign = np.array([[np.mean((np.take_along_axis(att[a], sel, 1)[i] > 0) == b0[i])
                            for i in range(len(imgs))] for a in range(len(att))])
        imp = [np.mean((clean[j][sel[i]] > 0) == b0[i]) for i in range(nb) for j in range(nb)
               if vid[i][0] != vid[j][0] and vid[i][1] == vid[j][1]]
        bb = benign[:, :nb].ravel(); imp = np.asarray(imp)
        thr = np.unique(np.concatenate([bb, imp]))
        e = min(max(np.mean(bb < t), np.mean(imp >= t)) for t in thr)
        res["q"][str(q)] = dict(eer=float(e), benign_mean=float(benign.mean()), benign_p5=float(np.percentile(benign, 5)),
                                benign_min=float(benign.min()), impostor_mean=float(np.mean(imp)),
                                impostor_p99=float(np.percentile(imp, 99)), impostor_max=float(np.max(imp)),
                                separation=float(np.percentile(benign, 5) - np.percentile(imp, 99)))
        print(q, json.dumps(res["q"][str(q)]), flush=True)
    res["selected_q"] = float(min(QS, key=lambda q: (round(res["q"][str(q)]["eer"], 6), -res["q"][str(q)]["benign_mean"])))
    res["weights_sha256"] = FN4.weights_sha256()
    json.dump(res, open(out, "w"), indent=1)
    print("selected q =", res["selected_q"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
