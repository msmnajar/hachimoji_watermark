# experiments/operating_points.py
"""Genuine/impostor analysis of the existing per-image results (no new runs).

Genuine attempts : NC of an image's own record after each operation (correct key).
Impostor attempts: (a) NC of a different BraTS test slice against a record (E6, 380 pairs);
                   (b) NC under the six wrong-key conditions on TCIA (E3_keytests).
For each threshold, reports the acceptance rate of genuine attempts per operation group
and the false-acceptance rate of impostors. tau = 1.00 itself was fixed on validation
volumes (threshold_selection_v4.json); the other thresholds are shown only to
characterise the trade-off and are NOT used to select anything on test data.

Also computes patient-clustered paired bootstrap CIs (BraTS: case id without the
time-point suffix; NIH: patient id = file-name prefix; ISIC: no patient ids available,
so image-level only).

Usage: python experiments/operating_points.py <results_v4_dir> <out_json>
"""
import json
import sys

import numpy as np

R, OUT = sys.argv[1], sys.argv[2]
P = json.load(open(f"{R}/results_v4_learned_v4_q0.7.json"))
HC = json.load(open(f"{R}/results_v4_handcrafted_qall.json"))
FN = json.load(open(f"{R}/results_v4_focalnet_q0.7.json"))

STD = ['Gaussian_5', 'Gaussian_10', 'Gaussian_20', 'Gaussian_40', 'SaltPepper_1%', 'SaltPepper_5%',
       'SaltPepper_10%', 'JPEG_90', 'JPEG_70', 'JPEG_50', 'JPEG_30', 'Median_3', 'Median_5', 'Rotation_5',
       'Rotation_10', 'Rotation_30', 'Scale_0.5', 'Scale_0.75', 'Crop_10%', 'Crop_25%', 'HistEq']
CLIN = ['PACS_PNG', 'PACS_JPEG95', 'Downsample_x2']
TAUS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 1.00]


def accept(vals, t):
    return float(np.mean(np.asarray(vals) >= t - 1e-9))


def per_attack(res, key, t):
    out = {}
    for a in res[key][0]["attacks"]:
        out[a] = accept([x["attacks"][a] for x in res[key]], t)
    return out


def patient_of(ds, name):
    if ds == "E2_brats":
        return name.rsplit("-", 1)[0]
    if ds == "E5_nih":
        return name.split("_")[0]
    return name


def mean_nc(x, attacks):
    return float(np.mean([x["attacks"][a] for a in attacks]))


def cluster_boot(ds, attacks, n=10000, seed=0):
    rs = np.random.RandomState(seed)
    pa = {x["name"]: mean_nc(x, attacks) for x in P[ds]}
    pb = {x["name"]: mean_nc(x, attacks) for x in FN[ds]}
    names = [x["name"] for x in P[ds]]
    diff = np.array([pa[k] - pb[k] for k in names])
    groups = {}
    for i, k in enumerate(names):
        groups.setdefault(patient_of(ds, k), []).append(i)
    G = list(groups.values())
    bs = []
    for _ in range(n):
        pick = rs.randint(len(G), size=len(G))
        idx = np.concatenate([G[g] for g in pick])
        bs.append(diff[idx].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n_images=len(names), n_patients=len(G), mean_adv=round(float(diff.mean()), 4),
                ci95=[round(float(lo), 4), round(float(hi), 4)])


res = {"note": "tau=1.00 selected on validation; other taus descriptive only"}
ALL = STD + ['Unseen_Rotation_45', 'Unseen_Gaussian_60', 'Unseen_JPEG_10', 'Unseen_Unsharp',
             'Unseen_Speckle_0.05', 'Unseen_MotionBlur_7', 'Cmp_Gaussian_var0.01', 'Cmp_Gaussian_var0.1',
             'Cmp_Rotation_25', 'Cmp_Median_7'] + CLIN
imp_slices = P["E6_distinguish_brats"]
imp_keys = [v for d in P["E3_keytests"] for kname, v in d.items() for v in v] if False else \
    [x for d in P["E3_keytests"] for k, v in d.items() for x in v]
res["impostor_counts"] = dict(other_slice=len(imp_slices), wrong_key=len(imp_keys))
res["impostor_max"] = dict(other_slice=max(imp_slices), wrong_key=max(imp_keys))
res["curve"] = []
for t in TAUS:
    row = dict(tau=t)
    for ds in ["E2_brats", "E5_nih", "E5_isic"]:
        pa = per_attack(P, ds, t)
        row[ds] = dict(clinical=round(np.mean([pa[a] for a in CLIN]), 4),
                       standard=round(np.mean([pa[a] for a in STD]), 4),
                       all34=round(np.mean([pa[a] for a in ALL]), 4))
    row["FAR_other_slice"] = round(accept(imp_slices, t), 4)
    row["FAR_wrong_key"] = round(accept(imp_keys, t), 4)
    res["curve"].append(row)
res["per_attack_accept_tau1_brats"] = per_attack(P, "E2_brats", 1.0)
res["per_attack_accept_tau1_nih"] = per_attack(P, "E5_nih", 1.0)
res["per_attack_accept_tau1_isic"] = per_attack(P, "E5_isic", 1.0)
res["per_attack_accept_tau092_brats_handcrafted"] = per_attack(HC, "E2_brats", 0.92)
res["cluster_bootstrap_vs_focalnet_sel"] = {ds: cluster_boot(ds, STD + CLIN) for ds in ["E2_brats", "E5_nih", "E5_isic"]}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("per_attack")}, indent=1))
for k in ["per_attack_accept_tau1_brats", "per_attack_accept_tau092_brats_handcrafted"]:
    print(k, {a: round(v, 2) for a, v in res[k].items()})
