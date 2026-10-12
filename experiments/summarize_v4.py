# experiments/summarize_v4.py  (v4)
"""Prints the numbers used in the paper for every results_v4_*.json file in a folder.
Usage: python experiments/summarize_v4.py <results_dir> [tau_learned]"""
import glob
import json
import os
import sys

import numpy as np

STD = ["Gaussian_5", "Gaussian_10", "Gaussian_20", "Gaussian_40", "SaltPepper_1%", "SaltPepper_5%",
       "SaltPepper_10%", "JPEG_90", "JPEG_70", "JPEG_50", "JPEG_30", "Median_3", "Median_5", "Rotation_5",
       "Rotation_10", "Rotation_30", "Scale_0.5", "Scale_0.75", "Crop_10%", "Crop_25%", "HistEq"]
CLIN = ["PACS_PNG", "PACS_JPEG95", "Downsample_x2"]
UNSEEN = ["Unseen_Rotation_45", "Unseen_Gaussian_60", "Unseen_JPEG_10", "Unseen_Unsharp",
          "Unseen_Speckle_0.05", "Unseen_MotionBlur_7"]
CMP = ["Cmp_Gaussian_var0.01", "Cmp_Gaussian_var0.1", "Cmp_Rotation_25", "Cmp_Median_7"]
SETS = [("E1_sipi", "SIPI"), ("E2_brats", "BraTS"), ("E5_nih", "NIH"), ("E5_isic", "ISIC")]


def mean_attack(rows, k):
    v = [r["attacks"][k] for r in rows if r["attacks"].get(k) is not None]
    return float(np.mean(v)), float(np.min(v))


def summary(R):
    out = {}
    for sk, sn in SETS:
        rows = R[sk]
        per = {k: mean_attack(rows, k) for k in STD + CLIN + UNSEEN + CMP}
        sc = [per[k][0] for k in STD + CLIN]
        worst = min(STD + CLIN, key=lambda k: per[k][0])
        out[sn] = dict(mean24=float(np.mean(sc)), worst=(worst, per[worst][0]),
                       unseen_mean=float(np.mean([per[k][0] for k in UNSEEN])), per=per)
    d = np.array(R["E6_distinguish_brats"])
    out["dist"] = (float(d.mean()), float(d.std()), float(d.max()))
    kt = R["E3_keytests"]
    out["wrongkey_mean"] = float(np.mean([s[k][0] for s in kt for k in s]))
    out["wrongkey_max"] = float(np.max([s[k][0] for s in kt for k in s]))
    out["tcia_min_correct"] = float(min(min(v for v in t["attacks"].values()) for t in R["E3_tcia"]))
    return out


def volumes(R, taus):
    if "E4_volumes" not in R:
        return None
    vols = R["E4_volumes"]
    res = {}
    for scen in vols[0]["scenarios"]:
        row = {}
        for tau in taus:
            if scen.startswith("benign") or scen == "clean":
                row[tau] = sum(sum(x < tau for x in v["scenarios"][scen]["nc"]) for v in vols)
            elif scen == "swap_5_12":
                row[tau] = sum(any(v["scenarios"][scen]["nc"][i] < tau for i in (5, 12)) for v in vols)
            elif scen == "delete_slice":
                row[tau] = sum(any(x < tau for x in v["scenarios"][scen]["nc"][10:]) for v in vols)
            else:
                row[tau] = sum(v["scenarios"][scen]["nc"][10] < tau for v in vols)
        row["fragile_first"] = [v["scenarios"][scen]["fragile_first"] for v in vols]
        res[scen] = row
    return res


if __name__ == "__main__":
    d = sys.argv[1]
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.98
    for f in sorted(glob.glob(os.path.join(d, "results_v4_*.json"))):
        R = json.load(open(f))
        s = summary(R)
        print("=" * 100); print(os.path.basename(f), R.get("config"))
        for sn in ("SIPI", "BraTS", "NIH", "ISIC"):
            x = s[sn]
            print(f"  {sn:6s} mean24 {x['mean24']:.4f}  worst {x['worst'][0]} {x['worst'][1]:.4f}  unseen {x['unseen_mean']:.4f}")
        b = s["BraTS"]["per"]
        print("  BraTS:", {k: round(b[k][0], 4) for k in ["JPEG_50", "JPEG_30", "Gaussian_20", "Gaussian_40", "SaltPepper_1%",
                                                       "SaltPepper_10%", "Median_3", "Rotation_5", "Rotation_10",
                                                       "Rotation_30", "HistEq", "Crop_25%"] + UNSEEN + CMP})
        print(f"  dist {s['dist']}  wrong-key mean {s['wrongkey_mean']:.4f} max {s['wrongkey_max']:.4f}  TCIA min correct {s['tcia_min_correct']:.4f}")
        v = volumes(R, [0.90, 0.95, tau])
        if v:
            for k, row in v.items():
                print("   vol", k, {t: row[t] for t in (0.90, 0.95, tau)}, row["fragile_first"])
