# experiments/make_tables_v4.py  (v4)
"""Writes the LaTeX tables of the paper (results part) and numbers_v4.json from
results_v4/*.json, so that every number in the tables comes from the results files.
Usage: python experiments/make_tables_v4.py <results_dir> <out_dir>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_v4 import STD, CLIN, UNSEEN, CMP, volumes

RD, OD = sys.argv[1], sys.argv[2]
os.makedirs(OD, exist_ok=True)


def L(tag):
    return json.load(open(os.path.join(RD, f"results_v4_{tag}.json")))


def am(rows, k):
    v = [r["attacks"][k] for r in rows if r["attacks"].get(k) is not None]
    return float(np.mean(v)), float(np.min(v))


def m24(R, sk):
    return float(np.mean([am(R[sk], k)[0] for k in STD + CLIN]))


def un(R, sk):
    return float(np.mean([am(R[sk], k)[0] for k in UNSEEN]))


def worst(R, sk):
    k = min(STD + CLIN, key=lambda k: am(R[sk], k)[0])
    return k, am(R[sk], k)[0]


SHORT = {"Rotation_30": "Rot30", "Crop_25%": "Crop25", "Gaussian_40": "G40", "SaltPepper_10%": "SP10",
         "HistEq": "HistEq", "Rotation_10": "Rot10", "Crop_10%": "Crop10"}
SETS = [("E1_sipi", "USC-SIPI"), ("E2_brats", "BraTS"), ("E5_nih", "NIH CXR"), ("E5_isic", "ISIC")]
R, RH = L("learned_v4_q0.7"), L("handcrafted_qall")
N = {}
tex = {}

# ---------------------------------------------------------------- SIPI per image
rows = []
for r in R["E1_sipi"]:
    a = r["attacks"]
    rows.append((r["name"].split("_")[0].capitalize(), a["JPEG_50"], a["Gaussian_20"], a["Median_3"], a["Rotation_10"],
                 r["t_total_dual_ms"]))
body = "\n".join(f"    {n:8s} & 1.0000 & 0 & {j:.4f} & {g:.4f} & {m:.4f} & {rt:.4f} & {t:.1f} \\\\" for n, j, g, m, rt, t in rows)
mean = np.mean([[x[i] for x in rows] for i in range(1, 6)], 1)
body += "\n    \\midrule\n" + f"    Mean     & 1.0000 & 0 & {mean[0]:.4f} & {mean[1]:.4f} & {mean[2]:.4f} & {mean[3]:.4f} & {mean[4]:.1f} \\\\"
tex["sipi"] = body
N["sipi"] = dict(rot10_range=[min(x[4] for x in rows), max(x[4] for x in rows)], t_range=[min(x[5] for x in rows), max(x[5] for x in rows)],
                 jpeg_g_med_min=min(min(x[1], x[2], x[3]) for x in rows))

# ---------------------------------------------------------------- BraTS table
LEFT = [("Gaussian_10", "Gaussian ($\\sigma = 10$)"), ("Gaussian_20", "Gaussian ($\\sigma = 20$)"),
        ("Gaussian_40", "Gaussian ($\\sigma = 40$)"), ("SaltPepper_1%", "Salt-and-pepper (1\\%)"),
        ("SaltPepper_5%", "Salt-and-pepper (5\\%)"), ("SaltPepper_10%", "Salt-and-pepper (10\\%)"),
        ("JPEG_70", "JPEG ($Q = 70$)"), ("JPEG_50", "JPEG ($Q = 50$)"), ("JPEG_30", "JPEG ($Q = 30$)"),
        ("Median_3", "Median ($3\\times3$)"), ("Median_5", "Median ($5\\times5$)")]
RIGHT = [("Rotation_5", "Rotation ($5^\\circ$)"), ("Rotation_10", "Rotation ($10^\\circ$)"),
         ("Rotation_30", "Rotation ($30^\\circ$)"), ("Scale_0.5", "Scaling (0.5)"), ("Scale_0.75", "Scaling (0.75)"),
         ("Crop_10%", "Crop (10\\%)"), ("Crop_25%", "Crop (25\\%)"), ("HistEq", "Histogram equalization"),
         ("PACS_PNG", "PACS PNG export"), ("PACS_JPEG95", "PACS JPEG ($Q = 95$)"),
         ("Downsample_x2", "Tele-radiology ($2\\times$ down-sampling)")]
b = R["E2_brats"]
tex["brats"] = "\n".join(
    f"    {ln:26s} & {am(b, lk)[0]:.4f} & {am(b, lk)[1]:.4f} & {rn:23s} & {am(b, rk)[0]:.4f} & {am(b, rk)[1]:.4f} \\\\"
    for (lk, ln), (rk, rn) in zip(LEFT, RIGHT))
N["brats"] = {k: am(b, k) for k in STD + CLIN + UNSEEN + CMP}

# ---------------------------------------------------------------- collections
COLS = ["JPEG_90", "JPEG_50", "JPEG_30", "Gaussian_10", "Gaussian_20", "SaltPepper_1%", "Median_3", "Rotation_5",
        "Rotation_10", "Crop_10%", "HistEq", "Downsample_x2"]
NM = {"E1_sipi": "USC-SIPI (4)", "E2_brats": "BraTS (20)", "E5_nih": "NIH CXR (20)", "E5_isic": "ISIC (20)"}
tex["collections"] = "\n".join(f"    {NM[sk]:13s} & " + " & ".join(f"{am(R[sk], k)[0]:.4f}" for k in COLS) + " \\\\"
                               for sk, _ in SETS)
N["collections"] = {sn: {k: am(R[sk], k)[0] for k in COLS} for sk, sn in SETS}

# ---------------------------------------------------------------- ablation (5 configurations)
CFG = [("handcrafted_qall", "Handcrafted (DWT--DCT--SVD)"), ("learned_v4_notta_qall", "AIF-Net"),
       ("learned_v4_qall", "AIF-Net + TTA"), ("learned_v4_notta_q0.7", "AIF-Net + selection"),
       ("learned_v4_q0.7", "AIF-Net + TTA + selection (proposed)")]
lines = []
N["ablation"] = {}
for tag, name in CFG:
    X = L(tag)
    d = np.array(X["E6_distinguish_brats"])
    vals = [m24(X, sk) for sk, _ in SETS] + [un(X, "E2_brats"), am(X["E2_brats"], "Rotation_10")[0],
                                             am(X["E2_brats"], "Rotation_30")[0]]
    lines.append(f"    {name} & " + " & ".join(f"{v:.3f}" for v in vals) + f" & {d.mean():.3f} \\\\")
    N["ablation"][tag] = dict(m24={sn: m24(X, sk) for sk, sn in SETS}, unseen=vals[4], rot10=vals[5], rot30=vals[6],
                              dist=(float(d.mean()), float(d.std()), float(d.max())),
                              brats={k: am(X["E2_brats"], k)[0] for k in STD + CLIN + UNSEEN + CMP})
tex["ablation"] = "\n".join(lines)

# ---------------------------------------------------------------- unseen
UL = [("Unseen_Rotation_45", "Rotation $45^\\circ$ (training: up to $35^\\circ$)"),
      ("Unseen_Gaussian_60", "Gaussian noise $\\sigma = 60$ (training: up to 45)"),
      ("Unseen_JPEG_10", "JPEG $Q = 10$ (training: $Q \\geq 15$)"),
      ("Unseen_Unsharp", "Unsharp masking (amount 1.5, $\\sigma = 1$)"),
      ("Unseen_Speckle_0.05", "Speckle noise (variance 0.05)"),
      ("Unseen_MotionBlur_7", "Motion blur (7 pixels)")]
RA = L("learned_v4_notta_qall")
tex["unseen"] = "\n".join(f"    {n:55s} & {am(RH['E2_brats'], k)[0]:.3f} & {am(RA['E2_brats'], k)[0]:.3f} & {am(b, k)[0]:.3f} \\\\" for k, n in UL)
tex["unseen"] += "\n    \\midrule\n" + f"    Mean & {un(RH, 'E2_brats'):.3f} & {un(RA, 'E2_brats'):.3f} & {un(R, 'E2_brats'):.3f} \\\\"

# ---------------------------------------------------------------- TCIA
t = R["E3_tcia"]; kt = R["E3_keytests"]
TL = [("RLE_lossless", None), ("PACS_JPEG95", "Correct key, PACS JPEG ($Q = 95$)"),
      ("Downsample_x2", "Correct key, $2\\times$ down-sampling"), ("Window_header", "Correct key, header window"),
      ("Window_p1-p99", "Correct key, 1st--99th percentile window"), ("Window_narrow50", "Correct key, half-width window")]
lines = ["    Correct key, no attack                     & 1.0000                   & 0/26 \\\\",
         "    Correct key, PACS PNG / RLE                & 1.0000                   & 0/26 \\\\"]
assert all(x["attacks"]["RLE_lossless"] == 1 and x["attacks"]["PACS_PNG"] == 1 for x in t)
N["tcia"] = {}
for k, n in TL[1:]:
    v = np.array([x["attacks"][k] for x in t])
    N["tcia"][k] = (float(v.mean()), float(v.min()), float(v.max()))
    lines.append(f"    {n:42s} & {v.mean():.4f} [{v.min():.4f}, {v.max():.4f}]  & {int((v < 0.6).sum())}/{len(v)} \\\\")
lines.append("    \\midrule")
KL = [("other_patient", "Other patient's identifiers"), ("same_patient_other_study", "Same patient, other StudyUID"),
      ("random_patient_id", "PatientID replaced"), ("wrong_institution_key", "Other institution key"),
      ("k_inst_1bit_flip", "$\\Kinst$ with one bit flipped"), ("x0_plus_1e-15", "$x_0 + 10^{-15}$")]
for k, n in KL:
    v = np.array([s[k][0] for s in kt])
    N["tcia"][k] = (float(v.mean()), float(v.min()), float(v.max()), int((v < 0.6).sum()))
    lines.append(f"    {n:42s} & {v.mean():.4f} [{v.min():.4f}, {v.max():.4f}]  & {int((v < 0.6).sum())}/{len(v)} \\\\")
tex["tcia"] = "\n".join(lines)

# ---------------------------------------------------------------- volumes
TAU = json.load(open(os.path.join(RD, "threshold_selection_v4.json")))["learned_v4"]["tau"]
V = volumes(R, [0.95, TAU])
RV3 = json.load(open(os.path.join(os.path.dirname(RD.rstrip("/")), "results", "results_v3_handcrafted.json")))
VH = volumes(RV3, [0.92, 0.95])
SC = [("clean", "Clean", "no change"), ("benign_png", "PNG export", "no change"),
      ("benign_jpeg90", "JPEG $Q = 90$, all slices", "changed from slice 0"),
      ("benign_jpeg50", "JPEG $Q = 50$, all slices", "changed from slice 0"),
      ("benign_noise5", "Gaussian noise $\\sigma = 5$, all slices", "changed from slice 0"), None,
      ("replace_grey", "Slice 10 constant gray", "slice 10"), ("replace_other_patient", "Slice 10 from another patient", "slice 10"),
      ("duplicate_neighbour", "Slice 10 duplicated from slice 11", "slice 10"),
      ("copy_move_48px", "$48\\times48$ copy--move in slice 10", "slice 10"),
      ("local_smooth_48px", "$48\\times48$ local smoothing", "slice 10"), ("swap_5_12", "Swap of slices 5 and 12", "slice 5"),
      ("delete_slice", "Deletion of slice 10", "slice 10")]
lines = []
for it in SC:
    if it is None:
        lines.append("    \\midrule"); continue
    k, n, fr = it
    den = "200" if k.startswith("benign") or k == "clean" else "10"
    ff = set(V[k]["fragile_first"])
    lines.append(f"    {n:40s} & {VH[k][0.92]}/{den} & {VH[k][0.95]}/{den} & {V[k][0.95]}/{den} & {V[k][TAU]}/{den} & {fr} \\\\")
tex["volumes"] = "\n".join(lines)
N["volumes"] = {k: {str(t): V[k][t] for t in (0.95, TAU)} | {"fragile_first": V[k]["fragile_first"]} for k in V}
N["volumes_hc"] = {k: {str(t): VH[k][t] for t in (0.92, 0.95)} for k in VH}
N["tau"] = TAU
N["reg_ms_per_slice"] = float(np.mean([v["reg_ms_per_slice"] for v in R["E4_volumes"]]))
N["volume_example_nc10"] = R["E4_volumes"][0]["scenarios"]["replace_other_patient"]["nc"][10]

# ---------------------------------------------------------------- same-data comparison
SAME = [("handcrafted_qall", "handcrafted_rel_q0.7", "DWT--DCT--SVD (handcrafted)"),
        ("vgg19_lbp_qall", "vgg19_lbp_q0.7", "VGG19 + LBP (after \\citet{gharib2025})"),
        ("focalnet_qall", "focalnet_q0.7", "FocalNet-T (after \\citet{li2026focalnet})"),
        ("learned_v4_notta_qall", "learned_v4_q0.7", "AIF-Net (proposed)")]
lines = []
N["same"] = {}
for plain, sel, name in SAME:
    for tag, var in ((plain, "all bits"), (sel, "+ selection" if "learned" not in sel else "+ TTA + selection")):
        X = L(tag)
        d = np.array(X["E6_distinguish_brats"])
        v = [m24(X, sk) for sk, _ in SETS[1:]] + [m24(X, "E1_sipi"), un(X, "E2_brats")]
        wk, wv = worst(X, "E2_brats")
        lines.append(f"    {name if var == 'all bits' else ''} & {var} & " + " & ".join(f"{x:.3f}" for x in v)
                     + f" & {wv:.3f} ({SHORT.get(wk, wk)}) & {d.mean():.3f} \\\\")
        N["same"][tag] = dict(m24={sn: m24(X, sk) for sk, sn in SETS}, unseen=v[4], worst=(wk, wv), dist=float(d.mean()),
                              dist_max=float(d.max()), brats={k: am(X["E2_brats"], k)[0] for k in STD + CLIN + UNSEEN + CMP})
    lines.append("    \\midrule")
tex["same"] = "\n".join(lines[:-1])

# ---------------------------------------------------------------- misc numbers
for sk, sn in SETS:
    rows = R[sk]
    N.setdefault("roi", {})[sn] = dict(
        pix=(float(np.mean([r["roi_pixel_fraction"] for r in rows])), float(np.std([r["roi_pixel_fraction"] for r in rows]))),
        cell=(float(np.mean([r["roi_block_fraction"] for r in rows])), float(np.std([r["roi_block_fraction"] for r in rows]))),
        ops=(float(np.mean([r["ops_saving_pct"] for r in rows])), float(np.std([r["ops_saving_pct"] for r in rows]))),
        nfull=float(np.mean([r["n_full"] for r in rows])),
        ks=(float(np.mean([r["t_uniform_postwarm_ms"] for r in rows])), float(np.mean([r["t_dual_postwarm_ms"] for r in rows]))),
        total=(float(np.mean([r["t_total_uniform_ms"] for r in rows])), float(np.mean([r["t_total_dual_ms"] for r in rows]))))
d = np.array(R["E6_distinguish_brats"]); dh = np.array(RH["E6_distinguish_brats"])
N["dist"] = dict(proposed=(float(d.mean()), float(d.std()), float(d.min()), float(d.max()), int((d >= TAU).sum())),
                 handcrafted=(float(dh.mean()), float(dh.std()), float(dh.min()), float(dh.max())),
                 jpeg50_min=min(r["attacks"]["JPEG_50"] for r in R["E2_brats"]))
N["m24"] = {sn: m24(R, sk) for sk, sn in SETS}
N["m24_hc"] = {sn: m24(RH, sk) for sk, sn in SETS}
N["unseen"] = {sn: un(R, sk) for sk, sn in SETS}
N["unseen_hc"] = {sn: un(RH, sk) for sk, sn in SETS}
N["worst"] = {sn: worst(R, sk) for sk, sn in SETS}
N["worst_hc"] = {sn: worst(RH, sk) for sk, sn in SETS}
N["hc_brats"] = {k: am(RH["E2_brats"], k)[0] for k in STD + CLIN + UNSEEN + CMP}
for k, v in tex.items():
    open(os.path.join(OD, f"tab_{k}.tex"), "w").write(v + "\n")
json.dump(N, open(os.path.join(OD, "numbers_v4.json"), "w"), indent=1)
print("tables:", sorted(tex))
