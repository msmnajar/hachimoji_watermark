# experiments/make_tables_2026.py  (v4)
"""LaTeX table and numbers for the 2026 test collections (results_2026/*.json).
Usage: python experiments/make_tables_2026.py <results_2026_dir> <out_dir>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_v4 import STD, CLIN, UNSEEN, CMP

RD, OD = sys.argv[1], sys.argv[2]
os.makedirs(OD, exist_ok=True)
COLS = [("BrainTumorMRI_Nickparvar", "Brain tumor MRI", "Taj, Niharika"),
        ("CT_Kidney", "Kidney CT", "Niharika"),
        ("BrainMRI_Navoneel", "Brain MRI", "George"),
        ("TCGA_LGG_Buda", "TCGA-LGG FLAIR", "George"),
        ("MIAS", "MIAS mammography", "Suzgen"),
        ("BoneBreak", "Bone X-ray", "Suzgen"),
        ("RetinaFundus", "Retinal fundus", "Suzgen"),
        ("COVID_CXR_Cohen", "COVID chest X-ray", "Anand")]
CFG = ["handcrafted_qall", "handcrafted_rel_q0.7", "vgg19_lbp_qall", "vgg19_lbp_q0.7", "focalnet_qall", "focalnet_q0.7",
       "learned_v4_notta_qall", "learned_v4_qall", "learned_v4_notta_q0.7", "learned_v4_q0.7"]
R = {c: json.load(open(os.path.join(RD, f"results_2026_{c}.json"))) for c in CFG
     if os.path.exists(os.path.join(RD, f"results_2026_{c}.json"))}


def am(rows, k):
    return float(np.mean([r["attacks"][k] for r in rows if r["attacks"].get(k) is not None]))


def m24(rows):
    return float(np.mean([am(rows, k) for k in STD + CLIN]))


N = {}
for c, X in R.items():
    N[c] = {}
    for col, _, _ in COLS:
        rows = X[col]
        d = np.array(X[col + "_distinguish"])
        N[c][col] = dict(m24=m24(rows), unseen=float(np.mean([am(rows, k) for k in UNSEEN])),
                         per={k: am(rows, k) for k in STD + CLIN + UNSEEN + CMP}, dist=float(d.mean()),
                         dist_max=float(d.max()), clean_all_one=all(r["nc_clean"] == 1 for r in rows))
    N[c]["_mean"] = dict(m24=float(np.mean([N[c][col]["m24"] for col, _, _ in COLS])),
                         unseen=float(np.mean([N[c][col]["unseen"] for col, _, _ in COLS])),
                         dist=float(np.mean([N[c][col]["dist"] for col, _, _ in COLS])))
# attack-by-attack wins of the proposed configuration over every re-implemented baseline
P = "learned_v4_q0.7"
wins = {}
for col, _, _ in COLS:
    others = [c for c in R if not c.startswith("learned")]
    keys = STD + CLIN + UNSEEN + CMP
    lost = [(k, max(others, key=lambda c: N[c][col]["per"][k]), max(N[c][col]["per"][k] for c in others), N[P][col]["per"][k])
            for k in keys if max(N[c][col]["per"][k] for c in others) > N[P][col]["per"][k] + 1e-9]
    wins[col] = dict(n=len(keys), best=len(keys) - len(lost), lost=lost)
N["_wins"] = wins

SHOW = [("handcrafted_qall", "HC"), ("vgg19_lbp_q0.7", "VGG"), ("focalnet_q0.7", "FN"), (P, "Prop")]
lines = []
for col, name, used in COLS:
    v = [N[c][col]["m24"] for c, _ in SHOW]
    best = max(v[:-1])
    cells = [f"{x:.3f}" for x in v]
    cells[-1] = "\\textbf{" + cells[-1] + "}" if v[-1] >= best else cells[-1]
    dist = [N[c][col]["dist"] for c in ("vgg19_lbp_q0.7", "focalnet_q0.7", P)]
    lines.append(f"    {name} & {used} & " + " & ".join(cells)
                 + f" & {N[P][col]['unseen']:.3f} & {min(dist[:2]):.3f} & {dist[2]:.3f} \\\\")
mv = [N[c]["_mean"]["m24"] for c, _ in SHOW]
lines.append("    \\midrule")
lines.append("    Mean & & " + " & ".join(f"{x:.3f}" for x in mv[:-1]) + f" & \\textbf{{{mv[-1]:.3f}}} & {N[P]['_mean']['unseen']:.3f} & "
             + f"{min(N['vgg19_lbp_q0.7']['_mean']['dist'], N['focalnet_q0.7']['_mean']['dist']):.3f} & {N[P]['_mean']['dist']:.3f} \\\\")
open(os.path.join(OD, "tab_2026.tex"), "w").write("\n".join(lines) + "\n")
json.dump(N, open(os.path.join(OD, "numbers_2026.json"), "w"), indent=1)
for c in R:
    print(f"{c:24s} mean24 {N[c]['_mean']['m24']:.4f} unseen {N[c]['_mean']['unseen']:.4f} dist {N[c]['_mean']['dist']:.4f}")
for col, w in wins.items():
    print(col, f"proposed best on {w['best']}/{w['n']}", [(k, c, round(b, 3), round(p, 3)) for k, c, b, p in w["lost"]])
