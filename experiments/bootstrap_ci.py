# experiments/bootstrap_ci.py
"""Paired bootstrap 95% confidence intervals (over images) of the per-image
advantage in mean NC over the 24 standard and clinical operations of the
proposed configuration over each re-implemented baseline with selection.

Usage: python experiments/bootstrap_ci.py
Reads results_v4/*.json and results_2026/*.json; 10,000 resamples, seed 0.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from summarize_v4 import STD, CLIN  # noqa: E402

K = STD + CLIN
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def per_image(rows):
    return np.array([np.mean([r["attacks"][k] for k in K]) for r in rows])


def ci(d, rng, n=10000):
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def main():
    rng = np.random.default_rng(0)
    p4 = json.load(open(os.path.join(HERE, "results_v4", "results_v4_learned_v4_q0.7.json")))
    p26 = json.load(open(os.path.join(HERE, "results_2026", "results_2026_learned_v4_q0.7.json")))
    for base in ["focalnet_q0.7", "vgg19_lbp_q0.7", "handcrafted_rel_q0.7"]:
        b4 = json.load(open(os.path.join(HERE, "results_v4", f"results_v4_{base}.json")))
        for key in ["E2_brats", "E5_nih", "E5_isic", "E1_sipi"]:
            d = per_image(p4[key]) - per_image(b4[key])
            lo, hi = ci(d, rng)
            print(f"{base:22s} {key:10s} n={len(d):2d} mean={d.mean():.4f} CI=[{lo:.4f}, {hi:.4f}]")
        b26 = json.load(open(os.path.join(HERE, "results_2026", f"results_2026_{base}.json")))
        for key, rows in p26.items():
            if not (isinstance(rows, list) and rows and isinstance(rows[0], dict) and "attacks" in rows[0]):
                continue
            d = per_image(rows) - per_image(b26[key])
            lo, hi = ci(d, rng)
            print(f"{base:22s} {key[:24]:24s} n={len(d):2d} mean={d.mean():.4f} CI=[{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
