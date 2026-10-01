"""
make_figures_v2.py
==================
Paper figures for revision 2, drawn from results/results_v2.json.
Same visual style (palette, fonts, box and bar styles) as the original
Implementation/Figurs/generate_figures.py; all numbers now come from the
v2 experiments instead of being typed in by hand.

Usage (from the hachimoji_watermark_v2 folder):
    python figures/make_figures_v2.py [path/to/results_v2.json]
Output: figures/paper_figures_v2/Fig1..Fig6 (.pdf and .png), numbered as in the paper
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ── Style (identical to the original generate_figures.py) ────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "axes.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "legend.framealpha": 0.92, "legend.edgecolor": "#AAAAAA",
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
NAVY, BLUE, TEAL = "#003366", "#0066B3", "#008080"
ORANGE, GREEN, RED = "#E87722", "#2E8B57", "#B00032"
LGREY, MGREY = "#F5F5F5", "#CCCCCC"
PURPLE = "#6A3D9A"          # only new colour: ROI-dependent diffusion stage

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results" / "results_v2.json"
OUT = HERE / "paper_figures_v2"
OUT.mkdir(exist_ok=True)
R = json.loads(RES.read_text())

# Lyapunov spectrum (results/lyapunov_spectrum.txt; 1e6 RK4 steps, h = 1e-3)
LYAP = [0.774, 0.312, -0.001, -0.410, -0.644, -14.699]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("  saved", name)


def attack_stats(rows, key):
    v = np.array([r["attacks"][key] for r in rows], float)
    return v.mean(), v.min()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Registration / verification pipeline
# ─────────────────────────────────────────────────────────────────────────────
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 16.9); ax.set_ylim(0, 9.5); ax.axis("off")
    fig.patch.set_facecolor("white")
    W, H = 2.15, 1.25

    def box(x, y, label, sub="", color=BLUE, alpha=0.15, w=W, h=H, fs=10.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    facecolor=color, alpha=alpha, edgecolor=color, lw=1.8))
        # opaque edge on top of the translucent face
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    facecolor="none", edgecolor=color, lw=1.8))
        nl = label.count("\n") + 1
        ax.text(x + w/2, y + h - 0.13 - 0.17*nl, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=color, linespacing=1.05)
        if sub:
            ax.text(x + w/2, y + 0.34, sub, ha="center", va="center",
                    fontsize=fs - 2.3, color="#333333", style="italic", linespacing=1.1)

    def arrow(x1, y1, x2, y2, col=NAVY, ls="-", rad=0.0, lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls,
                                    mutation_scale=14, shrinkA=0, shrinkB=0,
                                    connectionstyle=f"arc3,rad={rad}"))

    def lab(x, y, s, col="#444444", fs=8.3, **kw):
        ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=col,
                bbox=dict(fc="white", ec="none", pad=0.6), **kw)

    X = [0.2, 2.6, 5.0, 7.4, 9.8, 12.2, 14.6]
    yA, yB, yC, yE = 6.95, 5.05, 3.3, 0.3

    # ── GENERATION ───────────────────────────────────────────────────────────
    ax.text(8.45, 9.2, "REGISTRATION PHASE  (host image never modified)", ha="center",
            va="center", fontsize=11.5, fontweight="bold", color=NAVY,
            bbox=dict(fc=LGREY, ec=NAVY, lw=1.2, pad=3))

    # row A: key path + watermark path
    box(X[0], yA, "DICOM header", "PatientID, StudyUID,\nModality, SeriesDate", TEAL)
    box(X[1], yA, "Stage 1\nEncounter key k", "HMAC-SHA256(K_inst, ·)", TEAL)
    box(X[2], yA, "Stage 3\n6D Hyperchaotic", "seed HMAC(k,'pixel')\nRK4, 10⁴ warm-up steps", ORANGE)
    box(X[4], yA, "Watermark W", "64×64 binary logo", GREEN)
    box(X[5], yA, "Stage 6\nArnold scramble", "W′ = Arnold(W; a,b,n)", NAVY)

    # row B: image path
    box(X[0], yB, "DICOM image I", "8-bit rendering\nof pixel data", BLUE)
    box(X[1], yB, "Stage 2\nFeatures f", "2-level Haar DWT: 255\nDCT signs + 3 bits/block", BLUE)
    box(X[2], yB, "Stage 4\nHachimoji DNA", "t = ℓ or 7−ℓ (by f)\nβ = R_r[t], 384 rules", RED)
    box(X[3], yB, "Stage 5\nROI diffusion", "ROI: add→xor→sub→inv\nbackground: xor", PURPLE)
    box(X[4], yB, "Stage 7\nMWK binding", "parity(c) ⊕ W′\n4096 bits", TEAL)
    box(X[5], yB, "Stage 8\nTrusted authority", "stores record\nM = (MWK, c_ROI)", GREEN)

    # row C: ROI
    box(X[1], yC, "Stage 2b\nROI map", "Otsu + 4-px dilation;\nROI block if ≥50% ROI", BLUE)

    # arrows row A
    arrow(X[0] + W, yA + H/2, X[1], yA + H/2, TEAL)
    arrow(X[1] + W, yA + H/2, X[2], yA + H/2, TEAL)
    arrow(X[4] + W, yA + H/2, X[5], yA + H/2, GREEN)
    # key -> Arnold parameters (routed over the top)
    ya = yA + H + 0.42
    ax.plot([X[1] + W/2, X[1] + W/2, X[5] + W/2], [yA + H + 0.1, ya, ya], color=NAVY, lw=1.4, ls="--")
    arrow(X[5] + W/2, ya, X[5] + W/2, yA + H + 0.1, NAVY, ls="--", lw=1.4)
    lab(8.45, ya, "Arnold parameters (a, b, n) = HMAC(k, 'arnold')", col=NAVY, fs=8.5)

    # arrows row B
    arrow(X[0] + W, yB + H/2, X[1], yB + H/2, BLUE)
    arrow(X[1] + W, yB + H/2, X[2], yB + H/2, BLUE)
    arrow(X[2] + W, yB + H/2, X[3], yB + H/2, RED)
    arrow(X[3] + W, yB + H/2, X[4], yB + H/2, PURPLE)
    arrow(X[4] + W, yB + H/2, X[5], yB + H/2, TEAL)
    arrow(X[5] + W/2, yA, X[4] + W/2 + 0.35, yB + H, NAVY)     # W' -> MWK
    lab(X[5] + 0.05, yA - 0.42, "W′", col=NAVY, fs=9)

    # chaos -> coding / diffusion
    arrow(X[2] + W/2, yA, X[2] + W/2, yB + H, ORANGE, ls="--")
    lab(X[2] + W/2 - 0.02, (yA + yB + H) / 2, "positions p, rules r,\ntribits ℓ", col=ORANGE, fs=7.6)
    arrow(X[2] + W - 0.15, yA + 0.05, X[3] + 0.35, yB + H, ORANGE, ls="--")
    lab(X[3] + 0.25, (yA + yB + H) / 2 + 0.05, "k₁, k₂, k₃", col=ORANGE, fs=7.6)

    # image -> ROI ; ROI -> diffusion and -> record
    ax.annotate("", xy=(X[1], yC + H/2), xytext=(X[0] + W/2, yB),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=13,
                                connectionstyle="angle,angleA=-90,angleB=180,rad=0"))
    ax.annotate("", xy=(X[3] + W/2, yB), xytext=(X[1] + W, yC + H/2),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4, mutation_scale=13,
                                connectionstyle="angle,angleA=0,angleB=-90,rad=0"))
    lab((X[1] + W + X[3] + W/2) / 2 + 0.2, yC + H/2, "block flags c_ROI", col=PURPLE, fs=8.5)

    # ── VERIFICATION ─────────────────────────────────────────────────────────
    ax.text(8.45, 2.45, "VERIFICATION PHASE  (same key re-derived from the DICOM header of the received image)",
            ha="center", va="center", fontsize=10.5, color=RED,
            bbox=dict(fc="#FFF5F5", ec=RED, lw=1.1, pad=2))
    box(X[0], yE, "Received image Î", "possibly attacked\n+ its DICOM header", BLUE, 0.10)
    box(X[1], yE, "Re-derive key", "same HMAC with K_inst\n→ same chaotic stream", TEAL, 0.10)
    box(X[2], yE, "Features f̂", "same DWT / DCT /\nSVD descriptors", BLUE, 0.10)
    box(X[3], yE, "Coding + diffusion", "Stages 4–5 with\nstored c_ROI", RED, 0.10)
    box(X[4], yE, "Recover Ŵ", "MWK ⊕ parity(ĉ),\ninverse Arnold", TEAL, 0.10)
    box(X[5], yE, "Decision", "NC(W, Ŵ) ≥ τ\n→ authentic", GREEN, 0.10)
    box(X[6], yE, "Volumes", "robust seed chain\n+ fragile SHA-256\nhash chain", NAVY, 0.10)
    for i in range(6):
        arrow(X[i] + W, yE + H/2, X[i + 1], yE + H/2)
    # record from the authority down to the recovery step
    xr, yk = X[5] + W/2, 1.95
    ax.plot([xr, xr, X[4] + W - 0.3], [yB, yk, yk], color=GREEN, lw=1.5, ls="--")
    arrow(X[4] + W - 0.3, yk, X[4] + W - 0.3, yE + H, GREEN, ls="--", lw=1.5)
    lab(xr, 3.6, "record M", col=GREEN, fs=8.5, fontweight="bold")

    # legend of line styles
    handles = [Line2D([0], [0], color=NAVY, lw=1.6, label="data flow"),
               Line2D([0], [0], color=ORANGE, lw=1.6, ls="--", label="key / chaotic stream"),
               Line2D([0], [0], color=GREEN, lw=1.6, ls="--", label="stored record")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.405), fontsize=8.5,
              frameon=True, ncol=1)
    save(fig, "Fig1_Architecture")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Attack robustness on BraTS
# ─────────────────────────────────────────────────────────────────────────────
def fig2_attack_nc():
    rows = R["E2_brats"]
    std = [("Gaussian_5", "Gauss\nσ=5"), ("Gaussian_10", "Gauss\nσ=10"), ("Gaussian_20", "Gauss\nσ=20"),
           ("SaltPepper_1%", "S&P\n1%"), ("JPEG_90", "JPEG\nQ90"), ("JPEG_50", "JPEG\nQ50"),
           ("JPEG_30", "JPEG\nQ30"), ("Median_3", "Median\n3×3"), ("Median_5", "Median\n5×5"),
           ("Rotation_5", "Rot\n5°"), ("Rotation_10", "Rot\n10°"), ("Crop_10%", "Crop\n10%"),
           ("Crop_25%", "Crop\n25%"), ("HistEq", "Hist.\neq.")]
    clin = [("PACS_PNG", "PACS\nPNG†"), ("PACS_JPEG95", "PACS\nJ95†"), ("Downsample_x2", "Tele\n2×†")]
    keys = std + clin
    mean_min = [attack_stats(rows, k) for k, _ in keys]
    nc_vals = [m for m, _ in mean_min]
    err = [m - mn for m, mn in mean_min]
    x = np.arange(len(keys))
    colors = [BLUE] * len(std) + [ORANGE] * len(clin)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    bars = ax.bar(x, nc_vals, width=0.62, color=colors, alpha=0.82, edgecolor="white",
                  linewidth=0.8, zorder=3)
    ax.errorbar(x, nc_vals, yerr=[err, np.zeros_like(err)], fmt="none", ecolor="#333333",
                elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
    ax.axhline(0.90, color=RED, ls="--", lw=1.8, zorder=5)
    ax.axvline(len(std) - 0.5, color="#888888", ls=":", lw=1.5, zorder=2)
    ax.text((len(std) - 1) / 2, 0.622, "Standard attacks", ha="center", va="center", fontsize=9.5,
            color=NAVY, fontweight="bold", bbox=dict(fc="white", ec=NAVY, lw=0.8, pad=2))
    ax.text(len(std) + 1, 0.622, "Clinical pipeline †", ha="center", va="center", fontsize=9.5,
            color=ORANGE, fontweight="bold", bbox=dict(fc="white", ec=ORANGE, lw=0.8, pad=2))
    for b, v in zip(bars, nc_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}", ha="center", va="bottom",
                fontsize=7.4, color="#222222", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], fontsize=8.8)
    ax.set_ylim(0.60, 1.07)
    ax.set_yticks(np.arange(0.60, 1.01, 0.05))
    ax.set_ylabel("Normalized Correlation (NC)", color=NAVY)
    ax.set_xlabel("Attack Type", color=NAVY)
    ax.set_title(f"Attack Robustness on BraTS 2023 (n = {len(rows)} slices; bars: mean, whiskers: minimum)",
                 fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.yaxis.grid(True, ls="--", alpha=0.5, zorder=0)
    ax.set_facecolor(LGREY)
    ax.legend(handles=[mpatches.Patch(fc=BLUE, alpha=0.82, label="Standard attacks"),
                       mpatches.Patch(fc=ORANGE, alpha=0.82, label="Clinical pipeline† (PACS export, tele-radiology)"),
                       Line2D([0], [0], color=RED, lw=1.8, ls="--", label="Detection threshold τ = 0.90")],
              loc="lower left", bbox_to_anchor=(0.0, 0.06), fontsize=9)
    save(fig, "Fig2_AttackNC")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Reported robustness of related zero-watermarking schemes (radar)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_radar():
    rows = R["E2_brats"]
    cats = ["NC\nJPEG", "NC\nGaussian noise", "NC\nrotation", "NC\nmedian 3×3"]
    prop = [attack_stats(rows, k)[0] for k in ("JPEG_50", "Gaussian_20", "Rotation_5", "Median_3")]
    # values as reported by each paper under its own settings (Table 4 of the paper)
    methods = {
        "Proposed (BraTS; Q50, σ=20, 5°)": (prop, BLUE, "o-", 2.5),
        "Gharib 2025 (Q50, var 0.1, 10°)": ([1.0000, 1.0000, 0.9993, 0.9987], ORANGE, "s--", 2.0),
        "Shi 2023 (Q30, var 0.01, 5°)": ([0.9900, 0.9651, 0.9875, 0.9949], TEAL, "^:", 1.8),
        "Taj 2024 (Q50, 20%, 10°)": ([0.996, 0.990, 0.990, 1.000], GREEN, "D-.", 1.6),
    }
    N = len(cats)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7.4, 7.2), subplot_kw=dict(polar=True))
    for name, (vals, col, sty, lw) in methods.items():
        v = list(vals) + [vals[0]]
        ax.plot(ang, v, sty, color=col, lw=lw, label=name, markersize=6)
        ax.fill(ang, v, alpha=0.08, color=col)
    for a, v in zip(ang[:-1], prop):
        ax.text(a, v - 0.012, f"{v:.3f}", ha="center", va="center", fontsize=8.5, color=BLUE,
                fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(cats, size=10, color=NAVY)
    ax.set_ylim(0.85, 1.005)
    ax.set_yticks([0.88, 0.92, 0.96, 1.00])
    ax.set_yticklabels(["0.88", "0.92", "0.96", "1.00"], size=8, color="#666666")
    ax.yaxis.grid(True, ls="--", alpha=0.4); ax.xaxis.grid(True, ls="-", alpha=0.3)
    ax.spines["polar"].set_color(MGREY)
    ax.set_title("Robustness as Reported by Each Study\n(own images and attack settings; not a controlled benchmark)",
                 fontsize=11, fontweight="bold", color=NAVY, pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8.5, framealpha=0.9)
    save(fig, "Fig5_Comparison_Radar")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — ROI segmentation and measured saving of the ROI-dependent path
# ─────────────────────────────────────────────────────────────────────────────
def fig4_saving():
    rows = R["E2_brats"]
    names = [r["name"].replace("BraTS-", "") for r in rows]
    pix = np.array([r["roi_pixel_fraction"] for r in rows])
    blk = np.array([r["roi_block_fraction"] for r in rows])
    ops = np.array([r["ops_saving_pct"] for r in rows])
    tpw = np.array([r["saving_postwarm_pct"] for r in rows])
    tot = np.array([100 * (1 - r["t_total_dual_ms"] / r["t_total_uniform_ms"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1.05, 1]))
    fig.suptitle("ROI Segmentation and Measured Saving of the ROI-Dependent Diffusion",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.01)

    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.19, pix, height=0.38, color=BLUE, alpha=0.8, edgecolor="white", label="ROI pixel fraction")
    ax.barh(y - 0.19, blk, height=0.38, color=ORANGE, alpha=0.8, edgecolor="white", label="ROI block fraction (LL2)")
    ax.axvline(pix.mean(), color=RED, ls="--", lw=1.8, label=f"Mean pixel fraction = {pix.mean():.3f}")
    ax.axvline(blk.mean(), color=RED, ls=":", lw=1.8, label=f"Mean block fraction = {blk.mean():.3f}")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.8)
    ax.set_xlim(0, 0.80)
    ax.set_xlabel("Fraction of image", color=NAVY)
    ax.set_title("ROI Fraction per BraTS Slice", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.xaxis.grid(True, ls="--", alpha=0.4); ax.set_facecolor(LGREY)
    ax.legend(fontsize=8.3, loc="lower right")

    ax2 = axes[1]
    data = [ops, tpw, tot]
    labels = ["Diffusion\noperations", "Key-stream time\n(after warm-up)", "Total registration\ntime"]
    cols = [TEAL, BLUE, NAVY]
    xs = np.arange(3)
    means = [d.mean() for d in data]; sds = [d.std() for d in data]
    ax2.bar(xs, means, width=0.55, color=cols, alpha=0.78, edgecolor="white", zorder=3)
    ax2.errorbar(xs, means, yerr=sds, fmt="none", ecolor="#333333", capsize=5, elinewidth=1.2, zorder=4)
    rng = np.random.default_rng(0)
    for xi, d in zip(xs, data):
        ax2.scatter(xi + rng.uniform(-0.17, 0.17, d.size), d, s=14, color="white", edgecolor="#333333",
                    lw=0.7, zorder=5)
    for xi, m, s in zip(xs, means, sds):
        ax2.text(xi + 0.31, m, f"{m:.1f}%\n±{s:.1f}", ha="left", va="center", fontsize=9,
                 fontweight="bold", color=NAVY)
    ax2.axhline(0, color="#555555", lw=1.0)
    ax2.set_xticks(xs); ax2.set_xticklabels(labels, fontsize=9.3)
    ax2.set_xlim(-0.5, 2.85)
    ax2.set_ylabel("Saving vs. full strength everywhere (%)", color=NAVY)
    ax2.set_title(f"Measured Saving (n = {len(rows)} BraTS slices; mean ± SD)", fontsize=10.5,
                  fontweight="bold", color=NAVY)
    ax2.set_facecolor(LGREY); ax2.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    nfull = np.mean([r["n_full"] for r in rows])
    ax2.text(0.98, 0.97, f"{nfull:.0f} of 4096 positions use\nthe four-operation ROI path",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9, color=NAVY,
             bbox=dict(fc="white", ec=NAVY, pad=3))
    plt.tight_layout()
    save(fig, "Fig3_ROI_Saving")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Two-layer volumetric verification
# ─────────────────────────────────────────────────────────────────────────────
def fig5_chain():
    vols = R["E4_volumes"]
    v0 = vols[0]
    tam = v0["scenarios"]["replace_other_patient"]["nc"]
    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.12)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 14); ax.set_ylim(0, 5.2); ax.axis("off")

    ax.text(7, 4.9, "Two-Layer Chain — Slice 10 Replaced by Another Patient's Slice "
                    f"({v0['case'].replace('BraTS-', '').rsplit('-', 1)[0]})",
            ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY,
            bbox=dict(fc=LGREY, ec=NAVY, lw=1.2, pad=4))
    idx = [8, 9, 10, 11, 12]
    xs = [0.95, 3.25, 5.55, 7.85, 10.15]
    ax.text(0.05, 3.25, "Robust\nlayer", ha="left", va="center", fontsize=9.5, fontweight="bold", color=TEAL)
    ax.text(0.05, 1.05, "Fragile\nlayer", ha="left", va="center", fontsize=9.5, fontweight="bold", color=NAVY)
    for n, x in zip(idx, xs):
        bad = n == 10
        col = RED if bad else TEAL
        ax.add_patch(FancyBboxPatch((x, 2.55), 1.9, 1.4, boxstyle="round,pad=0.08", fc=col, alpha=0.18,
                                    ec=col, lw=2.5 if bad else 1.5))
        ax.text(x + 0.95, 3.72, f"Slice {n}" + ("  ★" if bad else ""), ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=col)
        ax.text(x + 0.95, 3.30, f"seed s{n} =\nSHA256(MWK{n-1}‖k‖{n})", ha="center", va="center",
                fontsize=7.2, color="#444444")
        ncv = tam[n]
        nc_col = GREEN if ncv >= 0.95 else RED
        ax.add_patch(FancyBboxPatch((x + 0.25, 2.62), 1.4, 0.36, boxstyle="round,pad=0.04", fc=nc_col,
                                    alpha=0.15, ec=nc_col, lw=1.2))
        ax.text(x + 0.95, 2.80, f"NC = {ncv:.4f}", ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=nc_col)
        # fragile box
        fcol = RED if n >= 10 else NAVY
        ax.add_patch(FancyBboxPatch((x, 0.55), 1.9, 1.0, boxstyle="round,pad=0.08", fc=fcol, alpha=0.10,
                                    ec=fcol, lw=1.5))
        ax.text(x + 0.95, 1.25, f"H{n} = SHA256(H{n-1}‖\nSHA256(pixels{n}))", ha="center", va="center",
                fontsize=7.2, color="#444444")
        ax.text(x + 0.95, 0.75, "mismatch" if n >= 10 else "match", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=fcol)
        ax.annotate("", xy=(x + 0.95, 1.63), xytext=(x + 0.95, 2.47),
                    arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0, ls=":", mutation_scale=10))
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.08, 3.25), xytext=(xs[i] + 1.98, 3.25),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.5, mutation_scale=14))
        c2 = RED if idx[i + 1] >= 10 else NAVY
        ax.annotate("", xy=(xs[i + 1] - 0.08, 1.05), xytext=(xs[i] + 1.98, 1.05),
                    arrowprops=dict(arrowstyle="-|>", color=c2, lw=1.5, mutation_scale=14))
    ax.text(13.05, 3.25, "record MWK\nfrom authority:\nonly slice 10\nfails", ha="center", va="center",
            fontsize=8, color=TEAL, style="italic")
    ax.text(13.05, 1.05, "chained hash:\nbreaks from\nslice 10 on", ha="center", va="center",
            fontsize=8, color=NAVY, style="italic")

    # bottom: per-slice NC averaged over the 10 volumes
    ax2 = fig.add_subplot(gs[1])
    scen = [("benign_jpeg50", "JPEG Q50, all slices (benign)", GREEN, "o-"),
            ("benign_noise5", "Gaussian σ=5, all slices (benign)", TEAL, "s-"),
            ("replace_other_patient", "Slice 10 from another patient", RED, "D-"),
            ("duplicate_neighbour", "Slice 10 duplicated from 11", ORANGE, "^-"),
            ("copy_move_48px", "48×48 copy–move in slice 10", PURPLE, "v-"),
            ("local_smooth_48px", "48×48 local smoothing in slice 10", "#888888", "x--"),
            ("swap_5_12", "Slices 5 and 12 swapped", BLUE, "P-")]
    for key, label, col, sty in scen:
        a = np.array([v["scenarios"][key]["nc"] for v in vols])
        ax2.plot(np.arange(a.shape[1]), a.mean(0), sty, color=col, lw=1.6, ms=5, label=label, alpha=0.9)
    ax2.axhline(0.95, color=RED, ls="--", lw=1.6)
    ax2.axhline(0.90, color=RED, ls=":", lw=1.6)
    ax2.text(19.4, 0.953, "τ = 0.95", color=RED, fontsize=9, ha="right", va="bottom")
    ax2.text(19.4, 0.903, "τ = 0.90", color=RED, fontsize=9, ha="right", va="bottom")
    ax2.set_xticks(range(20)); ax2.set_xlim(-0.4, 19.4); ax2.set_ylim(0.62, 1.015)
    ax2.set_xlabel("Slice index", color=NAVY); ax2.set_ylabel("Robust-layer NC (mean of 10 volumes)", color=NAVY)
    ax2.set_title("Per-Slice NC for Benign Processing and Tampering (10 BraTS volumes × 20 slices)",
                  fontsize=10.5, fontweight="bold", color=NAVY)
    ax2.set_facecolor(LGREY); ax2.yaxis.grid(True, ls="--", alpha=0.4)
    ax2.legend(fontsize=8.3, loc="lower left", ncol=2, framealpha=0.95)
    save(fig, "Fig4_Volumetric")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Security analysis
# ─────────────────────────────────────────────────────────────────────────────
def fig6_security():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[0.9, 1.25, 1.1]))
    fig.suptitle("Security Analysis: Chaotic Key Stream, Key Sensitivity and Distinguishability",
                 fontsize=12, fontweight="bold", color=NAVY, y=1.02)

    # (a) Lyapunov spectrum
    cols = [ORANGE if l > 0.01 else (MGREY if abs(l) <= 0.01 else "#A8C8E8") for l in LYAP]
    lab = [f"λ{i+1}" for i in range(6)]
    b = a1.bar(lab, LYAP, color=cols, edgecolor=NAVY, lw=1.3, alpha=0.9, zorder=3)
    a1.set_yscale("symlog", linthresh=1.0)
    a1.axhline(0, color="#333333", lw=1.0)
    for bb, l in zip(b, LYAP):
        a1.text(bb.get_x() + bb.get_width() / 2, l + (0.08 if l >= 0 else -0.08), f"{l:+.3f}",
                ha="center", va="bottom" if l >= 0 else "top", fontsize=8.6, fontweight="bold", color=NAVY)
    a1.set_ylim(-40, 3)
    a1.set_yticks([-10, -1, 0, 1]); a1.set_yticklabels(["−10", "−1", "0", "1"])
    a1.set_ylabel("Lyapunov exponent (symlog)", color=NAVY)
    a1.set_title("(a) Lyapunov Spectrum of the 6D System\ntwo positive exponents → hyperchaotic",
                 fontsize=10.5, fontweight="bold", color=NAVY)
    a1.set_facecolor(LGREY); a1.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)

    # (b) key sensitivity on TCIA
    kt = R["E3_keytests"]
    conds = [("other_patient", "Other\npatient"), ("same_patient_other_study", "Same patient,\nother study"),
             ("random_patient_id", "PatientID\nreplaced"), ("wrong_institution_key", "Other\ninstitution key"),
             ("k_inst_1bit_flip", "K_inst\n1-bit flip"), ("x0_plus_1e-15", "x₀ + 10⁻¹⁵")]
    tcia = R["E3_tcia"]
    correct = [("PACS_JPEG95", "Correct key,\nJPEG Q95")]
    vals = [np.array([t["attacks"]["PACS_JPEG95"] for t in tcia])]
    vals += [np.array([s[k][0] for s in kt]) for k, _ in conds]
    names = [c[1] for c in correct] + [c[1] for c in conds]
    colc = [BLUE] + [MGREY, "#A8C8E8", "#A8C8E8", TEAL, ORANGE, ORANGE]
    m = [v.mean() for v in vals]
    lo = [v.mean() - v.min() for v in vals]; hi = [v.max() - v.mean() for v in vals]
    xs = np.arange(len(vals))
    bars = a2.bar(xs, m, color=colc, edgecolor=NAVY, lw=1.3, alpha=0.85, zorder=3, width=0.65)
    a2.errorbar(xs, m, yerr=[lo, hi], fmt="none", ecolor="#333333", capsize=4, elinewidth=1.1, zorder=4)
    for bb, v in zip(bars, m):
        a2.text(bb.get_x() + bb.get_width() / 2, v + 0.045, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.8, fontweight="bold", color=NAVY)
    a2.axhline(0.5, color=ORANGE, lw=1.3, ls=":", label="Chance level (NC ≈ 0.5)")
    a2.axhline(0.90, color=RED, lw=1.5, ls="--", label="Threshold τ = 0.90")
    a2.set_xticks(xs); a2.set_xticklabels(names, fontsize=8.2)
    a2.set_ylim(0, 1.15); a2.set_ylabel("NC of extracted watermark", color=NAVY)
    a2.set_title(f"(b) Key Sensitivity on TCIA ({len(kt)} slices; whiskers: min–max)",
                 fontsize=10.5, fontweight="bold", color=NAVY)
    a2.set_facecolor(LGREY); a2.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    a2.legend(fontsize=8.3, loc="upper right")

    # (c) distinguishability
    cross = np.array(R["E6_distinguish_brats"])
    same = np.array([r["attacks"]["JPEG_50"] for r in R["E2_brats"]])
    bins = np.arange(0.55, 1.005, 0.015)
    a3.hist(cross, bins=bins, color=TEAL, alpha=0.75, edgecolor="white", zorder=3,
            label=f"Other slice, same key ({cross.size} pairs)")
    a3.hist(same, bins=bins, color=BLUE, alpha=0.85, edgecolor="white", zorder=3,
            label=f"Same slice after JPEG Q50 ({same.size})")
    a3.axvline(cross.mean(), color=TEAL, ls="--", lw=1.6)
    a3.axvline(0.90, color=RED, ls="--", lw=1.6, label="τ = 0.90")
    a3.text(cross.mean() - 0.045, a3.get_ylim()[1] * 0.80, f"mean {cross.mean():.3f}\n±{cross.std():.3f}",
            ha="right", va="top", fontsize=8.8, color=TEAL, fontweight="bold")
    a3.set_xlabel("NC", color=NAVY); a3.set_ylabel("Count", color=NAVY)
    a3.set_title("(c) Distinguishability on BraTS", fontsize=10.5, fontweight="bold", color=NAVY)
    a3.set_facecolor(LGREY); a3.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    a3.legend(fontsize=8.2, loc="upper right")
    plt.tight_layout()
    save(fig, "Fig6_Security")


if __name__ == "__main__":
    print("Generating v2 paper figures from", RES)
    fig1_architecture(); fig2_attack_nc(); fig3_radar()
    fig4_saving(); fig5_chain(); fig6_security()
    print("Done:", OUT)
