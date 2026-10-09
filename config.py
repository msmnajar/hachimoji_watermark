# config.py  (v3)
# ─────────────────────────────────────────────────────────────────────────────
# Reference values of the parameters used in the paper. The modules hold the
# same constants next to the code that uses them (core/, watermarking/,
# models/); this file documents them in one place. Values from v1 that were
# wrong (non-chaotic 6D parameters, 1000 warm-up steps, unkeyed SHA-256 key,
# B-Z / P-S base pairs) have been removed.
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# 6D hyperchaotic system of Wang et al. (2019) as used by Wu et al. (2024)
CHAOTIC = {"a": 10.0, "b": 8 / 3, "c": 28.0, "d": -1.0, "e": 8.0, "r": 3.0,
           "rk4_step": 5e-3, "warmup_steps": 10_000, "int_scale": 1e6}

# Hachimoji coding: pairs A-T, G-C, B-S, P-Z; 4! * 2^4 = 384 rules
HACHIMOJI = {"bases": ["A", "T", "G", "C", "B", "Z", "P", "S"], "n_rules": 384,
             "global_bits_per_rule_group": 8,
             "roi_operations": ["add", "xor", "sub", "inv"], "background_operations": ["xor"]}

# ROI map: Otsu on Gaussian-smoothed image, disk dilation; ROI block if >= 50% ROI
ROI = {"gaussian_sigma": 2.5, "dilation_radius_px": 4, "block_roi_fraction": 0.5}

# Zero-watermark
WATERMARK = {"wm_size": (64, 64), "wm_seed": 42, "dwt": "haar", "dwt_level": 2, "ll2_block": 8,
             "arnold_a_b_range": (1, 97), "arnold_rounds_range": (5, 24)}

# Feature extractors
FEATURES = {
    "handcrafted": {"global_bits": 255, "local_bits_per_block": 3},   # v2: DCT signs + SVD/energy bits
    "learned": {"model": "models/aifnet_v3.pt", "global_bits": 256, "local_bits_per_block": 3,
                "cell_px": 32},                                        # v3: AIF-Net
}

# Key binding: k = HMAC-SHA256(K_inst, PatientID|StudyInstanceUID|Modality|SeriesDate)
KEY_BINDING = {"dicom_fields": ["PatientID", "StudyInstanceUID", "Modality", "SeriesDate"],
               "mac": "HMAC-SHA256", "k_inst_bits": 256}

# Two-layer volume chain
VOLUMETRIC = {"robust_seed": "SHA256(MWK_{n-1} || k || n)", "fragile": "SHA256(H_{n-1} || SHA256(I_n))",
              "thresholds": [0.90, 0.95]}

# Kept for evaluation/attacks.py (imported there)
EVAL = {"nc_chance_level": 0.5, "nc_wrong_key_threshold": 0.6}

SEED = 42
