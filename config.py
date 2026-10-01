# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for all thesis experiments.
# Every hyperparameter lives here — never hard-code values in modules.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = Path(r"E:\AAST\Master CS\My Thesis\Reference Papers\papers")
RESULTS_DIR = ROOT / "results"
MODELS_DIR  = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Dataset paths (update to your local BraTS / TCIA / NIH paths) ─────────────
DATASETS = {
    "brats":     Path(r"E:\AAST\Master CS\My Thesis\Datasets\BraTS2023"),
    "tcia":      Path(r"E:\AAST\Master CS\My Thesis\Datasets\TCIA"),
    "chestxray": Path(r"E:\AAST\Master CS\My Thesis\Datasets\NIH_ChestXray14"),
    "isic":      Path(r"E:\AAST\Master CS\My Thesis\Datasets\ISIC2020"),
    "uscsipi":   Path(r"E:\AAST\Master CS\My Thesis\Datasets\USC_SIPI"),  # baseline
}

# ── 6D Hyperchaotic System ────────────────────────────────────────────────────
CHAOTIC = {
    # Default initial values (overridden by biometric key binding at runtime)
    "x0": 0.1, "y0": 0.2, "z0": 0.3,
    "w0": 0.4, "u0": 0.5, "v0": 0.6,
    # System parameters (from Wu et al. 2024 base paper)
    "a": 10.0, "b": 28.0, "c": 8/3,
    "d": 1.0,  "e": 1.0,  "f": 1.0,
    # ODE solver
    "dt": 0.001,
    "warmup_steps": 1000,   # discard transient
    "precision": 1e-15,     # float64 precision for key sensitivity
}

# ── Hachimoji DNA ─────────────────────────────────────────────────────────────
HACHIMOJI = {
    # 8 bases: A, T, G, C (standard) + B, Z, P, S (synthetic)
    "bases":         ["A", "T", "G", "C", "B", "Z", "P", "S"],
    "n_rules":       384,      # total Watson-Crick compliant rules
    "bits_per_base": 3,        # log2(8) — 3 bits per Hachimoji base
    # DNA operations used in diffusion
    "operations":    ["add", "sub", "xor", "inv"],
    # Rule selection mode: "chaotic" (random via chaos) or "adaptive" (CNN-guided)
    "rule_mode":     "chaotic",
}

# ── ROI Segmentation ──────────────────────────────────────────────────────────
ROI = {
    "method":          "unet",    # "unet" | "otsu" | "threshold"
    "unet_model_path": MODELS_DIR / "unet_roi.pth",
    "otsu_blur_ksize": 5,
    "min_roi_fraction": 0.05,    # at least 5% of image must be ROI
    "max_roi_fraction": 0.70,
    "roi_dilation_px":  4,       # pixels to dilate ROI mask boundary
}

# ── Zero-Watermarking ─────────────────────────────────────────────────────────
WATERMARK = {
    # DWT settings
    "dwt_wavelet":    "haar",
    "dwt_level":      2,
    # SVD settings
    "svd_n_singular": 64,        # number of singular values used as features
    # Watermark payload
    "wm_size":        (64, 64),  # watermark image dimensions
    "wm_bits":        64 * 64,   # total bits in watermark
    # Scrambling
    "arnold_rounds":  10,        # Arnold cat map scrambling iterations
    # Role-based keys
    "roles": {
        "radiologist": {"access": "full_roi",    "key_id": 0},
        "nurse":        {"access": "roni_only",   "key_id": 1},
        "admin":        {"access": "metadata",    "key_id": 2},
    },
}

# ── DICOM & Biometric Key Binding ─────────────────────────────────────────────
KEY_BINDING = {
    # Fields used to derive patient-biometric key
    "dicom_fields":    ["PatientID", "StudyInstanceUID", "Modality", "SeriesDate"],
    "hash_algorithm":  "sha256",
    "key_length_bits": 256,
    # Metadata watermark payload fields
    "metadata_fields": ["PatientID", "StudyInstanceUID", "InstitutionName"],
}

# ── 3D Volumetric Chain ───────────────────────────────────────────────────────
VOLUMETRIC = {
    "chain_hash":       "sha256",
    "slice_axis":       2,         # axis along which slices are taken (axial)
    "chain_overlap_px": 0,         # no spatial overlap between slices
}

# ── Evaluation & Metrics ──────────────────────────────────────────────────────
EVAL = {
    # Classical metrics
    "entropy_target":   7.999,
    "npcr_target":      99.6094,   # %
    "uaci_target":      33.4635,   # %
    "correlation_target": 0.005,   # absolute value

    # Medical quality metrics
    "nc_target":        0.97,
    "ber_target":       0.0005,    # 0.05%
    "psnr_target":      35.0,      # dB (for embedding schemes; inf for zero-WM)
    "ssim_target":      0.99,

    # Clinical metrics (thesis-exclusive)
    "diagnostic_delta_target":   0.001,   # < 0.1% classifier confidence drop
    "hausdorff_target_px":       2.0,     # ROI boundary fidelity

    # Attack suite
    "attacks": {
        "gaussian_noise":    {"std": [5, 10, 20, 40]},
        "salt_pepper":       {"density": [0.01, 0.05, 0.1]},
        "jpeg_compression":  {"quality": [90, 70, 50, 30]},
        "median_filter":     {"ksize": [3, 5, 7]},
        "rotation":          {"degrees": [5, 10, 30, 45]},
        "scaling":           {"factor": [0.5, 0.75, 1.25, 2.0]},
        "cropping":          {"fraction": [0.05, 0.10, 0.25]},
        "histogram_eq":      {},
        # Medical-specific attacks (thesis-exclusive)
        "dicom_compression": {"method": ["RLE", "JPEG2000_lossless", "JPEG2000_lossy"]},
        "window_level":      {"ww": [400, 1500], "wl": [-600, 40]},  # HU manipulation
        "pacs_conversion":   {"formats": ["PNG", "JPG", "BMP"]},
        "transcoding":       {"bitrate_kbps": [512, 256, 128]},
    },
}

# ── GPU / Performance ─────────────────────────────────────────────────────────
COMPUTE = {
    "device":         "cuda",    # "cuda" | "cpu"
    "batch_size":     8,
    "num_workers":    4,
    "target_ms_per_slice": 100,  # < 100 ms per 512×512 slice on GPU
}

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
