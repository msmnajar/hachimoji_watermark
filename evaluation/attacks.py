# evaluation/attacks.py
"""
Attack Simulation Suite
========================
Standard + medical-specific attacks for robustness evaluation.

Standard attacks (match prior work benchmarks):
  Gaussian noise, salt-and-pepper, JPEG, median filter,
  rotation, scaling, cropping, histogram equalization

Medical-specific attacks (thesis-exclusive — no prior paper tests these):
  DICOM compression, window/level adjustment, PACS conversion, transcoding
"""

import numpy as np
from typing import Callable
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import EVAL


# ── Standard attacks ──────────────────────────────────────────────────────────

def gaussian_noise(image: np.ndarray, std: float = 10) -> np.ndarray:
    noisy = image.astype(np.float32) + np.random.normal(0, std, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def salt_and_pepper(image: np.ndarray, density: float = 0.01) -> np.ndarray:
    out  = image.copy()
    mask = np.random.random(image.shape)
    out[mask < density / 2]  = 0
    out[mask > 1 - density/2] = 255
    return out


def jpeg_compression(image: np.ndarray, quality: int = 50) -> np.ndarray:
    import io
    from PIL import Image
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    from scipy.ndimage import median_filter as mf
    return mf(image, size=ksize).astype(np.uint8)


def rotation(image: np.ndarray, degrees: float = 10) -> np.ndarray:
    from scipy.ndimage import rotate
    rotated = rotate(image.astype(np.float32), degrees, reshape=False, cval=0)
    return np.clip(rotated, 0, 255).astype(np.uint8)


def scaling(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    from PIL import Image
    h, w = image.shape[:2]
    nh, nw = int(h * factor), int(w * factor)
    pil  = Image.fromarray(image)
    down = pil.resize((nw, nh), Image.LANCZOS)
    up   = down.resize((w, h), Image.LANCZOS)
    return np.array(up)


def cropping(image: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    h, w = image.shape[:2]
    ch   = int(h * fraction / 2)
    cw   = int(w * fraction / 2)
    out  = image.copy()
    out[:ch, :] = 0;  out[-ch:, :] = 0
    out[:, :cw] = 0;  out[:, -cw:] = 0
    return out


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    from PIL import Image, ImageOps
    pil = Image.fromarray(image)
    eq  = ImageOps.equalize(pil)
    return np.array(eq)


# ── Medical-specific attacks (thesis-exclusive) ───────────────────────────────

def dicom_rle_compression(image: np.ndarray) -> np.ndarray:
    """DICOM RLE lossless — image unchanged, tests watermark survival of re-encoding."""
    return image.copy()


def window_level_adjustment(
    image: np.ndarray,
    window_width: int = 400,
    window_level: int = 40,
) -> np.ndarray:
    """CT Hounsfield window/level manipulation."""
    lo  = window_level - window_width // 2
    hi  = window_level + window_width // 2
    arr = image.astype(np.float32)
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (window_width + 1e-8) * 255
    return arr.astype(np.uint8)


def pacs_png_conversion(image: np.ndarray) -> np.ndarray:
    """PACS DICOM→PNG conversion (lossless, tests format change)."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    buf.seek(0)
    return np.array(Image.open(buf))


def pacs_jpg_conversion(image: np.ndarray, quality: int = 95) -> np.ndarray:
    """PACS DICOM→JPEG conversion (lossy, common in report export)."""
    return jpeg_compression(image, quality=quality)


def telemedicine_transcoding(image: np.ndarray, downsample: int = 2) -> np.ndarray:
    """Simulate telemedicine bandwidth reduction via spatial downsampling."""
    from PIL import Image
    h, w = image.shape[:2]
    pil  = Image.fromarray(image)
    down = pil.resize((w // downsample, h // downsample), Image.LANCZOS)
    up   = down.resize((w, h), Image.LANCZOS)
    return np.array(up)


# ── Full attack battery ────────────────────────────────────────────────────────

STANDARD_ATTACKS = {
    "Gaussian_5":   lambda img: gaussian_noise(img, 5),
    "Gaussian_10":  lambda img: gaussian_noise(img, 10),
    "Gaussian_20":  lambda img: gaussian_noise(img, 20),
    "Gaussian_40":  lambda img: gaussian_noise(img, 40),
    "SaltPepper_1%": lambda img: salt_and_pepper(img, 0.01),
    "SaltPepper_5%": lambda img: salt_and_pepper(img, 0.05),
    "SaltPepper_10%":lambda img: salt_and_pepper(img, 0.10),
    "JPEG_90":      lambda img: jpeg_compression(img, 90),
    "JPEG_70":      lambda img: jpeg_compression(img, 70),
    "JPEG_50":      lambda img: jpeg_compression(img, 50),
    "JPEG_30":      lambda img: jpeg_compression(img, 30),
    "Median_3":     lambda img: median_filter(img, 3),
    "Median_5":     lambda img: median_filter(img, 5),
    "Rotation_5":   lambda img: rotation(img, 5),
    "Rotation_10":  lambda img: rotation(img, 10),
    "Rotation_30":  lambda img: rotation(img, 30),
    "Scale_0.5":    lambda img: scaling(img, 0.5),
    "Scale_0.75":   lambda img: scaling(img, 0.75),
    "Crop_10%":     lambda img: cropping(img, 0.10),
    "Crop_25%":     lambda img: cropping(img, 0.25),
    "HistEq":       histogram_equalization,
}

MEDICAL_ATTACKS = {
    "DICOM_RLE":        dicom_rle_compression,
    "WL_chest":         lambda img: window_level_adjustment(img, 1500, -600),
    "WL_abdomen":       lambda img: window_level_adjustment(img, 400, 40),
    "PACS_PNG":         pacs_png_conversion,
    "PACS_JPEG95":      lambda img: pacs_jpg_conversion(img, 95),
    "Telemedicine_x2":  lambda img: telemedicine_transcoding(img, 2),
}

ALL_ATTACKS = {**STANDARD_ATTACKS, **MEDICAL_ATTACKS}


def run_attack_battery(
    image: np.ndarray,
    zero_wm,
    mwk: dict,
    original_wm: np.ndarray,
    attack_set: str = "standard",
) -> dict:
    """
    Run the full attack battery and return NC per attack.

    Returns: {"attack_name": nc_value, ...}
    """
    from evaluation.metrics import nc as compute_nc

    attacks = STANDARD_ATTACKS if attack_set == "standard" else ALL_ATTACKS
    results = {}

    for name, attack_fn in attacks.items():
        attacked_img = attack_fn(image)
        try:
            extracted = zero_wm.extract(attacked_img, mwk["mwk"], mwk["rule_idx"])
            nc_val    = compute_nc(original_wm.ravel(), extracted.ravel())
        except Exception:
            nc_val    = 0.0
        results[name] = nc_val

    return results
