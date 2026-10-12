# watermarking/zero_watermark.py  (v4: handcrafted, AIF-Net v3/v4 and baseline features,
#                                   optional reliability-aware bit selection)
"""
ROI-selective Hachimoji zero-watermarking (v2).

Registration (host image is never modified):
 1. Features (see extract_features): 255 global DWT-DCT sign bits plus 3 bits
    per 8x8 LL2 block (sigma1, sigma2, LH2 energy vs. median).
 2. ROI: Otsu segmentation; an LL2 block is ROI if >= 50% of its 32x32 pixel
    footprint is ROI. The block class map (B bits) is stored in the record.
 3. Key-dependent expansion: position j (0..4095) reads feature p(j) =
    chaotic index in [0, 3B).
 4. Hachimoji coding with per-block rules: r_b in [0,384) drawn from the 6D
    hyperchaotic sequence for every LL2 block b (and every group of 8 global bits). Tribit t_j = l_j if
    f_p(j) = 0, else 7 - l_j (l_j: chaotic 3-bit value); base beta_j =
    R_{r_b}[t_j]. Complementary tribits map to complementary bases, whose
    index parities differ, so parity(beta_j) flips exactly when f flips.
 5. Diffusion: ROI-block and global positions -> add, xor, sub (three independent chaotic
    keystreams in 0..7) + bitwise inversion; background-block positions -> one
    chaotic XOR.
 6. MWK_j = parity(c_j) XOR W'_j, W' = Arnold-scrambled watermark.
Extraction repeats 1, 3-6 on the received image with the stored block map
and returns W' = MWK XOR parity(c'), then inverse Arnold.

Because only the parity of the final base enters the MWK and all diffusion
operations are affine in parity, the Hachimoji layer acts as a keyed
bit-substitution: robustness equals feature-bit stability, and security is
bounded by the key (HMAC-derived seed), not by the number of rules.
"""
import time
import numpy as np
import pywt

from core.chaotic_system import HyperchaoticSystem6D, seed_from_digest
from core import hachimoji_dna as H

WM_SIZE = 64
N_BITS = WM_SIZE * WM_SIZE
BLOCK = 8


# ---------------------------------------------------------------- features
N_GLOBAL = 255   # DCT sign bits of LL2 (16x16 low-frequency block minus DC)
GROUP = 8        # global bits per Hachimoji rule group


def extract_features(image):
    """Hybrid feature vector.
    (a) global: signs of the 16x16 lowest-frequency 2D-DCT coefficients of the
        level-2 Haar LL band (DC excluded) -> 255 bits (discriminative hash);
    (b) block-local: for every 8x8 LL2 block, sigma1, sigma2 of the block and
        the energy of the co-located LH2 block, each binarised at its median
        -> 3 bits per block (spatially localised, used for ROI selectivity).
    Returns f and, for each bit, its rule group and its LL2 block (-1 = global)."""
    from scipy.fft import dctn
    c = pywt.wavedec2(image.astype(np.float64), "haar", level=2)
    LL, LH = c[0], c[1][0]
    g = (dctn(LL, norm="ortho")[:16, :16].ravel()[1:] > 0).astype(np.uint8)
    Hh, Ww = LL.shape
    s1, s2, e = [], [], []
    for i in range(0, Hh - BLOCK + 1, BLOCK):
        for j in range(0, Ww - BLOCK + 1, BLOCK):
            s = np.linalg.svd(LL[i:i+BLOCK, j:j+BLOCK], compute_uv=False)
            s1.append(s[0]); s2.append(s[1])
            e.append(float(np.sum(LH[i:i+BLOCK, j:j+BLOCK] ** 2)))
    s1, s2, e = map(np.asarray, (s1, s2, e))
    nb = len(s1)
    loc = np.zeros(3 * nb, np.uint8)
    loc[0::3] = s1 > np.median(s1)
    loc[1::3] = s2 > np.median(s2)
    loc[2::3] = e > np.median(e)
    f = np.concatenate([g, loc])
    blk = np.concatenate([-np.ones(N_GLOBAL, np.int64), np.arange(3 * nb) // 3])
    n_ggroups = (N_GLOBAL + GROUP - 1) // GROUP
    grp = np.concatenate([np.arange(N_GLOBAL) // GROUP, n_ggroups + np.arange(3 * nb) // 3])
    return f, blk, grp, (Hh // BLOCK, Ww // BLOCK)


def _robust_z(v, thr):
    mad = np.median(np.abs(v - thr)) + 1e-12
    return np.abs(v - thr) / mad


def extract_features_handcrafted_rel(image):
    """Handcrafted features with a reliability value per bit (used only with
    reliability-aware selection): |DCT coefficient| / MAD for the global bits and
    |x - median| / MAD for the block bits."""
    from scipy.fft import dctn
    f, blk, grp, grid = extract_features(image)
    c = pywt.wavedec2(image.astype(np.float64), "haar", level=2)
    LL, LH = c[0], c[1][0]
    d = dctn(LL, norm="ortho")[:16, :16].ravel()[1:]
    rel_g = _robust_z(d, 0.0)
    Hh, Ww = LL.shape
    s1, s2, e = [], [], []
    for i in range(0, Hh - BLOCK + 1, BLOCK):
        for j in range(0, Ww - BLOCK + 1, BLOCK):
            sv = np.linalg.svd(LL[i:i+BLOCK, j:j+BLOCK], compute_uv=False)
            s1.append(sv[0]); s2.append(sv[1])
            e.append(float(np.sum(LH[i:i+BLOCK, j:j+BLOCK] ** 2)))
    rel_l = np.zeros(len(f) - N_GLOBAL)
    for k, x in enumerate((np.asarray(s1), np.asarray(s2), np.asarray(e))):
        rel_l[k::3] = _robust_z(x, np.median(x))
    return f, blk, grp, grid, np.concatenate([rel_g, rel_l])


def extract_features_learned(image, weights=None):
    """v3 learned features (models/feature_net.py, AIF-Net).
    (a) global: signs of the 256 global logits;
    (b) block-local: signs of the 3 local logits resampled to the LL2 block grid
        (same block grid and bit order as extract_features).
    Returns f, blk, grp, grid exactly like extract_features."""
    from models import feature_net as FN
    c = pywt.wavedec2(image.astype(np.float64), "haar", level=2)
    Hh, Ww = c[0].shape
    gh, gw = Hh // BLOCK, Ww // BLOCK
    g_log, l_log = FN.logits(image, (gh, gw), weights or FN.DEFAULT_WEIGHTS)
    g = (g_log > 0).astype(np.uint8)
    loc = (l_log > 0).astype(np.uint8)                         # (3, gh, gw)
    loc = loc.reshape(3, gh * gw).T.ravel()                    # block-major, 3 bits per block
    nb = gh * gw
    ng = len(g)
    f = np.concatenate([g, loc])
    blk = np.concatenate([-np.ones(ng, np.int64), np.arange(3 * nb) // 3])
    n_ggroups = (ng + GROUP - 1) // GROUP
    grp = np.concatenate([np.arange(ng) // GROUP, n_ggroups + np.arange(3 * nb) // 3])
    return f, blk, grp, (gh, gw)


CELLS = 8                                  # AIF-Net v4 local code: 8 x 8 cells of the image
V4_GROUP = 8


def extract_features_learned_v4(image, weights=None):
    """v4 AIF-Net features (models/feature_net_v4.py) with test-time augmentation.
    256 global bits (rule groups of 8) and 4 bits per cell of an 8 x 8 grid over the
    image (one rule group per cell). The reliability of a bit is |standardized logit|."""
    from models import feature_net_v4 as FN4
    z = FN4.logits(image, weights or FN4.DEFAULT_WEIGHTS)
    ng = FN4.N_GLOBAL
    nl = len(z) - ng
    f = (z > 0).astype(np.uint8)
    blk = np.concatenate([-np.ones(ng, np.int64), np.arange(nl) // FN4.N_LOCAL])
    n_gg = ng // V4_GROUP
    grp = np.concatenate([np.arange(ng) // V4_GROUP, n_gg + np.arange(nl) // FN4.N_LOCAL])
    return f, blk, grp, (CELLS, CELLS), np.abs(z)


def extract_features_learned_v4_notta(image, weights=None):
    """AIF-Net v4 without test-time augmentation (ablation)."""
    from models import feature_net_v4 as FN4
    z = FN4.logits(image, weights or FN4.DEFAULT_WEIGHTS, tta=False)
    ng = FN4.N_GLOBAL
    nl = len(z) - ng
    blk = np.concatenate([-np.ones(ng, np.int64), np.arange(nl) // FN4.N_LOCAL])
    grp = np.concatenate([np.arange(ng) // V4_GROUP, ng // V4_GROUP + np.arange(nl) // FN4.N_LOCAL])
    return (z > 0).astype(np.uint8), blk, grp, (CELLS, CELLS), np.abs(z)


_BIN = {}


def _baseline_rule(name):
    """Binarization rule of a baseline, chosen on validation data in its favor
    (experiments/tune_baselines_v4.py); default: per-image median."""
    if name not in _BIN:
        import json
        from pathlib import Path
        f = Path(__file__).resolve().parent.parent / "models" / "pretrained" / "baseline_binarization.json"
        cfg = json.loads(f.read_text()) if f.exists() else {}
        c = cfg.get(name, {})
        _BIN[name] = (c.get("selected", "median"), [np.asarray(p) for p in c.get("pop_thresholds", [])])
    return _BIN[name]


def _baseline(name):
    def fn(image):
        from scipy.fft import dct
        from models import baselines as B
        rule, pop = _baseline_rule(name)
        parts = B.BASELINES[name](image)
        f, rel = [], []
        for i, v in enumerate(parts):
            if rule == "dct":
                v = dct(v - v.mean(), norm="ortho")[1:]
                thr = 0.0
            elif rule == "pop":
                thr = pop[i]
            else:
                thr = np.median(v)
            f.append((v > thr).astype(np.uint8))
            d = np.abs(v - thr)
            rel.append(d / (np.median(d) + 1e-12))
        f = np.concatenate(f)
        n = len(f)
        return f, -np.ones(n, np.int64), np.arange(n) // GROUP, (1, 1), np.concatenate(rel)
    return fn


FEATURES = {"handcrafted": extract_features, "learned": extract_features_learned,
            "handcrafted_rel": extract_features_handcrafted_rel, "learned_v4": extract_features_learned_v4,
            "learned_v4_notta": extract_features_learned_v4_notta,
            "vgg19_lbp": _baseline("vgg19_lbp"), "focalnet": _baseline("focalnet")}
TILED = {"learned_v4", "learned_v4_notta"}                     # block flags per equal tile (cell) of the image
GLOBAL_ONLY = {"vgg19_lbp", "focalnet"}    # no spatial bits: every position is full strength


def block_classes(roi_mask, grid, tiled=False):
    """ROI flag per LL2 block (block covers 32x32 image pixels), or, if tiled, per
    equal tile of an gh x gw grid over the whole image (AIF-Net v4 cells)."""
    gh, gw = grid
    if tiled:
        H, W = roi_mask.shape
        ys = np.linspace(0, H, gh + 1).astype(int); xs = np.linspace(0, W, gw + 1).astype(int)
        return np.array([roi_mask[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean() >= 0.5
                         for i in range(gh) for j in range(gw)], bool)
    s = BLOCK * 4
    out = np.zeros(gh * gw, bool)
    for bi in range(gh):
        for bj in range(gw):
            out[bi * gw + bj] = roi_mask[bi*s:(bi+1)*s, bj*s:(bj+1)*s].mean() >= 0.5
    return out


def otsu_roi(image):
    from skimage.filters import gaussian, threshold_otsu
    from skimage.morphology import dilation, disk
    b = gaussian(image.astype(np.float32), sigma=2.5, preserve_range=True)
    m = b > threshold_otsu(b)
    return dilation(m, disk(4))


# ---------------------------------------------------------------- Arnold
def arnold(img, a, b, n, inverse=False):
    N = img.shape[0]
    x, y = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    out = img.copy()
    for _ in range(n):
        tmp = np.empty_like(out)
        if not inverse:
            tmp[(x + b*y) % N, (a*x + (a*b+1)*y) % N] = out
        else:
            tmp[((a*b+1)*x - b*y) % N, (-a*x + y) % N] = out
        out = tmp
    return out


# ---------------------------------------------------------------- core
class ZeroWatermarkV2:
    def __init__(self, seed_digest: bytes, arnold_params, features: str = "handcrafted", select_q=None):
        """select_q: None (all feature bits are used, v2/v3 behavior) or the fraction of
        most reliable feature bits to which the watermark is bound (reliability-aware
        selection, v4). The selected feature indices are stored in the record."""
        self.seed = seed_digest
        self.ap = arnold_params
        self.features = features
        self._feat = FEATURES[features]
        self.select_q = select_q

    def _features(self, image):
        out = self._feat(image)
        if len(out) == 4:
            return out + (None,)
        return out

    def _keystream(self, n_feat, n_groups, roi_pos, uniform_full=False, sel=None):
        """All chaotic material for one image. Returns dict + generation time.
        With reliability-aware selection, positions draw from the selected features only."""
        t0 = time.perf_counter()
        ch = HyperchaoticSystem6D(**seed_from_digest(self.seed))
        t_warm = (time.perf_counter() - t0) * 1000
        if sel is None:
            p = ch.ints(N_BITS, n_feat)          # position -> feature index
        else:
            p = np.asarray(sel)[ch.ints(N_BITS, len(sel))]
        rules = ch.ints(n_groups, H.N_RULES)     # one Hachimoji rule per group/block
        low = ch.ints(N_BITS, 8)                 # base tribit per position
        full = np.ones(N_BITS, bool) if uniform_full else roi_pos(p)
        k1 = ch.ints(N_BITS, 8)                  # every position
        nf = int(full.sum())
        k2 = np.zeros(N_BITS, np.int64); k3 = np.zeros(N_BITS, np.int64)
        k2[full] = ch.ints(nf, 8)                # full-strength positions only
        k3[full] = ch.ints(nf, 8)
        return dict(p=p, rules=rules, low=low, k1=k1, k2=k2, k3=k3, full=full, t_warm=t_warm), \
            (time.perf_counter() - t0) * 1000

    def _code_parity(self, f, grp, ks):
        p, full = ks["p"], ks["full"]
        fb = f[p]
        t = np.where(fb == 0, ks["low"], 7 - ks["low"])
        beta = H.encode(t, ks["rules"][grp[p]])
        c = beta.copy()
        # full-strength (ROI) chain: add -> xor -> sub -> inversion
        c[full] = H.op_inv(H.op_sub(H.op_xor(H.op_add(c[full], ks["k1"][full]),
                                              ks["k2"][full]), ks["k3"][full]))
        # lightweight (background): single XOR
        c[~full] = H.op_xor(c[~full], ks["k1"][~full])
        return (c % 2).astype(np.uint8)

    def register(self, image, watermark, roi_mask=None, uniform_full=False):
        t0 = time.perf_counter()
        f, blk, grp, grid, rel = self._features(image)
        nb = grid[0] * grid[1]
        if roi_mask is not None and self.features not in GLOBAL_ONLY:
            cls = block_classes(roi_mask, grid, self.features in TILED)
        else:
            cls = np.ones(nb, bool)
        sel = None
        if self.select_q is not None:
            k = max(1, int(round(self.select_q * len(f))))
            sel = np.sort(np.argsort(-rel, kind="stable")[:k]).astype(np.int32)
        full_fn = lambda p: np.where(blk[p] < 0, True, cls[np.maximum(blk[p], 0)])
        ks, t_ks = self._keystream(len(f), int(grp.max()) + 1, full_fn, uniform_full, sel)
        par = self._code_parity(f, grp, ks)
        wsc = arnold(watermark.astype(np.uint8), *self.ap).ravel()
        mwk = np.bitwise_xor(par, wsc)
        return dict(mwk=mwk, block_class=cls, grid=grid, uniform_full=uniform_full, features=self.features,
                    sel=sel, select_q=self.select_q,
                    t_keystream_ms=t_ks, t_warm_ms=ks["t_warm"], n_full=int(ks["full"].sum()),
                    t_total_ms=(time.perf_counter() - t0) * 1000,
                    roi_block_fraction=float(cls.mean()))

    def extract(self, image, record):
        f, blk, grp, grid, _ = self._features(image)
        if grid != record["grid"]:
            raise ValueError("image size changed")
        if record.get("features", "handcrafted") != self.features:
            raise ValueError("record was registered with another feature extractor")
        cls = record["block_class"]
        full_fn = lambda p: np.where(blk[p] < 0, True, cls[np.maximum(blk[p], 0)])
        ks, _ = self._keystream(len(f), int(grp.max()) + 1, full_fn, record["uniform_full"], record.get("sel"))
        par = self._code_parity(f, grp, ks)
        w = np.bitwise_xor(par, record["mwk"]).reshape(WM_SIZE, WM_SIZE)
        return arnold(w, *self.ap, inverse=True)


def nc(a, b):
    a = a.ravel().astype(float); b = b.ravel().astype(float)
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-15))


def ber(a, b):
    return float(np.mean(a.ravel().astype(np.uint8) != b.ravel().astype(np.uint8)))
