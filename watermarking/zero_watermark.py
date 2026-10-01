# watermarking/zero_watermark.py  (v2)
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


def block_classes(roi_mask, grid):
    """ROI flag per LL2 block (block covers 32x32 image pixels)."""
    gh, gw = grid
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
    def __init__(self, seed_digest: bytes, arnold_params):
        self.seed = seed_digest
        self.ap = arnold_params

    def _keystream(self, n_feat, n_groups, roi_pos, uniform_full=False):
        """All chaotic material for one image. Returns dict + generation time."""
        t0 = time.perf_counter()
        ch = HyperchaoticSystem6D(**seed_from_digest(self.seed))
        t_warm = (time.perf_counter() - t0) * 1000
        p = ch.ints(N_BITS, n_feat)              # position -> feature index
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
        f, blk, grp, grid = extract_features(image)
        nb = grid[0] * grid[1]
        cls = block_classes(roi_mask, grid) if roi_mask is not None else np.ones(nb, bool)
        full_fn = lambda p: np.where(blk[p] < 0, True, cls[np.maximum(blk[p], 0)])
        ks, t_ks = self._keystream(len(f), int(grp.max()) + 1, full_fn, uniform_full)
        par = self._code_parity(f, grp, ks)
        wsc = arnold(watermark.astype(np.uint8), *self.ap).ravel()
        mwk = np.bitwise_xor(par, wsc)
        return dict(mwk=mwk, block_class=cls, grid=grid, uniform_full=uniform_full,
                    t_keystream_ms=t_ks, t_warm_ms=ks["t_warm"], n_full=int(ks["full"].sum()),
                    t_total_ms=(time.perf_counter() - t0) * 1000,
                    roi_block_fraction=float(cls.mean()))

    def extract(self, image, record):
        f, blk, grp, grid = extract_features(image)
        if grid != record["grid"]:
            raise ValueError("image size changed")
        cls = record["block_class"]
        full_fn = lambda p: np.where(blk[p] < 0, True, cls[np.maximum(blk[p], 0)])
        ks, _ = self._keystream(len(f), int(grp.max()) + 1, full_fn, record["uniform_full"])
        par = self._code_parity(f, grp, ks)
        w = np.bitwise_xor(par, record["mwk"]).reshape(WM_SIZE, WM_SIZE)
        return arnold(w, *self.ap, inverse=True)


def nc(a, b):
    a = a.ravel().astype(float); b = b.ravel().astype(float)
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-15))


def ber(a, b):
    return float(np.mean(a.ravel().astype(np.uint8) != b.ravel().astype(np.uint8)))
