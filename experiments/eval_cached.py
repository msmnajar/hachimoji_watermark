# experiments/eval_cached.py
"""Exact evaluation of the decision rules on cached features (experiments/cache_features.py).

parity_nc(): the zero-watermark of the paper (Hachimoji parity coding, ROI diffusion, MWK,
             inverse Arnold), run with ZeroWatermarkV2 itself; only the feature extractor
             call is replaced by the cached output.  Reproduces run_all_v4 exactly (self-test).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from core.key_generator import PatientKey, DEFAULT_K_INST                      # noqa: E402
from watermarking.zero_watermark import ZeroWatermarkV2, arnold, nc, WM_SIZE  # noqa: E402

WM = np.random.RandomState(42).randint(0, 2, (64, 64)).astype(np.uint8)


def uid(s):
    import hashlib
    return "2.25." + str(int(hashlib.sha256(s.encode()).hexdigest()[:30], 16))


class CachedZW(ZeroWatermarkV2):
    """ZeroWatermarkV2 whose feature extractor returns a cached (f, blk, grp, grid, rel)."""
    def __init__(self, seed, ap, features, q):
        super().__init__(seed, ap, features, q)
        self._cur = None

    def _features(self, image):
        return self._cur


def zw_for(meta, features, q, k_inst=DEFAULT_K_INST):
    k = PatientKey(meta, k_inst)
    return CachedZW(k.sub("pixel"), k.arnold_params(), features, q)


def meta_2d(name, modality="OT", prefix=""):
    pid = prefix + name
    return dict(PatientID=pid, StudyInstanceUID=uid(name), Modality=modality, SeriesDate="UNKNOWN")


def entry_tuple(e, c):
    rel = None if e["rel"] is None else e["rel"][c]
    return (e["f"][c], e["blk"], e["grp"], tuple(e["grid"]), rel)


def register(zw, e):
    """Registration with the cached clean features and the stored ROI cell flags."""
    zw._cur = entry_tuple(e, 0)
    import watermarking.zero_watermark as Z
    orig = Z.block_classes
    Z.block_classes = lambda roi, grid, tiled=False: e["cls"]
    try:
        rec = zw.register(np.zeros((8, 8), np.uint8), WM, roi_mask=np.ones((8, 8), bool))
    finally:
        Z.block_classes = orig
    return rec


def extract_nc(zw, e, c, rec):
    if e["f"][c].max() > 1:                     # condition marked invalid (feature length changed)
        return None
    zw._cur = entry_tuple(e, c)
    return round(nc(WM, zw.extract(None, rec)), 4)


if __name__ == "__main__":
    # self-test against the published per-image results (USC-SIPI, AIF-Net v4, q = 0.7)
    import json
    cache = np.load(sys.argv[1], allow_pickle=True).item()
    pub = json.load(open("/home/claude/v3/results_v4/results_v4_learned_v4_q0.7.json"))["E1_sipi"]
    cond = cache["conditions"]
    mism = 0; n = 0
    for e, p in zip(cache["sets"]["sipi"], pub):
        assert e["name"] == p["name"]
        zw = zw_for(meta_2d(e["name"], prefix="SIPI_"), "learned_v4", 0.7)
        rec = register(zw, e)
        for ci, cn in enumerate(cond[1:], 1):
            v = extract_nc(zw, e, ci, rec); n += 1
            if v != p["attacks"][cn]:
                mism += 1; print("MISMATCH", e["name"], cn, v, p["attacks"][cn])
    print(f"self-test: {n} values, {mism} mismatches")


def fast_nc(zw, e, rec):
    """Returns g(f) = NC of the watermark recovered from feature bits f with record rec.
    Identical to extract_nc (same key stream, coding, MWK and inverse Arnold), but the key
    stream, which depends only on the record and the key, is generated once."""
    blk, grp = e["blk"], e["grp"]
    cls = rec["block_class"]
    full_fn = lambda p: np.where(blk[p] < 0, True, cls[np.maximum(blk[p], 0)])
    ks, _ = zw._keystream(len(e["f"][0]), int(grp.max()) + 1, full_fn, rec["uniform_full"], rec.get("sel"))

    def g(f):
        if f.max() > 1:
            return None
        par = zw._code_parity(f, grp, ks)
        w = np.bitwise_xor(par, rec["mwk"]).reshape(WM_SIZE, WM_SIZE)
        return round(nc(WM, arnold(w, *zw.ap, inverse=True)), 4)
    return g
