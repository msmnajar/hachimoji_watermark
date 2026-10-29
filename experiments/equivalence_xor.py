# experiments/equivalence_xor.py
"""Empirical check of Proposition 1 (parity reduction of the Hachimoji layer).

For every test image and every operation of the paper, the proposed pipeline
(AIF-Net + TTA + reliable-bit selection, q = 0.7) is run twice on the SAME
feature bits and the SAME positions p_j:
  (H) Hachimoji coding with per-group rules and ROI-dependent diffusion (as in the paper);
  (X) a plain key-stream XOR: parity_j = f_{p_j} XOR b_j, where b_j is one bit drawn
      from an independent 6D hyperchaotic stream seeded with SHA-256(seed || "xor").
Proposition 1 predicts identical recovered watermarks (hence identical NC) under the
correct key, chance-level NC for wrong keys in both cases, and differs only in cost.

Also verifies Proposition 1 exhaustively over all 384 rules, 8 tribits, and all
diffusion values (k1, k2, k3) in {0..7}^3 for both diffusion paths.

Usage: python experiments/equivalence_xor.py <datasets_root> <out_dir> [nproc]
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT, OUT = sys.argv[1], sys.argv[2]
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 2
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
_argv = sys.argv
sys.argv = [_argv[0], ROOT, OUT, "learned_v4", "0.7", "0"]
import run_all_v4 as RA                                                 # noqa: E402
sys.argv = _argv
import core.hachimoji_dna as H                                          # noqa: E402
from core.chaotic_system import HyperchaoticSystem6D, seed_from_digest   # noqa: E402
from core.key_generator import PatientKey, DEFAULT_K_INST                # noqa: E402
from watermarking.zero_watermark import (ZeroWatermarkV2, otsu_roi, nc, arnold,  # noqa: E402
                                         block_classes, N_BITS, WM_SIZE, GLOBAL_ONLY, TILED)

WM = RA.WM
FEAT, Q = "learned_v4", 0.7


# ------------------------------------------------------------- exhaustive check
def exhaustive_check():
    """parity(c) XOR f must not depend on f, for every rule, tribit and key value."""
    viol = 0
    n = 0
    for r in range(H.N_RULES):
        for l in range(8):
            for k1 in range(8):
                for f in (0, 1):
                    pass
                # background path
                p0 = (H.op_xor(H.encode(np.array([l]), np.array([r])), k1) % 2)[0]
                p1 = (H.op_xor(H.encode(np.array([7 - l]), np.array([r])), k1) % 2)[0]
                viol += int(p0 == p1); n += 1
                for k2 in range(8):
                    for k3 in range(8):
                        def full(t):
                            b = H.encode(np.array([t]), np.array([r]))
                            return (H.op_inv(H.op_sub(H.op_xor(H.op_add(b, k1), k2), k3)) % 2)[0]
                        viol += int(full(l) == full(7 - l)); n += 1
    return dict(cases=n, violations=viol)


# ------------------------------------------------------------- two codings
def keystream_and_feats(zw, image, rec=None, roi_mask=None):
    f, blk, grp, grid, rel = zw._features(image)
    if rec is None:
        nb = grid[0] * grid[1]
        cls = block_classes(roi_mask, grid, zw.features in TILED) if roi_mask is not None else np.ones(nb, bool)
        k = max(1, int(round(zw.select_q * len(f))))
        sel = np.sort(np.argsort(-rel, kind="stable")[:k]).astype(np.int32)
    else:
        cls, sel = rec["block_class"], rec["sel"]
    full_fn = lambda p: np.where(blk[p] < 0, True, cls[np.maximum(blk[p], 0)])
    ks, _ = zw._keystream(len(f), int(grp.max()) + 1, full_fn, False, sel)
    return f, grp, ks, cls, sel


def xor_bits(seed):
    ch = HyperchaoticSystem6D(**seed_from_digest(hashlib.sha256(seed + b"xor").digest()))
    return ch.ints(N_BITS, 2).astype(np.uint8)


def parity_H(zw, f, grp, ks):
    return zw._code_parity(f, grp, ks)


def parity_X(f, ks, b):
    return np.bitwise_xor(f[ks["p"]].astype(np.uint8), b)


def recover(par, mwk, ap):
    return arnold(np.bitwise_xor(par, mwk).reshape(WM_SIZE, WM_SIZE), *ap, inverse=True)


def register_both(zw, img, roi):
    f, grp, ks, cls, sel = keystream_and_feats(zw, img, roi_mask=roi)
    wsc = arnold(WM.astype(np.uint8), *zw.ap).ravel()
    b = xor_bits(zw.seed)
    rec = dict(block_class=cls, sel=sel)
    t0 = time.perf_counter(); pH = parity_H(zw, f, grp, ks); tH = time.perf_counter() - t0
    t0 = time.perf_counter(); pX = parity_X(f, ks, b); tX = time.perf_counter() - t0
    return rec, np.bitwise_xor(pH, wsc), np.bitwise_xor(pX, wsc), tH, tX


def extract_both(zw, img, rec, mH, mX):
    f, grp, ks, _, _ = keystream_and_feats(zw, img, rec=rec)
    wH = recover(parity_H(zw, f, grp, ks), mH, zw.ap)
    wX = recover(parity_X(f, ks, xor_bits(zw.seed)), mX, zw.ap)
    return round(nc(WM, wH), 4), round(nc(WM, wX), 4), int(np.array_equal(wH, wX))


def attacks_of():
    atk = dict(RA.STANDARD_ATTACKS); atk.update(RA.UNSEEN_ATTACKS); atk.update(RA.COMPARE_ATTACKS)
    atk.update(RA.MED2D)
    return atk


def run_image(args):
    name, img, meta = args
    zw, _ = RA.zw_for(meta)
    roi = otsu_roi(img)
    rec, mH, mX, tH, tX = register_both(zw, img, roi)
    out = dict(name=name, t_code_H_ms=tH * 1000, t_code_X_ms=tX * 1000, attacks={})
    rs = np.random.RandomState(7)
    for an, fn in [("clean", lambda im: im)] + list(attacks_of().items()):
        if an != "clean":
            np.random.seed(rs.randint(1 << 30))
        a, b, same = extract_both(zw, fn(img), rec, mH, mX)
        out["attacks"][an] = (a, b, same)
    return out


def run_tcia(f):
    import pydicom
    ds = pydicom.dcmread(f)
    img = RA.to_u8(ds.pixel_array.astype(np.float64))
    meta = {k: str(getattr(ds, k, "UNKNOWN")) for k in ["PatientID", "StudyInstanceUID", "Modality", "SeriesDate"]}
    return meta, img


def wrong_key_tests(meta, img, other_meta):
    zw, _ = RA.zw_for(meta)
    rec, mH, mX, _, _ = register_both(zw, img, otsu_roi(img))
    res = {}
    flipped = bytearray(DEFAULT_K_INST); flipped[0] ^= 1
    conds = {"other_patient": (other_meta, DEFAULT_K_INST),
             "same_patient_other_study": (dict(meta, StudyInstanceUID=meta["StudyInstanceUID"] + ".1"), DEFAULT_K_INST),
             "random_patient_id": (dict(meta, PatientID="WRONG_PATIENT"), DEFAULT_K_INST),
             "wrong_institution_key": (meta, hashlib.sha256(b"another-hospital").digest()),
             "k_inst_1bit_flip": (meta, bytes(flipped))}
    for cn, (m, kinst) in conds.items():
        z, _ = RA.zw_for(m, kinst)
        a, b, _ = extract_both(z, img, rec, mH, mX)
        res[cn] = (a, b)
    a, b, same = extract_both(zw, img, rec, mH, mX)
    res["correct_key"] = (a, b, same)
    return res


if __name__ == "__main__":
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    res = dict(exhaustive=exhaustive_check())
    print("exhaustive", res["exhaustive"], flush=True)
    sets = {}
    sets["SIPI"] = RA.load_folder(f"{ROOT}/USC_SIPI/*.tif", 4, "SIPI_")
    sets["BraTS"], _ = RA.load_brats(20)
    sets["NIH"] = RA.load_folder(f"{ROOT}/NIH_ChestXray14/images/*.png", 20, "NIH_", 512)
    sets["ISIC"] = RA.load_folder(f"{ROOT}/ISIC2020/Training_JPEG/train/*.jpg", 20, "ISIC_", 512)
    sys.argv = [_argv[0], os.path.join(ROOT, "datasets2026"), OUT, FEAT, "0.7"]
    import run_2026_v4 as R26                                            # noqa: E402
    sys.argv = _argv
    for col in R26.COLLECTIONS:
        sets[col] = R26.load(col)
    with Pool(NPROC) as pool:
        for sn, data in sets.items():
            res[sn] = pool.map(run_image, data)
            print(sn, "done", round(time.time() - t_all), flush=True)
            json.dump(res, open(os.path.join(OUT, "equivalence_xor.json"), "w"))
    import pydicom
    dfiles = sorted(glob.glob(f"{ROOT}/TCIA/RIDER_Breast_MRI/*/*.dcm"))
    dfiles = [f for f in dfiles if (pydicom.dcmread(f).pixel_array > 0).mean() >= 0.05]
    tc = [run_tcia(f) for f in dfiles]
    kt = []
    for meta, img in tc:
        other = next(m for (m, _) in tc if m["PatientID"] != meta["PatientID"])
        kt.append(wrong_key_tests(meta, img, other))
    res["TCIA_keytests"] = kt
    res["seconds"] = round(time.time() - t_all)
    json.dump(res, open(os.path.join(OUT, "equivalence_xor.json"), "w"))
    print("ALL DONE", res["seconds"], flush=True)
