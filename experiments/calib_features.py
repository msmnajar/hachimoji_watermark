# experiments/calib_features.py  (revision 9)
"""Calibration features for one AIF-Net model (validation data only; no test patient).

  * cal_clean  : clean logits of 200 NIH and 200 ISIC calibration patients (one image each,
                 disjoint from the training pool and the test sets), for impostor pairs;
  * cal_att    : logits of the first 50 NIH and 50 ISIC calibration images after each of the
                 24 standard and clinical operations (genuine attempts);
  * brats_att  : logits of the 32 BraTS validation slices (8 validation volumes x 4 depths)
                 after the same 24 operations, with their clean logits;
  * ROI cell flags of every image (for the NC rule).
Attacks use RandomState(11) seeding per operation, as in select_q_v4.py.
Usage: python experiments/calib_features.py <weights|-> <train_set.npz> <out.npz>
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
W, NPZ, OUT = sys.argv[1:4]
_argv = sys.argv
sys.argv = [_argv[0], "/home/claude/data/datasets", "/tmp/claude-0/unused", "learned_v4", "0.7", "0"]
import run_all_v4 as RA                                                   # noqa: E402
sys.argv = _argv
from models import feature_net_v4 as FN4                                  # noqa: E402
if W != "-":
    FN4.DEFAULT_WEIGHTS = W
from watermarking.zero_watermark import otsu_roi, block_classes            # noqa: E402
from eval_test_decoders import STD, CLIN                                   # noqa: E402

CAL = json.load(open("/home/claude/runs/calibration_selection.json"))
OPS = STD + CLIN


def ops():
    atk = dict(RA.STANDARD_ATTACKS); atk.update(RA.MED2D)
    return [(k, atk[k]) for k in OPS]


def logits(im):
    return FN4.logits(im, FN4.DEFAULT_WEIGHTS)


def attacked(im):
    rs = np.random.RandomState(11)
    out = []
    for _, fn in ops():
        np.random.seed(rs.randint(1 << 30))
        out.append(logits(fn(im)))
    return np.stack(out)


def load_png(name):
    from PIL import Image
    return np.array(Image.open(f"/home/claude/data/calibration/{name}").convert("L"), np.uint8)


def job(args):
    kind, name, img, do_att = args
    z = logits(img)
    za = attacked(img) if do_att else None
    cls = block_classes(otsu_roi(img), (8, 8), True)
    return kind, name, z, za, cls


if __name__ == "__main__":
    from multiprocessing import Pool
    items = []
    for i, n in enumerate(CAL["nih_calibration"]):
        items.append(("nih", n, load_png(n.rsplit(".", 1)[0] + ".png"), i < 50))
    for i, n in enumerate(CAL["isic_calibration"]):
        items.append(("isic", n, load_png(n + ".png"), i < 50))
    _D = np.load(NPZ)
    D = {k: _D[k] for k in ("brats", "vol_id", "n_volumes")}
    vol, nv = D["vol_id"], int(D["n_volumes"])
    for v in range(nv - 8, nv):
        ids = [i for i in range(len(vol)) if vol[i] == v]
        for r in (0.3, 0.45, 0.6, 0.75):
            items.append(("brats", f"vol{v}_d{r}", D["brats"][ids[int(r * (len(ids) - 1))]].copy(), True))
    with Pool(2) as pool:
        res = pool.map(job, items, chunksize=2)
    out = {}
    for kind in ("nih", "isic", "brats"):
        R = [r for r in res if r[0] == kind]
        out[f"{kind}_names"] = np.array([r[1] for r in R])
        out[f"{kind}_z"] = np.stack([r[2] for r in R])
        out[f"{kind}_cls"] = np.stack([r[4] for r in R])
        A = [r[3] for r in R if r[3] is not None]
        out[f"{kind}_att"] = np.stack(A)                     # (n_att_images, 24, 512)
    out["ops"] = np.array(OPS)
    np.savez(OUT, **out)
    print({k: v.shape for k, v in out.items()})
