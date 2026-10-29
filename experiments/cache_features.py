# experiments/cache_features.py
"""Feature cache for the expanded evaluation (revision 7).

For every test image and every condition (clean + the 34 operations of the paper,
applied in the same order and with the same seeding as run_all_v4.attack_battery),
stores the output of one feature extractor: the feature bits f and their reliability
values rel (None for the handcrafted features without selection), together with the
bit structure (blk, grp, grid) and the ROI cell flags of the clean image.
All decision rules (NC threshold of the parity zero-watermark, Hachimoji Reed-Solomon
code-offset layer, binary BCH, GF(4)) are then evaluated exactly on these features by
experiments/eval_cached.py, because every one of them depends on the image only
through (f, rel).

Usage: python experiments/cache_features.py <feature> <weights|-> <out.npz> <set> [<set> ...]
sets: brats100 nih100 isic100 sipi c2026
"""
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
FEAT, WEIGHTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SETS = sys.argv[4:]
EXP = "/home/claude/data/expanded"
ORIG = "/home/claude/data/datasets"
_argv = sys.argv
sys.argv = [_argv[0], ORIG, "/tmp/claude-0/unused", "learned_v4", "0.7", "0"]
import run_all_v4 as RA                                                   # noqa: E402
sys.argv = _argv
from models import feature_net_v4 as FN4                                  # noqa: E402
if WEIGHTS != "-":
    FN4.DEFAULT_WEIGHTS = WEIGHTS
from watermarking.zero_watermark import FEATURES, otsu_roi, block_classes, TILED, GLOBAL_ONLY  # noqa: E402

SEL = json.load(open("/home/claude/runs/expanded_selection.json"))


def conditions():
    atk = dict(RA.STANDARD_ATTACKS); atk.update(RA.UNSEEN_ATTACKS); atk.update(RA.COMPARE_ATTACKS)
    atk.update(RA.MED2D)
    return atk


def load_set(name):
    from PIL import Image
    out = []
    if name == "brats100":
        import nibabel as nib
        for c in SEL["brats"]:
            f = f"{EXP}/brats2023/asnr-miccai-brats2023-gli-challenge-trainingdata/{c}/{c}-t1c.nii.gz"
            d = nib.load(f).get_fdata()
            out.append((c, RA.to_u8(d[:, :, d.shape[2] // 2])))
    elif name in ("nih100", "isic100"):
        lst = SEL["nih"] if name == "nih100" else [x + ".jpg" for x in SEL["isic"]]
        sub = "NIH_ChestXray14/images" if name == "nih100" else "ISIC2020/Training_JPEG/train"
        for x in lst:
            im = Image.open(f"{EXP}/{sub}/{x}").convert("L").resize((512, 512), Image.LANCZOS)
            out.append((x, np.array(im, np.uint8)))
    elif name == "sipi":
        out = [(n, im) for n, im, _ in RA.load_folder(f"{ORIG}/USC_SIPI/*.tif", 4, "SIPI_", 512)]
    elif name == "c2026":
        sys.argv = [_argv[0], os.path.join(ORIG, "datasets2026"), "/tmp/claude-0/unused", "learned_v4", "0.7"]
        import run_2026_v4 as R26                                         # noqa: E402
        sys.argv = _argv
        for col in R26.COLLECTIONS:
            out += [(f"{col}/{n}", im) for n, im, _ in R26.load(col)]
    return out


def one(args):
    name, img = args
    fn = FEATURES[FEAT]
    o = fn(img)
    f, blk, grp, grid = o[:4]
    rel = o[4] if len(o) > 4 else None
    if FEAT in GLOBAL_ONLY:
        cls = np.ones(grid[0] * grid[1], bool)
    else:
        cls = block_classes(otsu_roi(img), grid, FEAT in TILED)
    F = [f.astype(np.uint8)]; R = [rel.astype(np.float32) if rel is not None else None]
    rs = np.random.RandomState(7)
    for an, afn in conditions().items():
        np.random.seed(rs.randint(1 << 30))
        o2 = fn(afn(img))
        if o2[0].shape != f.shape:              # size change: the pipeline rejects such an image (NC None)
            F.append(np.full(f.shape, 255, np.uint8)); R.append(None if R[0] is None else np.zeros(f.shape, np.float32))
            continue
        F.append(o2[0].astype(np.uint8)); R.append(o2[4].astype(np.float32) if len(o2) > 4 and o2[4] is not None else None)
    return dict(name=name, shape=img.shape, f=np.stack(F), rel=None if R[0] is None else np.stack(R),
                blk=blk, grp=grp, grid=grid, cls=cls)


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    res = {}
    if os.path.exists(OUT):                     # resume: keep the sets already cached in OUT
        res = np.load(OUT, allow_pickle=True).item()["sets"]
        print("resume with", list(res), flush=True)
    cond = ["clean"] + list(conditions())
    with Pool(int(os.environ.get("NPROC", "2"))) as pool:
        for s in SETS:
            if s in res:
                continue
            data = load_set(s)
            res[s] = pool.map(one, data, chunksize=1)
            print(s, len(data), round(time.time() - t0), flush=True)
            np.save(OUT, dict(feature=FEAT, weights=WEIGHTS, conditions=cond, sets=res), allow_pickle=True)
    print("ALL DONE", round(time.time() - t0), flush=True)
