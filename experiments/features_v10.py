# experiments/features_v10.py  (revision 10)
"""AIF-Net logits for manifest sets (calibration C or, only through eval_final_v10.py, untouched U).

Images: produced on the laptop by prep_v10.py from the release files listed in manifest_v10.json
  BraTS: middle axial T1c slice (shape[2] // 2, on-disk order) as float32 .npy -> run_all_v4.to_u8
         (pixel-identical to nibabel + to_u8: checked on 5 test volumes, max abs difference 0)
  NIH, ISIC: gray level, LANCZOS 512 x 512 PNG (pixel-identical to the container pipeline on 6 test images)
Per patient: clean logits (5-view TTA), ROI cell flags (Otsu ROI, 8 x 8 cells), and, for the genuine subset,
logits after each of the 24 operations (RandomState(11) seeding per operation, as revision 9).
Genuine subset: a fixed random subset of the set (numpy RandomState(2027) permutation of the manifest order):
  C: 100 patients per modality; U: 200 patients per modality.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
_argv = sys.argv
sys.argv = [_argv[0], "/home/claude/data/datasets", "/tmp/claude-0/unused", "learned_v4", "0.7", "0"]
import run_all_v4 as RA                                                   # noqa: E402
sys.argv = _argv
from models import feature_net_v4 as FN4                                  # noqa: E402
from watermarking.zero_watermark import otsu_roi, block_classes            # noqa: E402
from eval_test_decoders import STD, CLIN                                   # noqa: E402

OPS = STD + CLIN
N_GENUINE = {"calib": 100, "untouched": 200}


def genuine_subset(items, set_name):
    """Fixed random subset of the FIRST occurrence of every patient (NIH U may repeat patients)."""
    first = sorted({it["patient"]: i for i, it in reversed(list(enumerate(items)))}.values())
    perm = np.random.RandomState(2027).permutation(len(first))[:N_GENUINE[set_name]]
    return sorted(first[i] for i in perm)


def load_image(mod, item, img_root):
    from PIL import Image
    base = os.path.basename(item["file"])
    if mod == "brats":
        return RA.to_u8(np.load(os.path.join(img_root, "brats", base.replace("-t1c.nii.gz", ".npy"))))
    return np.array(Image.open(os.path.join(img_root, mod, base.rsplit(".", 1)[0] + ".png")).convert("L"), np.uint8)


def key_name(item):
    """Name that keys the encounter (meta_2d(name)): the image file name without directories."""
    b = os.path.basename(item["file"])
    return b.replace("-t1c.nii.gz", "") if b.endswith(".nii.gz") else b.rsplit(".", 1)[0] if b.startswith("ISIC") else b


def _ops():
    atk = dict(RA.STANDARD_ATTACKS); atk.update(RA.MED2D)
    return [atk[k] for k in OPS]


def job(args):
    img, weights, do_att = args
    z = FN4.logits(img, weights)
    za = None
    if do_att:
        rs = np.random.RandomState(11); out = []
        for fn in _ops():
            np.random.seed(rs.randint(1 << 30))
            out.append(FN4.logits(fn(img), weights))
        za = np.stack(out)
    cls = block_classes(otsu_roi(img), (8, 8), True)
    return z, za, cls


def extract(manifest, mod, set_name, weights, img_root, nproc=2):
    from multiprocessing import Pool
    items = manifest["modalities"][mod][set_name]
    gen = set(genuine_subset(items, set_name))
    args = [(load_image(mod, it, img_root), weights, i in gen) for i, it in enumerate(items)]
    with Pool(nproc) as pool:
        res = pool.map(job, args, chunksize=2)
    return dict(names=np.array([key_name(it) for it in items]), patients=np.array([it["patient"] for it in items]),
                z=np.stack([r[0] for r in res]), cls=np.stack([r[2] for r in res]),
                gen_idx=np.array(sorted(gen)), z_att=np.stack([res[i][1] for i in sorted(gen)]), ops=np.array(OPS))
