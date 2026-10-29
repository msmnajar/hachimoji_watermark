# experiments/val_impostors.py
"""Clean AIF-Net v4 logits of ALL slices of the 8 validation volumes (training pool,
never used for training), used as impostors when choosing decoder parameters:
  * other patient, nearest relative depth (slice replacement by another patient);
  * same volume, 1 or 2 slice steps away (3 or 6 slices: neighbour duplication).
Also stores the ROI cell flags of each slice.
Usage: python experiments/val_impostors.py <train_set.npz> <weights|-> <out.npz>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import feature_net_v4 as FN4                                    # noqa: E402
from watermarking.zero_watermark import otsu_roi, block_classes             # noqa: E402

npz, W, out = sys.argv[1:4]
if W != "-":
    FN4.DEFAULT_WEIGHTS = W
D = np.load(npz)
vol, sl, nv = D["vol_id"], D["slice_id"], int(D["n_volumes"])
idx = [i for i in range(len(vol)) if vol[i] >= nv - 8]
z = np.stack([FN4.logits(D["brats"][i], FN4.DEFAULT_WEIGHTS) for i in idx])
cls = np.stack([block_classes(otsu_roi(D["brats"][i]), (8, 8), True) for i in idx])
v = vol[idx]; s = sl[idx]
rel = np.zeros(len(idx))
for u in np.unique(v):
    m = v == u
    rel[m] = (s[m] - s[m].min()) / max(1, s[m].max() - s[m].min())
np.savez(out, z=z, cls=cls, vid=v, slice=s, reldepth=rel)
print(len(idx), "validation slices")
