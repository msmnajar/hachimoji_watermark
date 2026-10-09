# training/build_training_set.py  (v3)
"""
Builds the self-supervised training set for AIF-Net from images that are
disjoint from every test image used in the paper:
  * BraTS 2023 GLI: cases 101-190 in sorted order (test: cases 1-20),
    every 3rd axial T1c slice in the central 60% of the volume;
  * NIH ChestX-ray14: 100 images, one per patient, patient IDs >= 00014810
    (test: the first 20 files, patient IDs 00000001-00000005);
  * ISIC 2020: 100 images from the second half of the training folder
    (test: the first 20 files in sorted order).
TCIA RIDER Breast MRI and USC-SIPI are never used for training.

No labels are used. The slice index and volume id are kept so that nearby
slices of the same volume can serve as hard negatives.

Usage: python training/build_training_set.py <train_root> <out.npz>
"""
import glob
import os
import sys

import numpy as np
from PIL import Image


def to_u8(a):
    a = a.astype(np.float64)
    return ((a - a.min()) / (a.max() - a.min() + 1e-8) * 255).astype(np.uint8)


def main(root, out):
    import nibabel as nib
    brats, vol_id, sl_id = [], [], []
    files = sorted(glob.glob(f"{root}/datasets/brats2023/*/*/*t1c.nii.gz"))
    for v, f in enumerate(files):
        d = nib.load(f).get_fdata()
        z = d.shape[2]
        for s in range(int(0.2 * z), int(0.8 * z), 3):
            sl = d[:, :, s]
            if (sl > 0).mean() < 0.05:
                continue
            brats.append(to_u8(sl)); vol_id.append(v); sl_id.append(s)
    big = []
    for pat in (f"{root}/NIH_ChestXray14/images/*.png", f"{root}/ISIC2020/Training_JPEG/train/*.jpg"):
        for f in sorted(glob.glob(pat)):
            big.append(np.array(Image.open(f).convert("L").resize((512, 512), Image.LANCZOS), np.uint8))
    np.savez_compressed(out, brats=np.stack(brats), vol_id=np.array(vol_id), slice_id=np.array(sl_id),
                        big=np.stack(big), n_volumes=len(files),
                        brats_cases=np.array([os.path.basename(os.path.dirname(f)) for f in files]))
    print(f"BraTS slices {len(brats)} from {len(files)} volumes; 512x512 images {len(big)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
