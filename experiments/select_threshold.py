# experiments/select_threshold.py  (v3)
"""
Chooses the robust-layer threshold tau on VALIDATION volumes only (the last 8
BraTS training-pool volumes, cases 183-190 in sorted order, which are neither
test volumes nor used for gradient updates of AIF-Net).

For each feature extractor, every slice of 20 evenly spaced central slices is
registered in a hash chain and verified after the benign volume operations of
the paper (JPEG Q50 and Gaussian noise sigma = 5 on every slice). tau is the
1st percentile of these benign NC values, rounded down to 0.01, i.e. the
largest threshold with at most about 1% benign false alarms. Replacement of a
slice by the same position of another validation volume is also reported.

Usage: python experiments/select_threshold.py <train_root> <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.key_generator import PatientKey
from watermarking.volumetric import VolumeChain, slice_seed
from watermarking.zero_watermark import ZeroWatermarkV2, nc
from evaluation.attacks import jpeg_compression, gaussian_noise

WM = np.random.RandomState(42).randint(0, 2, (64, 64)).astype(np.uint8)


def to_u8(a):
    a = a.astype(np.float64)
    return ((a - a.min()) / (a.max() - a.min() + 1e-8) * 255).astype(np.uint8)


def load_vol(f, n=20):
    import nibabel as nib
    d = nib.load(f).get_fdata()
    idx = np.linspace(d.shape[2] // 4, 3 * d.shape[2] // 4, n, dtype=int)
    return np.stack([to_u8(d[:, :, i]) for i in idx], 2)


def main(root, out):
    files = sorted(glob.glob(f"{root}/datasets/brats2023/*/*/*t1c.nii.gz"))[-8:]
    vols = [load_vol(f) for f in files]
    res = {"validation_cases": [os.path.basename(os.path.dirname(f)) for f in files]}
    for feat in ("handcrafted", "learned"):
        benign, replaced = [], []
        for vi, (f, vol) in enumerate(zip(files, vols)):
            case = os.path.basename(os.path.dirname(f))
            k = PatientKey(dict(PatientID=case, StudyInstanceUID=case, Modality="MR", SeriesDate="UNKNOWN"))
            ch = VolumeChain(k.master, k.arnold_params(), feat)
            recs, frag = ch.register(vol, WM)
            v = np.stack([jpeg_compression(vol[:, :, i], 50) for i in range(vol.shape[2])], 2)
            benign += ch.verify(v, WM, recs, frag, 0.0)["nc"]
            np.random.seed(100 + vi)
            v = np.stack([gaussian_noise(vol[:, :, i], 5) for i in range(vol.shape[2])], 2)
            benign += ch.verify(v, WM, recs, frag, 0.0)["nc"]
            other = vols[(vi + 1) % len(vols)]
            for t in range(vol.shape[2]):         # robust-layer NC of slice t replaced by another patient
                prev = recs[t - 1]["mwk"].tobytes() if t > 0 else k.master
                zw = ZeroWatermarkV2(slice_seed(prev, k.master, t), k.arnold_params(), feat)
                replaced.append(nc(WM, zw.extract(other[:, :, t], recs[t])))
        benign, replaced = np.array(benign), np.array(replaced)
        tau = float(np.floor(np.percentile(benign, 1) * 100) / 100)
        res[feat] = dict(tau=tau, benign_n=int(benign.size), benign_p1=float(np.percentile(benign, 1)),
                         benign_min=float(benign.min()), benign_false_alarms_at_tau=int((benign < tau).sum()),
                         replaced_n=int(replaced.size), replaced_max=float(replaced.max()),
                         replaced_mean=float(replaced.mean()),
                         replaced_missed_at_tau=int((replaced >= tau).sum()))
        print(feat, json.dumps(res[feat]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
