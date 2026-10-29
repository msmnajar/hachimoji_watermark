# experiments/run_volumes_exp.py  (revision 7)
"""Volumetric integrity on the expanded test set: the first 30 volumes of BraTS-100
(one time point per patient, disjoint from training), 20 slices each, with the same
scenarios as run_all_v4.run_volume.  The replacement slice for "replace_other_patient"
comes from the last BraTS-100 patient, which is not one of the 30 volumes.
Robust layer: NC per slice (tau applied afterwards); fragile layer: hash chain.
Usage: python experiments/run_volumes_exp.py <out.json> [n_volumes]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
OUTF = sys.argv[1]
NV = int(sys.argv[2]) if len(sys.argv) > 2 else 30
_argv = sys.argv
sys.argv = [_argv[0], "/home/claude/data/datasets", "/tmp/claude-0/unused", "learned_v4", "0.7", "0"]
import run_all_v4 as RA                                                   # noqa: E402
sys.argv = _argv
from core.key_generator import PatientKey                                 # noqa: E402
from watermarking.volumetric import VolumeChain                           # noqa: E402
from evaluation.attacks import pacs_png_conversion, jpeg_compression, gaussian_noise  # noqa: E402

SEL = json.load(open("/home/claude/runs/expanded_selection.json"))["brats"]
P = "/home/claude/data/expanded/brats2023/asnr-miccai-brats2023-gli-challenge-trainingdata"


def run_volume(case):
    import nibabel as nib
    from scipy.ndimage import uniform_filter
    n_slices = 20
    d = nib.load(f"{P}/{case}/{case}-t1c.nii.gz").get_fdata()
    idx = np.linspace(d.shape[2] // 4, 3 * d.shape[2] // 4, n_slices, dtype=int)
    vol = np.stack([RA.to_u8(d[:, :, i]) for i in idx], 2)
    k = PatientKey(dict(PatientID=case, StudyInstanceUID=RA.uid(case), Modality="MR", SeriesDate="UNKNOWN"))
    ch = VolumeChain(k.master, k.arnold_params(), "learned_v4", 0.7)
    t0 = time.perf_counter(); recs, frag = ch.register(vol, RA.WM); t_reg = (time.perf_counter() - t0) * 1000
    scen = {}

    def run(name, v):
        r = ch.verify(v, RA.WM, recs, frag, tau=0.0)
        scen[name] = dict(nc=[round(x, 4) for x in r["nc"]], fragile_first=r["fragile_first"])
    run("clean", vol)
    run("benign_png", np.stack([pacs_png_conversion(vol[:, :, i]) for i in range(n_slices)], 2))
    run("benign_jpeg90", np.stack([jpeg_compression(vol[:, :, i], 90) for i in range(n_slices)], 2))
    run("benign_jpeg50", np.stack([jpeg_compression(vol[:, :, i], 50) for i in range(n_slices)], 2))
    np.random.seed(5)
    run("benign_noise5", np.stack([gaussian_noise(vol[:, :, i], 5) for i in range(n_slices)], 2))
    t = 10
    v = vol.copy(); v[:, :, t] = 128; run("replace_grey", v)
    other = SEL[-1]
    od = nib.load(f"{P}/{other}/{other}-t1c.nii.gz").get_fdata()
    v = vol.copy(); v[:, :, t] = RA.to_u8(od[:, :, int(round(idx[t] / d.shape[2] * od.shape[2]))]); run("replace_other_patient", v)
    v = vol.copy(); v[:, :, t] = vol[:, :, t + 1]; run("duplicate_neighbour", v)
    v = vol.copy(); s = v[:, :, t].copy(); s[130:178, 130:178] = s[60:108, 60:108]; v[:, :, t] = s; run("copy_move_48px", v)
    v = vol.copy(); s = v[:, :, t].astype(float)
    s[96:144, 96:144] = uniform_filter(s, 15)[96:144, 96:144]; v[:, :, t] = s.astype(np.uint8); run("local_smooth_48px", v)
    v = vol.copy(); v[:, :, [5, 12]] = v[:, :, [12, 5]]; run("swap_5_12", v)
    v = np.delete(vol, t, axis=2); run("delete_slice", v)
    return dict(case=case, n_slices=n_slices, slice_spacing=float(np.mean(np.diff(idx))),
                reg_ms_per_slice=round(t_reg / n_slices, 1), scenarios=scen)


if __name__ == "__main__":
    from multiprocessing import Pool
    done = json.load(open(OUTF)) if os.path.exists(OUTF) else []
    have = {r["case"] for r in done}
    todo = [c for c in SEL[:NV] if c not in have]
    with Pool(2) as pool:
        for r in pool.imap(run_volume, todo):
            done.append(r); json.dump(done, open(OUTF, "w"))
            print(r["case"], len(done), flush=True)
    print("ALL DONE", len(done))
