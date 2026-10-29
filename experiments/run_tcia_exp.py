# experiments/run_tcia_exp.py  (revision 7)
"""TCIA RIDER Breast MRI on the expanded local copy (15 series, 2 patients): every 5th slice
with >= 5% non-zero pixels, DICOM operations of run_all_v4.run_dicom and the six wrong-key
tests of run_all_v4.key_tests (exact pipeline).
Usage: python experiments/run_tcia_exp.py <out.json>
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
OUTF = sys.argv[1]
_argv = sys.argv
sys.argv = [_argv[0], "/home/claude/data/expanded", "/tmp/claude-0/unused", "learned_v4", "0.7", "0"]
import run_all_v4 as RA                                                   # noqa: E402
sys.argv = _argv


def job(f):
    r, (meta, img, rec) = RA.run_dicom(f)
    return r, meta, img, rec


if __name__ == "__main__":
    import pydicom
    from multiprocessing import Pool
    fs = sorted(glob.glob("/home/claude/data/expanded/TCIA/RIDER_Breast_MRI/*/*.dcm"))
    usable = [f for f in fs if (pydicom.dcmread(f).pixel_array > 0).mean() >= 0.05]
    sel = usable[::5]
    with Pool(2) as pool:
        out = pool.map(job, sel)
    res = dict(n_files=len(fs), n_usable=len(usable), n_selected=len(sel), slices=[o[0] for o in out], keytests=[])
    metas = [(o[1], o[2], o[3]) for o in out]
    for meta, img, rec in metas:
        other = next(m for (m, _, _) in metas if m["PatientID"] != meta["PatientID"])
        res["keytests"].append(RA.key_tests(meta, img, rec, other))
        json.dump(res, open(OUTF, "w"))
    res["patients"] = sorted({m["PatientID"] for m, _, _ in metas})
    res["studies"] = len({m["StudyInstanceUID"] for m, _, _ in metas})
    json.dump(res, open(OUTF, "w"))
    print("done", len(sel), res["patients"], res["studies"])
