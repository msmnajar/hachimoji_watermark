# experiments/run_2026_v4.py  (v4)
"""Evaluates one feature configuration on the public test collections used by
zero-watermarking and medical-image-protection studies published in 2026
(datasets2026/, see datasets2026/manifest.json):

  BrainTumorMRI_Nickparvar  Taj et al. 2026; Niharika et al. 2026   brain MRI (Kaggle)
  CT_Kidney                 Niharika et al. 2026                   kidney CT (Kaggle)
  BrainMRI_Navoneel         George et al. 2026                     brain MRI (Kaggle)
  TCGA_LGG_Buda             George et al. 2026 (TCGA-LGG)          brain MRI, FLAIR channel
  MIAS                      Suzgen et al. 2026                     mammography
  BoneBreak                 Suzgen et al. 2026                     bone X-ray
  RetinaFundus              Suzgen et al. 2026                     retinal fundus
  COVID_CXR_Cohen           Anand & Singh 2026                     chest X-ray (GitHub)

Every image is converted to gray level (TCGA_LGG_Buda: FLAIR, the middle channel of
the RGB composite) and resized to 512 x 512 (Lanczos), as for NIH and ISIC.
Per collection: the full attack battery of run_all_v4.py and the NC between the
record of each image and every other image of the collection (same key).

Usage: python experiments/run_2026_v4.py <datasets2026_dir> <out_dir> <feature> [q|none]
"""
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

D26, OUT = sys.argv[1], sys.argv[2]
FEAT = sys.argv[3]
QARG = sys.argv[4] if len(sys.argv) > 4 else "none"
sys.argv = [sys.argv[0], "/nonexistent", OUT, FEAT, QARG, "0"]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_v4 as RA                                      # noqa: E402
from watermarking.zero_watermark import otsu_roi, nc, ber    # noqa: E402

COLLECTIONS = ["BrainTumorMRI_Nickparvar", "CT_Kidney", "BrainMRI_Navoneel", "TCGA_LGG_Buda",
               "MIAS", "BoneBreak", "RetinaFundus", "COVID_CXR_Cohen"]


def load(col):
    out = []
    for f in sorted(glob.glob(os.path.join(D26, col, "*"))):
        if f.endswith((".json", ".csv", ".txt")):
            continue
        im = Image.open(f)
        if col == "TCGA_LGG_Buda":
            im = Image.fromarray(np.asarray(im.convert("RGB"))[:, :, 1])
        else:
            im = im.convert("L")
        im = im.resize((512, 512), Image.LANCZOS)
        nm = col + "/" + os.path.basename(f)
        out.append((nm, np.array(im, np.uint8),
                    dict(PatientID=nm, StudyInstanceUID=RA.uid(nm), Modality="OT", SeriesDate="UNKNOWN")))
    return out


def run_one(args):
    name, img, meta = args
    zw, _ = RA.zw_for(meta)
    roi = otsu_roi(img)
    rec = zw.register(img, RA.WM, roi)
    ext = zw.extract(img, rec)
    return dict(name=name, nc_clean=nc(RA.WM, ext), ber_clean=ber(RA.WM, ext),
                roi_block_fraction=round(rec["roi_block_fraction"], 4),
                attacks=RA.attack_battery(img, zw, rec, RA.MED2D)), (name, img, rec, meta)


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    res = {}
    with Pool(int(os.environ.get("NPROC", "2"))) as pool:
        for col in COLLECTIONS:
            data = load(col)
            r = pool.map(run_one, data)
            res[col] = [x[0] for x in r]
            recs = [x[1] for x in r]
            dist = []
            for i, (ni, _, reci, meta) in enumerate(recs):
                zw, _ = RA.zw_for(meta)
                for j, (nj, imgj, _, _) in enumerate(recs):
                    if i != j:
                        dist.append(round(nc(RA.WM, zw.extract(imgj, reci)), 4))
            res[col + "_distinguish"] = dist
            print(col, "done", round(time.time() - t0), flush=True)
    res["config"] = dict(features=FEAT, select_q=RA.Q, collections=COLLECTIONS)
    json.dump(res, open(os.path.join(OUT, f"results_2026_{RA.TAG}.json"), "w"), indent=1)
    print("ALL DONE", time.time() - t0)
