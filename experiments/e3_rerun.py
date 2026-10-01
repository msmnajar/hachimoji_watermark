import sys, json, glob, numpy as np, pydicom
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "../../datasets", sys.argv[2] if len(sys.argv) > 2 else "results"]
sys.path.insert(0, ".")
import experiments.run_all_v2 as R
from multiprocessing import Pool
files = [f for f in sorted(glob.glob(f"{R.ROOT}/TCIA/RIDER_Breast_MRI/*/*.dcm"))
         if (pydicom.dcmread(f).pixel_array > 0).mean() >= 0.05]
print("usable DICOM files", len(files))
with Pool(2) as p:
    r = p.map(R.run_dicom, files)
res = json.load(open(f"{R.OUT}/results_v2.json"))
res["E3_tcia"] = [x[0] for x in r]
metas = [x[1] for x in r]
kt = []
for meta, img, rec in metas:
    other = next(m for (m, _, _) in metas if m["PatientID"] != meta["PatientID"])
    kt.append(R.key_tests(meta, img, rec, other))
res["E3_keytests"] = kt
res["E3_excluded_blank"] = 30 - len(files)
json.dump(res, open(f"{R.OUT}/results_v2.json", "w"), indent=1)
print("done")
