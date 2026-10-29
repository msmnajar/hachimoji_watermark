# experiments/build_manifest_v10.py  (revision 10)
"""Builds manifest_v10.json: every patient and file of every set, per modality, from release listings only
(no image content is read).

Sets per modality: pool (training/validation), test (revision-9 test set), calib (C), untouched (U).
Patient identifiers: BraTS = 5-digit case number (training and validation releases together);
NIH = 8-digit patient number; ISIC = patient_id of ISIC_2020_Training_GroundTruth.csv.
One image per patient: BraTS = first time point (training release preferred), middle axial T1c slice;
NIH = <patient>_000.png; ISIC = the patient's first image name in sorted order.
Selection of the new patients (simple random sampling, numpy RandomState(2026), from all eligible patients,
i.e. every patient of the release that is not in pool, test or the revision-9 calibration set):
  BraTS: random permutation of the 1142 eligible patients -> first 300 = C, remaining 842 = U (census)
  NIH  : C = the revision-9 calibration patients (200) + first 100 of the permutation;
         U = 1000 draws WITH replacement (numpy RandomState(2028).choice) from the remaining eligible patients
         (population P = 30,305), kept in draw order (repeated patients possible), so the draws are exactly
         i.i.d. from that finite population
  ISIC : C = the revision-9 calibration patients (200) + first 100 of the permutation;
         U = ALL remaining eligible patients (1557; census of the remainder)
Usage: python experiments/build_manifest_v10.py <brats_all.lst> <nih_all.lst> <isic_groundtruth.csv> <out.json>
"""
import csv
import json
import sys

import numpy as np

RUNS = "/home/claude/runs"
SEED = 2026


def main(brats_lst, nih_lst, isic_csv, out):
    P = json.load(open(f"{RUNS}/partitions.json"))
    C9 = json.load(open(f"{RUNS}/calibration_selection.json"))
    M = dict(seed=SEED, rule=__doc__.split("Usage:")[0].strip(), modalities={})

    # ---------------- BraTS
    pid = lambda case: int(case.split("-")[2])
    rel = {}
    for line in sorted(l.strip() for l in open(brats_lst) if l.strip()):
        d, case = line.split("/")
        p = pid(case)
        cand = (0 if "trainingdata" in d.lower() else 1, case.split("-")[3], d, case)
        if p not in rel or cand < rel[p]:
            rel[p] = cand
    f_of = lambda p: f"brats2023/{rel[p][2]}/{rel[p][3]}/{rel[p][3]}-t1c.nii.gz"
    pool = sorted(pid(c) for c in P["brats_train"] + P["brats_validation"]); test = sorted(pid(c) for c in P["brats_test"])
    elig = sorted(set(rel) - set(pool) - set(test))
    perm = list(np.random.RandomState(SEED).permutation(elig))
    sets = dict(pool=pool, test=test, calib=sorted(int(x) for x in perm[:300]), untouched=sorted(int(x) for x in perm[300:]))
    M["modalities"]["brats"] = {k: [dict(patient=f"{p:05d}", file=f_of(p)) for p in v] for k, v in sets.items()}
    M["modalities"]["brats"]["n_eligible"] = len(elig)
    M["modalities"]["brats"]["untouched_design"] = dict(kind="census of all remaining eligible patients", population_size=len(elig) - 300)

    # ---------------- NIH
    names = sorted(l.strip() for l in open(nih_lst) if l.strip().endswith(".png"))
    first = {}
    for n in names:
        p = n.split("_")[0]
        first[p] = min(first.get(p, n), n)
    assert all(first[p].endswith("_000.png") for p in first), "a patient has no _000 image"
    p_of = lambda n: n.split("_")[0]
    pool = sorted(p_of(n) for n in P["nih_train_and_validation"]); test = sorted(p_of(n) for n in P["nih_test"])
    cal9 = sorted(p_of(n) for n in C9["nih_calibration"])
    elig = sorted(set(first) - set(pool) - set(test) - set(cal9))
    perm = list(np.random.RandomState(SEED).permutation(elig))
    rest = sorted(str(x) for x in perm[100:])
    draws = [str(x) for x in np.random.RandomState(2028).choice(rest, 1000, replace=True)]
    sets = dict(pool=pool, test=test, calib=sorted(cal9 + [str(x) for x in perm[:100]]), untouched=draws)
    M["modalities"]["nih"] = {k: [dict(patient=p, file=f"NIH_ChestXray14/images/{first[p]}") for p in v] for k, v in sets.items()}
    M["modalities"]["nih"]["n_eligible"] = len(elig)
    M["modalities"]["nih"]["untouched_design"] = dict(kind="iid draws with replacement", population_size=len(rest),
                                                      draws=len(draws), distinct_patients=len(set(draws)))

    # ---------------- ISIC
    rows = list(csv.DictReader(open(isic_csv)))
    pat = {r["image_name"]: r["patient_id"] for r in rows}
    firsti = {}
    for n in sorted(pat):
        firsti.setdefault(pat[n], n)
    pool = sorted(set(pat[n] for n in P["isic_train_and_validation"])); test = sorted(set(pat[n] for n in P["isic_test"]))
    cal9 = sorted(set(pat[n] for n in C9["isic_calibration"]))
    elig = sorted(set(firsti) - set(pool) - set(test) - set(cal9))
    perm = list(np.random.RandomState(SEED).permutation(elig))
    sets = dict(pool=pool, test=test, calib=sorted(cal9 + [str(x) for x in perm[:100]]), untouched=sorted(str(x) for x in perm[100:]))
    # pool/test/rev-9 calibration keep the images actually used (not necessarily the patient's first image)
    used = {}
    for n in P["isic_train_and_validation"] + P["isic_test"] + C9["isic_calibration"]:
        used.setdefault(pat[n], n)
    M["modalities"]["isic"] = {k: [dict(patient=p, file=f"ISIC2020/Training_JPEG/train/{used.get(p, firsti[p])}.jpg") for p in v]
                               for k, v in sets.items()}
    M["modalities"]["isic"]["n_eligible"] = len(elig)
    M["modalities"]["isic"]["untouched_design"] = dict(kind="census of all remaining eligible patients", population_size=len(elig) - 100)
    json.dump(M, open(out, "w"), indent=1)
    print({m: {k: len(v) if isinstance(v, list) else v for k, v in d.items()} for m, d in M["modalities"].items()})


if __name__ == "__main__":
    main(*sys.argv[1:5])
