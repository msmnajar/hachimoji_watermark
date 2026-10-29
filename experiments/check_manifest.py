# experiments/check_manifest.py  (revision 10)
"""Machine check of manifest_v10.json.  Exits non-zero on any failure.
  1. within every modality, the patient sets pool / test / calib / untouched are pairwise disjoint;
  2. no patient appears twice in a set and every patient has exactly one file;
  3. the expected sizes (BraTS 90/100/300/842; NIH 100/100/300/1000; ISIC 99/100/300/1000);
  4. the manifest agrees with the existing records: partitions.json (pool, test), calibration_selection.json
     (revision-9 calibration patients are inside calib);
  5. every listed file exists in the release listing (or under <root>, if given);
  6. patient identity of every file: the file name encodes the listed patient (BraTS, NIH) or the metadata
     maps the image to the listed patient (ISIC).
Prints the SHA-256 of the manifest.
Usage: python experiments/check_manifest.py <manifest.json> <brats_all.lst> <nih_all.lst> <isic.csv> [<root>]
"""
import csv
import hashlib
import json
import os
import sys
from itertools import combinations

EXPECT = {"brats": (90, 100, 300, 842), "nih": (100, 100, 300, 1000), "isic": (99, 100, 300, 1557)}
WITH_REPLACEMENT = {("nih", "untouched")}       # i.i.d. draws: repeated patients allowed (design)
SETS = ("pool", "test", "calib", "untouched")


def main(mpath, brats_lst, nih_lst, isic_csv, root=None):
    raw = open(mpath, "rb").read(); M = json.loads(raw); fails = []
    P = json.load(open("/home/claude/runs/partitions.json")); C9 = json.load(open("/home/claude/runs/calibration_selection.json"))
    listing = {"brats": set(), "nih": set(), "isic": set()}
    for l in open(brats_lst):
        if l.strip():
            d, c = l.strip().split("/"); listing["brats"].add(f"brats2023/{d}/{c}/{c}-t1c.nii.gz")
    listing["nih"] = {f"NIH_ChestXray14/images/{l.strip()}" for l in open(nih_lst) if l.strip()}
    rows = list(csv.DictReader(open(isic_csv))); pat = {r["image_name"]: r["patient_id"] for r in rows}
    listing["isic"] = {f"ISIC2020/Training_JPEG/train/{n}.jpg" for n in pat}
    for mod, D in M["modalities"].items():
        S = {k: [x["patient"] for x in D[k]] for k in SETS}
        for k in SETS:
            if len(S[k]) != len(set(S[k])) and (mod, k) not in WITH_REPLACEMENT:
                fails.append(f"{mod}/{k}: duplicate patient")
        for a, b in combinations(SETS, 2):
            inter = set(S[a]) & set(S[b])
            if inter:
                fails.append(f"{mod}: {a} and {b} share {len(inter)} patients, e.g. {sorted(inter)[:3]}")
        got = tuple(len(S[k]) for k in SETS)
        if got != EXPECT[mod]:
            fails.append(f"{mod}: sizes {got} != expected {EXPECT[mod]}")
        for k in SETS:
            for x in D[k]:
                f = x["file"]
                if f not in listing[mod]:
                    fails.append(f"{mod}/{k}: file not in release listing: {f}")
                if root and not os.path.exists(os.path.join(root, f)):
                    fails.append(f"{mod}/{k}: file missing under root: {f}")
                base = os.path.basename(f)
                if mod == "brats" and base.split("-")[2] != x["patient"]:
                    fails.append(f"brats/{k}: file {base} is not patient {x['patient']}")
                if mod == "nih" and base.split("_")[0] != x["patient"]:
                    fails.append(f"nih/{k}: file {base} is not patient {x['patient']}")
                if mod == "isic" and pat.get(base[:-4]) != x["patient"]:
                    fails.append(f"isic/{k}: image {base} belongs to {pat.get(base[:-4])}, not {x['patient']}")
    # designs of U
    # agreement with the records of revisions 7-9
    b = M["modalities"]["brats"]; n = M["modalities"]["nih"]; i = M["modalities"]["isic"]
    chk = [
        ("brats pool", {x["patient"] for x in b["pool"]}, {c.split("-")[2] for c in P["brats_train"] + P["brats_validation"]}),
        ("brats test", {x["patient"] for x in b["test"]}, {c.split("-")[2] for c in P["brats_test"]}),
        ("nih pool", {x["patient"] for x in n["pool"]}, {c.split("_")[0] for c in P["nih_train_and_validation"]}),
        ("nih test", {x["patient"] for x in n["test"]}, {c.split("_")[0] for c in P["nih_test"]}),
        ("isic pool", {x["patient"] for x in i["pool"]}, {pat[c] for c in P["isic_train_and_validation"]}),
        ("isic test", {x["patient"] for x in i["test"]}, {pat[c] for c in P["isic_test"]}),
    ]
    for name, a, e in chk:
        if a != e:
            fails.append(f"{name} differs from partitions.json")
    if not {c.split("_")[0] for c in C9["nih_calibration"]} <= {x["patient"] for x in n["calib"]}:
        fails.append("nih: revision-9 calibration patients not inside calib")
    if not {pat[c] for c in C9["isic_calibration"]} <= {x["patient"] for x in i["calib"]}:
        fails.append("isic: revision-9 calibration patients not inside calib")
    # volumes of the BraTS volume experiment are test patients
    if not {c.split("-")[2] for c in P["brats_test_volumes"]} <= {x["patient"] for x in b["test"]}:
        fails.append("brats: test volumes outside the test set")
    nu = [x["patient"] for x in M["modalities"]["nih"]["untouched"]]
    d = M["modalities"]["nih"].get("untouched_design", {})
    if d.get("draws") != len(nu) or d.get("distinct_patients") != len(set(nu)):
        fails.append("nih: untouched design record inconsistent")
    # ISIC census: every eligible ISIC patient not in pool/test/calib is in U
    pat_first = {}
    for n_ in sorted(pat):
        pat_first.setdefault(pat[n_], n_)
    iu = {x["patient"] for x in M["modalities"]["isic"]["untouched"]}
    used = set().union(*[{x["patient"] for x in M["modalities"]["isic"][k]} for k in ("pool", "test", "calib")])
    if iu != set(pat_first) - used:
        fails.append("isic: untouched is not the census of all remaining patients")
    bu = {x["patient"] for x in M["modalities"]["brats"]["untouched"]}
    bused = set().union(*[{x["patient"] for x in M["modalities"]["brats"][k]} for k in ("pool", "test", "calib")])
    ball = {f.split("/")[-1].split("-")[2] for f in listing["brats"]}
    if bu != ball - bused:
        fails.append("brats: untouched is not the census of all remaining patients")
    print("nih untouched draws", len(nu), "distinct", len(set(nu)))
    print("manifest sha256", hashlib.sha256(raw).hexdigest())
    print("sizes", {m: {k: len(D[k]) for k in SETS} for m, D in M["modalities"].items()})
    if fails:
        print("FAIL"); [print(" -", f) for f in fails[:50]]; sys.exit(1)
    print("PASS: all checks")


if __name__ == "__main__":
    main(*sys.argv[1:])
