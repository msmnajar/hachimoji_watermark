# experiments/make_freeze_v10.py  (revision 10)
"""Builds frozen_v10.json from the three calibration selections and writes freeze_record.json.
Refuses to freeze unless:
  * check_manifest.py passes on the manifest;
  * calib_selection_seed{0,1,2}.json exist and were produced under the approved protocol hash;
  * conformance_final.json (conformance_v10.py final mode on every selected configuration) reports
    0 mismatches in T1, T2, T3 and registration;
  * the git working tree is clean (all scripts committed).
freeze_record.json lists, with SHA-256: the protocol, frozen_v10.json, manifest, every script under experiments/,
models/, watermarking/, core/, the three model weights, the calibration selections and feature files, the
benchmark outputs (runs/v10/bench), the conformance outputs (runs/v10/conf), the manifest-check output;
the local import closure (dependency_closure.txt; refuses if any listed module is not hashed), the compiled
creedsolo module and the reedsolo source archive it was built from, the pure-Python reedsolo used by the reference,
the release listings the manifest was built from; and records: git commit, UTC time, pip freeze, python version,
uname, lscpu model name, galois version.
Usage: python experiments/make_freeze_v10.py <out_dir> <weights_seed0> <weights_seed1> <weights_seed2>
"""
import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=isinstance(cmd, str)).stdout.strip()


def frozen_list(out_dir):
    FZ = {}
    for s in (0, 1, 2):
        S = json.load(open(os.path.join(out_dir, f"calib_selection_seed{s}.json")))
        FZ[str(s)] = {}
        for mod, sel in S["selection"].items():
            codes = {}
            for T in (32, 64, 128):
                for op in ("OP-A", "OP-B"):
                    hach = sel[f"Hachimoji|T{T}|{op}"]
                    codes[f"Hachimoji|T{T}|{op}"] = "none" if hach == "none" else sel[f"{hach}|T{T}|{op}"]
                    for fam in ("DNA4", "BCH"):
                        codes[f"{fam}|T{T}|{op}"] = sel[f"{fam}|T{T}|{op}"]
            FZ[str(s)][mod] = dict(codes=codes, nc=sel["NC"])
    return FZ


def collect_files(out_dir, weights):
    files = [os.path.join(out_dir, f) for f in ("PROTOCOL_v10.md", "approval_v10.json", "frozen_v10.json", "manifest_v10.json",
                                                "check_manifest_output.txt", "conformance_final.json")]
    files += sorted(glob.glob(os.path.join(out_dir, "calib_selection_seed*.json")) + glob.glob(os.path.join(out_dir, "calib_features_seed*.npz")))
    files += sorted(glob.glob(os.path.join(out_dir, "bench", "*")) + glob.glob(os.path.join(out_dir, "conf", "*")))
    for d in ("experiments", "models", "watermarking", "core", "evaluation", "training"):
        files += sorted(glob.glob(os.path.join(HERE, d, "*.py")))
    files += sorted(glob.glob(os.path.join(HERE, "*.py")))               # config.py and other root modules
    files += [os.path.join(out_dir, f) for f in ("dependency_closure.txt", "brats_all.lst", "nih_all.lst", "isic.csv",
                                                 "reedsolo-1.7.0.tar.gz")]
    import creedsolo, reedsolo, galois                                   # compiled / third-party decoders
    files += [creedsolo.__file__, reedsolo.__file__]
    files += list(weights)
    clos = [l.strip() for l in open(os.path.join(out_dir, "dependency_closure.txt")) if l.strip().endswith(".py")
            and "/" in l]
    missing = [c for c in clos if os.path.join(HERE, c) not in files]
    if missing:
        raise SystemExit(f"dependency closure not covered by the record: {missing}")
    return files, galois


def main(out_dir, *weights):
    chk = run([sys.executable, os.path.join(HERE, "experiments", "check_manifest.py"), os.path.join(out_dir, "manifest_v10.json"),
               os.path.join(out_dir, "brats_all.lst"), os.path.join(out_dir, "nih_all.lst"), os.path.join(out_dir, "isic.csv")])
    open(os.path.join(out_dir, "check_manifest_output.txt"), "w").write(chk)
    if "PASS: all checks" not in chk:
        raise SystemExit("manifest check failed")
    ap = json.load(open(os.path.join(out_dir, "approval_v10.json")))
    for s in (0, 1, 2):
        cs = json.load(open(os.path.join(out_dir, f"calib_selection_seed{s}.json")))
        if cs["approval"] != ap:
            raise SystemExit(f"calibration seed {s} not produced under the approved protocol")
        if cs.get("weights_sha256") != sha(weights[s]):                    # [v5] same weights in calibration and freeze
            raise SystemExit(f"calibration seed {s} used different weights from those being frozen")
    cf = json.load(open(os.path.join(out_dir, "conformance_final.json")))
    if any(c[k] for c in cf["configs"] for k in ("T1_mismatch", "T2_mismatch", "T3_mismatch", "helper_mismatch")):
        raise SystemExit("conformance failures - cannot freeze")
    if run(["git", "-C", HERE, "status", "--porcelain"]):
        raise SystemExit("git tree not clean")
    FZ = frozen_list(out_dir)
    json.dump(FZ, open(os.path.join(out_dir, "frozen_v10.json"), "w"), indent=1, sort_keys=True)
    files, galois = collect_files(out_dir, weights)
    rec = dict(utc=datetime.datetime.utcnow().isoformat() + "Z", git_commit=run(["git", "-C", HERE, "rev-parse", "HEAD"]),
               python=sys.version, platform=platform.platform(), uname=run("uname -a"),
               cpu=run("lscpu | grep 'Model name'"), pip_freeze=run([sys.executable, "-m", "pip", "freeze"]).splitlines(),
               weights={f"seed{i}": w for i, w in enumerate(weights)},
               versions=dict(galois=galois.__version__, creedsolo_source="reedsolo 1.7.0 sdist, built locally with Cython"),
               files={p: sha(p) for p in files})
    json.dump(rec, open(os.path.join(out_dir, "freeze_record.json"), "w"), indent=1)
    print("frozen:", len(files), "files; record sha256", sha(os.path.join(out_dir, "freeze_record.json")))


if __name__ == "__main__":
    if sys.argv[1] == "--check-coverage":          # pre-freeze self-test: which files would be hashed
        fl, _ = collect_files(sys.argv[2], sys.argv[3:])
        print(len(fl), "files would be recorded; present:", sum(os.path.exists(f) for f in fl))
        for f in fl:
            print(("OK  " if os.path.exists(f) else "MISSING (created by the pipeline before freezing) ") + f)
    else:
        main(*sys.argv[1:])
