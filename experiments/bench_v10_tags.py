# experiments/bench_v10_tags.py  (revision 10 protocol review; REVISION-9 TEST data only)
"""Checks on the seed-0 test cache, for every rev-9 selected code configuration:
  (1) creedsolo and the pure-Python reedsolo give identical statuses on every attempt;
  (2) message invariance: the decodability of an attempt (decoded within the radius or not) and the
      condition do not depend on the message (content + tag), because the codes are linear;
      checked by registering two different messages (and, where feasible, 64- and 128-bit tags);
  (3) runtime per attempt with creedsolo.
Usage: python experiments/bench_v10_tags.py <out.json>
"""
import importlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import v10_decision as VD                                               # noqa: E402
from eval_test_decoders import z_of, STD, CLIN                          # noqa: E402
from eval_cached import meta_2d                                         # noqa: E402
from core.key_generator import PatientKey                               # noqa: E402
import reedsolo as pyrs                                                 # noqa: E402

out = sys.argv[1]
C = np.load("/home/claude/runs/cache/seed0_test.npy", allow_pickle=True).item()
CAL = json.load(open("/home/claude/runs/calib/calibrated_seed0.json"))["selected"]["modality_aware"]
MOD = {"brats100": "brats", "nih100": "nih", "isic100": "isic"}
res = {}
for s, E in C["sets"].items():
    if s not in MOD:
        continue
    cond_l = C["conditions"][1:]; idx = [cond_l.index(n) for n in STD + CLIN]
    Z0 = np.stack([z_of(e)[0] for e in E]); Za = np.stack([z_of(e)[1:] for e in E])[:, idx]
    for f in ("HRS64", "HRS8", "DNA4"):
        v = CAL[f].get(MOD[s]) if CAL[f] else None
        if v is None:
            continue
        cfg = v["cfg"]; R = dict(cfg=cfg, py_vs_c_mismatch=0, msg_invariance_mismatch=0, n=0, tags={})
        rs = np.random.RandomState(1); tc = 0.0
        for i, e in enumerate(E[:50]):
            key = PatientKey(meta_2d(e["name"])).sub("hrs")
            Zall = np.concatenate([Za[i], np.delete(Z0, i, 0)])
            dec_ref = None
            for T in (32, 64, 128):
                try:
                    cd = VD.Coder(key, cfg, T)
                except AssertionError:
                    R["tags"][T] = "infeasible (< 64 content bits)"; continue
                for rep in range(2):
                    rec = cd.register(Z0[i], list(rs.randint(0, 2, cd.content_bits)))
                    a = time.time(); cnd, st = cd.verify_batch(Zall, rec); tc += time.time() - a
                    decd = [x != "reject_fail" for x in st]
                    if dec_ref is None:
                        dec_ref = (cnd.copy(), decd)
                        cd_py = VD.Coder(key, cfg, T); cd_py.rs = pyrs.RSCodec(nsym=cd.nc - cd.K, nsize=2 ** cd.m - 1,
                                                                             c_exp=cd.m, prim=VD.PRIM[cd.m], generator=2)
                        VD_rs = VD.reedsolo; VD.reedsolo = pyrs
                        _, st_py = cd_py.verify_batch(Zall, rec); VD.reedsolo = VD_rs
                        R["py_vs_c_mismatch"] += sum(a != b for a, b in zip(st, st_py))
                    else:
                        R["msg_invariance_mismatch"] += int((cnd != dec_ref[0]).sum()) + sum(a != b for a, b in zip(decd, dec_ref[1]))
                    R["n"] += len(st)
                    R["tags"][T] = cd.content_bits
        R["ms_per_attempt_creedsolo"] = 1e3 * tc / R["n"]
        res[f"{s}|{f}"] = R
        print(s, f, json.dumps(R), flush=True)
json.dump(res, open(out, "w"), indent=1)
