# experiments/bench_v10.py  (revision 10 protocol review: benchmark on REVISION-9 TEST data only)
"""For the rules selected in revision 9 (calibrated_seed<s>.json, modality-aware) and a 32-bit tag:
  1. exactness: does the bounded-distance condition (computed against the registrant's own symbols,
     with the decoder's own key-stream order) coincide with 'accept_genuine' of the real decoder
     for every genuine attempt and every impostor pair?
  2. mechanism counts (accept_genuine / accept_other / reject_fail / reject_tag),
  3. comparison with the unenforced revision-9 decoder (hachimoji_rs.py, Hachimoji families only),
  4. runtime per attempt.
No new-cohort data are read.
Usage: python experiments/bench_v10.py <seed> <out.json>
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
from v10_decision import Coder                                         # noqa: E402
from eval_test_decoders import z_of, STD, CLIN                          # noqa: E402
from eval_cached import meta_2d                                         # noqa: E402
from core.key_generator import PatientKey                               # noqa: E402

seed, out = int(sys.argv[1]), sys.argv[2]
C = np.load(f"/home/claude/runs/cache/seed{seed}_test.npy", allow_pickle=True).item()
CAL = json.load(open(f"/home/claude/runs/calib/calibrated_seed{seed}.json"))["selected"]["modality_aware"]
MOD = {"brats100": "brats", "nih100": "nih", "isic100": "isic"}
FAMS = [f for f in os.environ.get("FAMS", "HRS64,HRS8,DNA4,BCH").split(",")]
res = {}
for s, E in C["sets"].items():
    if s not in MOD:
        continue
    cond_l = C["conditions"][1:]; idx = [cond_l.index(n) for n in STD + CLIN]
    Z0 = np.stack([z_of(e)[0] for e in E]); Za = np.stack([z_of(e)[1:] for e in E])[:, idx]
    for f in FAMS:
        v = CAL[f].get(MOD[s]) if CAL[f] else None
        if v is None:
            continue
        cfg = v["cfg"]; rs = np.random.RandomState(1)
        t0 = time.time(); G = dict(cond=0, n=0, mism=0); I = dict(cond=0, n=0, mism=0); stat = {"gen": {}, "imp": {}}
        tg = ti = 0.0
        for i, e in enumerate(E):
            cd = Coder(PatientKey(meta_2d(e["name"])).sub("hrs"), cfg, 32)
            rec = cd.register(Z0[i], list(rs.randint(0, 2, cd.content_bits)))
            a = time.time(); c1, st1 = cd.verify_batch(Za[i], rec); tg += time.time() - a
            a = time.time(); c2, st2 = cd.verify_batch(np.delete(Z0, i, 0), rec); ti += time.time() - a
            for (cnd, st, D, k) in ((c1, st1, G, "gen"), (c2, st2, I, "imp")):
                D["n"] += len(st); D["cond"] += int(cnd.sum())
                D["mism"] += int(sum((x == "accept_genuine") != bool(y) for x, y in zip(st, cnd)))
                for x in st:
                    stat[k][x] = stat[k].get(x, 0) + 1
        res[f"{s}|{f}"] = dict(cfg=cfg, content_bits=cd.content_bits, genuine=G, impostor=I, status=stat,
                               ms_per_genuine=1e3 * tg / G["n"], ms_per_impostor=1e3 * ti / I["n"])
        print(s, f, json.dumps(res[f"{s}|{f}"]["genuine"]), json.dumps(res[f"{s}|{f}"]["impostor"]), stat,
              "ms/gen %.3f ms/imp %.3f" % (1e3 * tg / G["n"], 1e3 * ti / I["n"]), flush=True)
json.dump(res, open(out, "w"), indent=1)
