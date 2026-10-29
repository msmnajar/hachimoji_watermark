# experiments/eval_hrs_real_cal.py  (revision 9)
"""Calibrated Hachimoji decoder with the REAL decoder (watermarking/hachimoji_rs.py) on the test sets:
for each collection, the Hachimoji family (GF(64) dinucleotide or GF(8) base code) with the higher
CALIBRATION acceptance within the false-acceptance budget, with its calibrated parameters.
Genuine acceptance = exact content recovery; false acceptances over all 9900 ordered pairs.
Usage: python experiments/eval_hrs_real_cal.py <cache_test.npy> <calibrated.json> <out.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from eval_cached import meta_2d                                    # noqa: E402
from eval_test_decoders import z_of, STD, CLIN                     # noqa: E402
from watermarking.hachimoji_rs import HachimojiRS                  # noqa: E402
from core.key_generator import PatientKey                          # noqa: E402

MOD = {"brats100": "brats", "nih100": "nih", "isic100": "isic"}
cache, calj, out = sys.argv[1:4]
C = np.load(cache, allow_pickle=True).item()
CAL = json.load(open(calj))["selected"]["modality_aware"]
res = {}
for s, E in C["sets"].items():
    m = MOD[s]
    cands = [(f, CAL[f][m]) for f in ("HRS64", "HRS8") if CAL[f].get(m)]
    f, v = max(cands, key=lambda x: x[1]["GAR"])
    c = v["cfg"]
    cond = C["conditions"][1:]; idx = [cond.index(n) for n in STD + CLIN]
    rs = np.random.RandomState(1)

    def mk(key):
        return HachimojiRS(key, q=c["q"], m=c["m"], nsym=c["nc"] - c["k"], k_info=c["k"], erase=c["s"],
                           interleave=c["il"], blocks=c["nblk"])
    H, R, CT = [], [], []
    for e in E:
        h = mk(PatientKey(meta_2d(e["name"])).sub("hrs"))
        ct = list(rs.randint(0, 2, h.payload_bits - 32))
        H.append(h); CT.append(ct); R.append(h.register(z_of(e)[0], ct))
    acc = np.zeros((len(E), len(idx)), bool)
    for i, e in enumerate(E):
        Z = z_of(e)
        for a, ci in enumerate(idx):
            ok, got = H[i].verify(Z[ci + 1], R[i]); acc[i, a] = ok and got == CT[i]
    fa = sum(int(H[i].verify(z_of(E[j])[0], R[i])[0]) for i in range(len(E)) for j in range(len(E)) if i != j)
    res[s] = dict(family=f, cfg=c, content_bits=H[0].payload_bits - 32, GAR=float(acc.mean()), FA=fa,
                  per_image=acc.mean(1).tolist(), per_op={STD[k] if k < len(STD) else CLIN[k - len(STD)]: float(acc[:, k].mean()) for k in range(len(idx))})
    print(s, f, "content bits", res[s]["content_bits"], "GAR", round(res[s]["GAR"], 4), "FA", fa, flush=True)
json.dump(res, open(out, "w"))
