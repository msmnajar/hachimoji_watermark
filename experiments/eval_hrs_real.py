# experiments/eval_hrs_real.py  (revision 7)
"""Proposed full-base Hachimoji RS layer with the REAL decoder (watermarking/hachimoji_rs.py),
validation-frozen parameters, on a test set: genuine acceptance (exact content recovery) after each
operation, and false acceptances over all ordered pairs of different patients (record of i, clean
image of j, key of i).  The content field is random per image (seed 1).
Usage: python experiments/eval_hrs_real.py <cache.npy> <set> <sel_decoders.json> <out.json>
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

cache_f, set_name, sd, out = sys.argv[1:5]
C = np.load(cache_f, allow_pickle=True).item()
E = C["sets"][set_name]; cond = C["conditions"][1:]
c = json.load(open(sd))["families"]["A_other_patient"]["HRS64"]["selected"]
rs = np.random.RandomState(1)
H, recs, contents = [], [], []
for e in E:
    key = PatientKey(meta_2d(e["name"])).sub("hrs")
    h = HachimojiRS(key, q=c["q"], m=6, nsym=c["nc"] - c["k"], k_info=c["k"], erase=c["s"], interleave=c["il"])
    content = list(rs.randint(0, 2, h.payload_bits - 32))
    H.append(h); contents.append(content); recs.append(h.register(z_of(e)[0], content))
acc = np.zeros((len(E), len(cond)), bool)
for i, e in enumerate(E):
    Z = z_of(e)
    for ci in range(len(cond)):
        ok, got = H[i].verify(Z[ci + 1], recs[i])
        acc[i, ci] = ok and got == contents[i]
fa = 0; fa_pairs = []
for i in range(len(E)):
    for j in range(len(E)):
        if i != j:
            ok, _ = H[i].verify(z_of(E[j])[0], recs[i])
            if ok:
                fa += 1; fa_pairs.append((E[i]["name"], E[j]["name"]))
idx = [cond.index(n) for n in STD + CLIN]
res = dict(set=set_name, config=c, payload_content_bits=H[0].payload_bits - 32, tag_bits=32,
           GAR_24=float(acc[:, idx].mean()), GAR_34=float(acc.mean()),
           per_condition={n: float(acc[:, k].mean()) for k, n in enumerate(cond)},
           FA=fa, n_impostor=len(E) * (len(E) - 1), fa_pairs=fa_pairs)
json.dump(res, open(out, "w"), indent=1)
print(set_name, "GAR24", round(res["GAR_24"], 4), "GAR34", round(res["GAR_34"], 4), "FA", fa, "/", res["n_impostor"])
