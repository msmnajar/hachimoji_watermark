# experiments/hn_separation.py  (revision 7)
"""Separation of the hard-negative kinds on the validation volumes, per model:
agreement of the selected bits (q = 0.7, selected on the registered slice) between a slice and
(a) the slice of another patient at the nearest relative depth, (b) a slice 3 or (c) 6 slices away
in the same volume; and the fraction of pairs in which no selected bit differs (these pairs pass
the strict NC rule up to the key-stream sampling).
Usage: python experiments/hn_separation.py name=val_impostors.npz [...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from select_decoders import impostor_pairs, sel_idx            # noqa: E402

for arg in sys.argv[1:]:
    nm, f = arg.split("=")
    V = np.load(f)
    out = {}
    for i, j, kind in impostor_pairs(V):
        S = sel_idx(V["z"][i], 0.7)
        a = float(np.mean((V["z"][i, S] > 0) == (V["z"][j, S] > 0)))
        out.setdefault(kind, []).append(a)
    print(nm, {k: (round(float(np.mean(v)), 4), int(sum(x == 1.0 for x in v)), len(v)) for k, v in sorted(out.items())})
