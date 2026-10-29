# experiments/select_nc_tau.py  (revision 7)
"""NC rule of the parity zero-watermark on VALIDATION data, with the same genuine and impostor
sets as select_decoders.py, evaluated with the exact pipeline (CachedZW).
Reports genuine acceptance at the paper's tau = 1.00 and at the smallest tau with zero
validation impostor acceptance (the NC rule tuned in its own favour).
Usage: python experiments/select_nc_tau.py <val_logits.npz> <val_impostors.npz> <train_set.npz> <out.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "experiments"))
from eval_cached import zw_for, meta_2d, register, extract_nc            # noqa: E402
from select_decoders import impostor_pairs                              # noqa: E402
from watermarking.zero_watermark import otsu_roi, block_classes         # noqa: E402
from models import feature_net_v4 as FN4                                # noqa: E402

NG = FN4.N_GLOBAL


def entry(z_list, cls):
    nl = 512 - NG
    blk = np.concatenate([-np.ones(NG, np.int64), np.arange(nl) // FN4.N_LOCAL])
    grp = np.concatenate([np.arange(NG) // 8, NG // 8 + np.arange(nl) // FN4.N_LOCAL])
    z = np.stack(z_list)
    return dict(f=(z > 0).astype(np.uint8), rel=np.abs(z).astype(np.float32), blk=blk, grp=grp, grid=(8, 8), cls=cls)


def val_images(npz):
    _D = np.load(npz)
    D = {k: _D[k] for k in ("brats", "vol_id", "big", "n_volumes")}      # decompress once
    vol, nv = D["vol_id"], int(D["n_volumes"])
    imgs = []
    for v in range(nv - 8, nv):
        ids = [i for i in range(len(vol)) if vol[i] == v]
        for r in (0.3, 0.45, 0.6, 0.75):
            imgs.append(D["brats"][ids[int(r * (len(ids) - 1))]])
    imgs += [D["big"][i] for i in list(range(90, 100)) + list(range(190, 200))]
    return imgs


if __name__ == "__main__":
    vl, vi, npz, out = sys.argv[1:5]
    L = np.load(vl); V = np.load(vi)
    imgs = val_images(npz)
    gen = []
    for i, img in enumerate(imgs):
        cls = block_classes(otsu_roi(img), (8, 8), True)
        e = entry([L["clean"][i]] + list(L["att"][:, i]), cls)
        zw = zw_for(meta_2d(f"VAL_{i}"), "learned_v4", 0.7)
        rec = register(zw, e)
        gen.append([extract_nc(zw, e, c, rec) for c in range(1, e["f"].shape[0])])
    gen = np.array(gen)                                      # (I, A)
    imp = []
    for i, j, kind in impostor_pairs(V):
        e = entry([V["z"][i], V["z"][j]], V["cls"][i])
        zw = zw_for(meta_2d(f"VALS_{i}"), "learned_v4", 0.7)
        rec = register(zw, e)
        imp.append((extract_nc(zw, e, 1, rec), kind))
    taus = np.round(np.arange(0.80, 1.0001, 0.001), 3)
    res = dict(n_genuine=int(gen.size), n_impostor=len(imp),
               impostor_max_by_kind={k: max(v for v, kk in imp if kk == k) for k in set(k for _, k in imp)},
               GAR_tau_1=float((gen >= 1 - 1e-9).mean()), worst_tau_1=float((gen >= 1 - 1e-9).mean(0).min()),
               FA_tau_1_by_kind={k: int(sum(v >= 1 - 1e-9 for v, kk in imp if kk == k)) for k in set(k for _, k in imp)})
    for tm, kinds in (("A_other_patient", ("other_patient",)), ("B_all", ("other_patient", "neighbour_1", "neighbour_2"))):
        vals = [v for v, k in imp if k in kinds]
        ok = [t for t in taus if all(v < t - 1e-9 for v in vals)]
        t0 = float(min(ok)) if ok else None
        res[tm] = dict(tau_far0=t0, GAR=None if t0 is None else float((gen >= t0 - 1e-9).mean()),
                       worst=None if t0 is None else float((gen >= t0 - 1e-9).mean(0).min()))
    res["genuine_nc"] = gen.tolist()
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "genuine_nc"}, indent=1))
