# experiments/calib_v10.py  (revision 10) - calibration cohort C only; never reads the untouched cohort U
"""Steps (protocol Section 4):
  1. guard: runs only if approval_v10.json records the approved PROTOCOL_v10.md hash (written by the authors
     after the reviewer's explicit approval);
  2. features of C for one model seed (features_v10.extract), cached in <out_dir>/calib_features_seed<s>.npz;
  3. exact condition table on C for every grid configuration (bounded-distance condition, decoder's own symbols,
     key-stream order, erasure rule of Section 2.3): genuine acceptance (100 patients x 24 operations) and
     impostor count (all 300 x 299 ordered pairs), per modality;
  4. NC table: genuine and impostor NC (exact vectorized NC) and counts for every tau in 0.900..1.000;
  5. selection per family, tag length T in {32, 64, 128}, operating point OP-A (FA_C <= 8) and OP-B (FA_C = 0):
     highest genuine acceptance; ties: fewer FA_C, larger content, smaller q, index order before keyed order,
     grid order. No feasible configuration -> "none" (no fallback, no relaxation);
  6. full decoding (v10_decision.Coder.verify_batch) of the selected configuration on all C pairs and genuine
     attempts; if the full count (mechanisms a + b) breaks the budget, the next ranked configuration is
     taken, down the whole ranked list until one passes or all are exhausted (every step logged);
  7. Hachimoji decoder = the Hachimoji alphabet (GF(64) or GF(8)) with the higher genuine acceptance on C
     (tie: GF(64)).
Writes <out_dir>/calib_selection_seed<s>.json.
Usage: python experiments/calib_v10.py <seed> <weights> <manifest.json> <img_root> <out_dir>
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import v10_decision as VD                                                  # noqa: E402
from conformance_v10 import grid_configs                                   # noqa: E402
from core.key_generator import PatientKey                                  # noqa: E402
from eval_cached import zw_for, meta_2d, register                          # noqa: E402
from select_nc_tau import entry                                            # noqa: E402

TAGS = (32, 64, 128)
BUDGET = {"OP-A": 8, "OP-B": 0}                    # floor(1e-4 * 89,700) = 8
TAUS = np.round(np.arange(0.900, 1.0001, 0.001), 3)
FAMILIES = ("HRS64", "HRS8", "DNA4", "BCH")


def guard(protocol_path, approval_path):
    if not os.path.exists(approval_path):
        raise SystemExit("not approved: approval_v10.json missing")
    ap = json.load(open(approval_path))
    h = hashlib.sha256(open(protocol_path, "rb").read()).hexdigest()
    if ap.get("protocol_sha256") != h:
        raise SystemExit("approved protocol hash does not match PROTOCOL_v10.md")
    return ap


def group_key(c):
    return (c["fam"], c["q"], c["il"]) + ((c["nblk"], c["nc"]) if c["fam"] == "BCH" else ())


def condition_table(F, cfgs):
    """F: features of one modality. Returns per config: genuine accept matrix (n_gen, 24) and impostor count."""
    z, names, gi, za = F["z"], F["names"], F["gen_idx"], F["z_att"]
    N = len(z); res = {}
    groups = {}
    for i, c in enumerate(cfgs):
        groups.setdefault(group_key(c), []).append(i)
    gpos = {int(g): k for k, g in enumerate(gi)}
    for gk, members in groups.items():
        c0 = cfgs[members[0]]
        G = {m: np.zeros((len(gi), za.shape[1]), bool) for m in members}; FA = {m: 0 for m in members}
        for i in range(N):
            cd = VD.Coder(PatientKey(meta_2d(str(names[i]))).sub("hrs"), c0, 0, check_content=False)
            S = VD.sel_idx(z[i], c0["q"]); order, rules = cd.stream(len(S))
            sig0, _ = cd.symbols(z[i][S][None], order, rules)
            recv = [np.delete(z, i, 0)]
            if i in gpos:
                recv.append(za[gpos[i]])
            Zr = np.concatenate(recv)
            sig, rel = cd.symbols(Zr[:, S], order, rules)
            P = len(Zr); nb, nc = cd.nblk, cd.nc
            err = (sig != sig0).reshape(P, nb, nc)
            o = np.argsort(rel.reshape(P, nb, nc), -1, kind="stable")
            cum = np.concatenate([np.zeros((P, nb, 1), int), np.cumsum(np.take_along_axis(err, o, -1), -1)], -1)
            tot = err.sum(-1)
            for m in members:
                c = cfgs[m]; s = c["s"]
                bound = 2 * c["t"] if c["fam"] == "BCH" else c["nc"] - c["k"]
                acc = (2 * (tot - cum[..., s]) + s <= bound).all(-1)
                FA[m] += int(acc[:N - 1].sum())
                if i in gpos:
                    G[m][gpos[i]] = acc[N - 1:]
        for m in members:
            res[m] = (G[m], FA[m])
    return res


def nc_table(F):
    z, names, cls, gi, za = F["z"], F["names"], F["cls"], F["gen_idx"], F["z_att"]
    N = len(z); gpos = {int(g): k for k, g in enumerate(gi)}
    gen = np.zeros((len(gi), za.shape[1])); imp = []
    for i in range(N):
        e = entry([z[i]], cls[i]); zw = zw_for(meta_2d(str(names[i])), "learned_v4", 0.7); rec = register(zw, e)
        G = VD.nc_batch_fn(zw, e, rec)
        imp.append(G((np.delete(z, i, 0) > 0).astype(np.uint8)))
        if i in gpos:
            gen[gpos[i]] = G((za[gpos[i]] > 0).astype(np.uint8))
    imp = np.concatenate(imp)
    return {float(t): dict(GAR=float((gen >= t - 1e-9).mean()), FA=int((imp >= t - 1e-9).sum())) for t in TAUS}, gen, imp


def content_bits(c, T):
    return c["msg_bits"] - T


def rank(cfgs, table, fam, T, budget):
    cand = []
    for i, c in enumerate(cfgs):
        if c["fam"] != fam or content_bits(c, T) < 64:
            continue
        G, fa = table[i]
        if fa <= budget:
            cand.append((-float(G.mean()), fa, -content_bits(c, T), c["q"], int(c["il"]), i))
    return [x[-1] for x in sorted(cand)]


def full_check(F, c, T):
    """Full decoding of every C impostor pair and genuine attempt (mechanisms a and b)."""
    z, names, gi, za = F["z"], F["names"], F["gen_idx"], F["z_att"]
    N = len(z); rs = np.random.RandomState(1); gpos = {int(g): k for k, g in enumerate(gi)}
    st = {"imp": {}, "gen": {}}; gen = np.zeros((len(gi), za.shape[1]), bool)
    for i in range(N):
        cd = VD.Coder(PatientKey(meta_2d(str(names[i]))).sub("hrs"), c, T)
        rec = cd.register(z[i], list(rs.randint(0, 2, cd.content_bits)))
        _, s1 = cd.verify_batch(np.delete(z, i, 0), rec)
        for x in s1:
            st["imp"][x] = st["imp"].get(x, 0) + 1
        if i in gpos:
            _, s2 = cd.verify_batch(za[gpos[i]], rec)
            gen[gpos[i]] = [x in ("accept_genuine",) for x in s2]
            for x in s2:
                st["gen"][x] = st["gen"].get(x, 0) + 1
    fa = st["imp"].get("accept_genuine", 0) + st["imp"].get("accept_other", 0)
    return dict(FA_full=fa, GAR_full=float(gen.mean()), status=st)


def select_modality(F, cfgs, mod, budgets=BUDGET):
    """Selection for one modality (protocol Section 4): condition table, NC table, ranked selection with
    full-decoding check, Hachimoji alphabet choice, NC threshold."""
    log = []
    table = condition_table(F, cfgs)
    tab = [dict(GAR=float(table[i][0].mean()), FA=table[i][1]) for i in range(len(cfgs))]
    nct, _, _ = nc_table(F)
    sel = {}
    for fam in FAMILIES:
        for T in TAGS:
            for op, b in budgets.items():
                order = rank(cfgs, table, fam, T, b); chosen = None; steps = []
                for i in order:                      # no limit: until one passes or all are exhausted
                    chk = full_check(F, cfgs[i], T)
                    steps.append(dict(cfg_index=i, FA_full=chk["FA_full"], GAR_full=chk["GAR_full"]))
                    if chk["FA_full"] <= b:
                        chosen = dict(cfg_index=i, cfg=cfgs[i], T=T, content_bits=content_bits(cfgs[i], T),
                                      GAR_C=float(table[i][0].mean()), FA_C_condition=table[i][1], full=chk)
                        break
                sel[f"{fam}|T{T}|{op}"] = chosen if chosen else "none"
                sel[f"{fam}|T{T}|{op}|steps"] = steps
                log.append(f"{mod} {fam} T{T} {op}: {'none' if not chosen else chosen['cfg_index']} after {len(steps)} full checks")
    for T in TAGS:
        for op in budgets:
            a, b8 = sel[f"HRS64|T{T}|{op}"], sel[f"HRS8|T{T}|{op}"]
            ga = a["GAR_C"] if isinstance(a, dict) else -1; gb = b8["GAR_C"] if isinstance(b8, dict) else -1
            sel[f"Hachimoji|T{T}|{op}"] = "none" if ga < 0 and gb < 0 else ("HRS64" if ga >= gb else "HRS8")
    sel["NC"] = {}
    for op, b in budgets.items():
        ok = [t for t, v in nct.items() if v["FA"] <= b]
        if not ok:
            sel["NC"][op] = "none"
        else:                                       # highest acceptance; ties: fewer FA, larger tau
            best = max(ok, key=lambda t: (nct[t]["GAR"], -nct[t]["FA"], t))
            sel["NC"][op] = dict(tau=best, **nct[best])
    return sel, tab, nct, log


def main(seed, weights, manifest_path, img_root, out_dir):
    ap = guard(os.path.join(out_dir, "PROTOCOL_v10.md"), os.path.join(out_dir, "approval_v10.json"))
    M = json.load(open(manifest_path)); t0 = time.time(); log = []
    feats = {}
    for mod in ("brats", "nih", "isic"):
        fp = os.path.join(out_dir, f"calib_features_seed{seed}_{mod}.npz")
        if not os.path.exists(fp):
            import features_v10 as FV
            np.savez(fp, **FV.extract(M, mod, "calib", weights, img_root))
        feats[mod] = dict(np.load(fp))
        log.append(f"features {mod} ready {time.time() - t0:.0f}s")
    cfgs = grid_configs(); out = dict(seed=seed, approval=ap, n_configs=len(cfgs), selection={}, nc={}, tables={},
                                      weights=os.path.abspath(weights),                  # [v5] recorded for the freeze check
                                      weights_sha256=hashlib.sha256(open(weights, "rb").read()).hexdigest())
    for mod, F in feats.items():
        sel, tab, nct, lg = select_modality(F, cfgs, mod)
        out["selection"][mod] = sel; out["tables"][mod] = tab; out["nc"][mod] = nct; log += lg
    out["log"] = log; out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(out_dir, f"calib_selection_seed{seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
