# experiments/eval_final_v10.py  (revision 10) - the ONLY script that reads the untouched cohort U
"""Guarded final evaluation (protocol Sections 5-7).

Guard (before any U file is opened):
  * freeze_record.json exists; the SHA-256 of every file it lists (protocol, frozen_v10.json, manifest,
    scripts, model weights, benchmark and conformance outputs) matches the file on disk;
  * the git working tree of the code is clean and HEAD equals the commit in the record;
  * one line is appended to access_log_U.txt (UTC time, seed, command, record hash) for every run.
Evaluation for one model seed:
  1. U features (features_v10.extract, set "untouched"), cached in <out_dir>/U_features_seed<s>_<mod>.npz;
  2. for every frozen configuration of this seed (identical configurations evaluated once):
       registration of every U patient (content: RandomState(1) in manifest order);
       FULL decoding of every ordered pair (i, j), i != j, of clean U images -> one status per pair:
         accept_genuine (mechanism a), accept_other (mechanism b), reject_fail, reject_tag;
       check: sum of status counts = N (N - 1);
       sparse lists of the pairs with status != reject_fail;
       genuine attempts: the 200 genuine U patients x 24 operations, and the revision-9 test set
       (100 x 24 genuine, 9900 impostor pairs) for continuity;
       E_tag = #reject_tag + #accept_other (pairs that reached a tag comparison) and the IDEALIZED expectation
       E_tag * 2^-T of tag collisions under the PRF assumption for HMAC-SHA256 (not an observed quantity);
  3. NC rule: exact vectorized NC for all U pairs and genuine attempts, at tau = 1.00 and the frozen tau;
  4. statistics (stats_v10.far_report): FA / M, M1 (Hoeffding U-statistic bound), M2 (patient-cluster
     bootstrap, reported only if FA >= 3), M3 (jackknife), patient-disjoint check.
Writes <out_dir>/U_results_seed<s>.json and the sparse pair lists.
Usage: python experiments/eval_final_v10.py <seed> <weights> <img_root> <out_dir>
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def guard(out_dir, seed):
    rp = os.path.join(out_dir, "freeze_record.json")
    if not os.path.exists(rp):
        raise SystemExit("GUARD: no freeze_record.json - protocol not frozen")
    R = json.load(open(rp)); bad = []
    for path, h in R["files"].items():
        if not os.path.exists(path) or sha(path) != h:
            bad.append(path)
    if bad:
        raise SystemExit(f"GUARD: {len(bad)} frozen files changed or missing, e.g. {bad[:3]}")
    st = subprocess.run(["git", "-C", HERE, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if st or head != R["git_commit"]:
        raise SystemExit("GUARD: code tree not clean or HEAD differs from the frozen commit")
    with open(os.path.join(out_dir, "access_log_U.txt"), "a") as f:
        f.write(json.dumps(dict(utc=datetime.datetime.utcnow().isoformat() + "Z", seed=seed, argv=sys.argv,
                                freeze_record_sha256=sha(rp), git_commit=head)) + "\n")
    return R


def statuses(cd, rec, Z):
    return cd.verify_batch(Z, rec)[1]


def eval_config(F, c, T, rs_seed=1):
    import v10_decision as VD
    from core.key_generator import PatientKey
    from eval_cached import meta_2d
    z, names, gi, za = F["z"], F["names"], F["gen_idx"], F["z_att"]
    N = len(z); rs = np.random.RandomState(rs_seed); gpos = {int(g): k for k, g in enumerate(gi)}
    counts = {}; sparse = []; gen = np.zeros((len(gi), za.shape[1]), object); t_gen = 0.0
    for i in range(N):
        cd = VD.Coder(PatientKey(meta_2d(str(names[i]))).sub("hrs"), c, T)
        rec = cd.register(z[i], list(rs.randint(0, 2, cd.content_bits)))
        others = [j for j in range(N) if j != i]
        st = statuses(cd, rec, z[others])
        for j, x in zip(others, st):
            counts[x] = counts.get(x, 0) + 1
            if x != "reject_fail":
                sparse.append((i, j, x))
        if i in gpos:
            a = time.time(); gen[gpos[i]] = statuses(cd, rec, za[gpos[i]]); t_gen += time.time() - a
    assert sum(counts.values()) == N * (N - 1), "pair-count check failed"
    fa_pairs = [(i, j) for i, j, x in sparse if x in ("accept_genuine", "accept_other")]
    e_tag = counts.get("reject_tag", 0) + counts.get("accept_other", 0)
    g = gen.astype(str)
    return dict(N=N, M=N * (N - 1), counts=counts, FA=len(fa_pairs),
                FA_a=counts.get("accept_genuine", 0), FA_b=counts.get("accept_other", 0),
                E_tag=e_tag, E_tag_idealized_expected_collisions=e_tag * 2.0 ** (-T),
                genuine_accept=float((g == "accept_genuine").mean()),
                genuine_status={k: int((g == k).sum()) for k in np.unique(g)},
                genuine_per_op=(g == "accept_genuine").mean(0).tolist(),
                ms_per_genuine_attempt=1e3 * t_gen / g.size), fa_pairs, sparse


def test_set_F(seed, mod):
    """Revision-9 test set (100 patients) from the existing feature cache (continuity; not confirmatory)."""
    from eval_test_decoders import z_of, STD, CLIN
    C = np.load(f"/home/claude/runs/cache/seed{seed}_test.npy", allow_pickle=True).item()
    E = C["sets"][{"brats": "brats100", "nih": "nih100", "isic": "isic100"}[mod]]
    cl = C["conditions"][1:]; idx = [cl.index(x) for x in STD + CLIN]
    return dict(z=np.stack([z_of(e)[0] for e in E]), names=np.array([e["name"] for e in E]),
                cls=np.stack([e["cls"] for e in E]), gen_idx=np.arange(len(E)),
                z_att=np.stack([z_of(e)[1:] for e in E])[:, idx])


def nc_eval(F, taus, design=None):
    import stats_v10 as SV
    import v10_decision as VD
    from eval_cached import zw_for, meta_2d, register
    from select_nc_tau import entry
    z, names, cls, gi, za = F["z"], F["names"], F["cls"], F["gen_idx"], F["z_att"]
    N = len(z); gpos = {int(g): k for k, g in enumerate(gi)}
    imp = np.zeros((N, N)); gen = np.zeros((len(gi), za.shape[1]))
    for i in range(N):
        e = entry([z[i]], cls[i]); zw = zw_for(meta_2d(str(names[i])), "learned_v4", 0.7); rec = register(zw, e)
        G = VD.nc_batch_fn(zw, e, rec)
        imp[i] = G((z > 0).astype(np.uint8)); imp[i, i] = np.nan
        if i in gpos:
            gen[gpos[i]] = G((za[gpos[i]] > 0).astype(np.uint8))
    out = {}
    for op, tau in taus:
        with np.errstate(invalid="ignore"):
            acc = imp >= tau - 1e-9
        pairs = [(int(i), int(j)) for i, j in zip(*np.where(acc))]
        out[op] = dict(tau=tau, FA=len(pairs), M=N * (N - 1), genuine_accept=float((gen >= tau - 1e-9).mean()),
                       stats=SV.far_report(pairs, N, patients=F.get("patients"), design=design))
    return out


def main(seed, weights, img_root, out_dir):
    R = guard(out_dir, seed)
    if os.path.abspath(weights) != os.path.abspath(R["weights"][f"seed{seed}"]):      # [v5] frozen weights only
        raise SystemExit("GUARD: weights differ from the frozen weights of this seed")
    import features_v10 as FV
    import stats_v10 as SV
    M = json.load(open(os.path.join(out_dir, "manifest_v10.json")))
    FZ = json.load(open(os.path.join(out_dir, "frozen_v10.json")))[str(seed)]
    res = dict(seed=seed, freeze_record_sha256=sha(os.path.join(out_dir, "freeze_record.json")), modalities={})
    for mod in ("brats", "nih", "isic"):
        fp = os.path.join(out_dir, f"U_features_seed{seed}_{mod}.npz")
        if not os.path.exists(fp):
            np.savez(fp, **FV.extract(M, mod, "untouched", weights, img_root))
        F = dict(np.load(fp)); FT = test_set_F(seed, mod); out = {}; done = {}
        design = M["modalities"][mod]["untouched_design"]
        for label, item in FZ[mod]["codes"].items():          # label = family|T|OP, item = cfg or "none"
            if item == "none":
                out[label] = "none"; continue
            key = json.dumps([item["cfg"], item["T"]], sort_keys=True)
            if key not in done:
                r, fa_pairs, sparse = eval_config(F, item["cfg"], item["T"])
                r["stats"] = SV.far_report(fa_pairs, len(F["z"]), patients=F["patients"], design=design)
                r["accepted_draw_pairs_incl_same_patient"] = r["FA"]       # raw count of accepted draw pairs
                r["FA"] = r["stats"]["FA"]                                 # impostor FA: different patients only
                rt, _, _ = eval_config(FT, item["cfg"], item["T"])
                r["test_set_rev9"] = {k: rt[k] for k in ("N", "M", "counts", "FA", "FA_a", "FA_b", "genuine_accept")}
                np.save(os.path.join(out_dir, f"U_pairs_seed{seed}_{mod}_{len(done)}.npy"), np.array(sparse, dtype=object))
                done[key] = r
            out[label] = dict(cfg=item["cfg"], T=item["T"], **done[key])
        taus = [("uncalibrated", 1.0)] + [(k, v["tau"]) for k, v in FZ[mod]["nc"].items() if v != "none"]
        nc_out = nc_eval(F, taus, design)
        out["NC"] = nc_out
        res["modalities"][mod] = out
        json.dump(res, open(os.path.join(out_dir, f"U_results_seed{seed}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4])
