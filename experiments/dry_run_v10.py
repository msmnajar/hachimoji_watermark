# experiments/dry_run_v10.py  (revision 10) - end-to-end dry run of the pipeline on REVISION-9 TEST data
"""Exercises calib_v10.select_modality, conformance_v10.final_config, eval_final_v10.eval_config / nc_eval and
stats_v10.far_report on the already-seen revision-9 test sets (seed-0 features), WITHOUT the approval and freeze
guards and without any calibration (C) or untouched (U) data.  Budgets are scaled to the 9900 test pairs
(OP-A: floor(1e-4 * 9900) = 0 would equal OP-B, so OP-A uses 8 as on C purely to exercise the code path).
Numbers produced here are pipeline tests, not results.
Usage: python experiments/dry_run_v10.py <out.json>
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "experiments"))
import calib_v10 as CV
import conformance_v10 as CF
import eval_final_v10 as EF
import stats_v10 as SV
from conformance_v10 import grid_configs

out = {}; t0 = time.time(); cfgs = grid_configs()
for mod in os.environ.get("MODS", "brats,nih,isic").split(","):
    F = EF.test_set_F(0, mod)
    F["gen_idx"] = np.arange(0, 100, 3); F["z_att"] = F["z_att"][::3]
    F["patients"] = F["names"].copy()
    Fsel = {k: v.copy() for k, v in F.items()}          # selection and conformance: no repeated draws (as on C)
    design = dict(kind="census of all remaining eligible patients", population_size=len(F["z"]))
    if mod == "nih":       # exercise the with-replacement design: draws 0, 1, 2 repeated at the end
        rep = [0, 1, 2]
        for k in ("z", "names", "cls", "patients"):
            F[k] = np.concatenate([F[k], F[k][rep]])
        design = dict(kind="iid draws with replacement", population_size=100)
    a = time.time(); sel, tab, nct, log = CV.select_modality(Fsel, cfgs, mod); t_sel = time.time() - a
    picked = {k: v for k, v in sel.items() if isinstance(v, dict) and "cfg" in v}
    conf, evals, seen = [], [], set()
    for lab, item in picked.items():
        key = json.dumps([item["cfg"], item["T"]], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        a = time.time(); st = CF.final_config(Fsel, item["cfg"], item["T"]); tc = time.time() - a
        a = time.time(); r, fa_pairs, sparse = EF.eval_config(F, item["cfg"], item["T"]); te = time.time() - a
        r["stats"] = SV.far_report(fa_pairs, len(F["z"]), patients=F["patients"], design=design)
        r["pair_count_check"] = sum(r["counts"].values()) == r["M"]
        conf.append(dict(label=lab, cfg=item["cfg"], T=item["T"], seconds=tc, **{k: st[k] for k in ("n_attempts", "T1_mismatch", "T2_mismatch", "T3_n", "T3_mismatch", "helper_mismatch", "n_pipeline_pairs")}))
        evals.append(dict(label=lab, seconds=te, **{k: r[k] for k in ("N", "M", "counts", "FA", "FA_a", "FA_b", "E_tag", "E_tag_idealized_expected_collisions", "genuine_accept", "pair_count_check")}, stats=r["stats"]))
        print(mod, lab, "T1", conf[-1]["T1_mismatch"], "T2", conf[-1]["T2_mismatch"], "T3", conf[-1]["T3_mismatch"], "/", conf[-1]["T3_n"],
              "reg", conf[-1]["helper_mismatch"], "| pairs", r["M"], "count-check", r["pair_count_check"], r["counts"],
              "| FA", r["stats"]["FA"], "same-patient acc", r["stats"]["same_patient_accepted"], "M1 %.2e" % r["stats"]["M1_upper"],
              "| %.0fs/%.0fs" % (tc, te), flush=True)
    taus = [("uncalibrated", 1.0)] + [(k, v["tau"]) for k, v in sel["NC"].items() if v != "none"]
    nc = EF.nc_eval(F, taus, design)
    print(mod, "NC", {k: (v["tau"], v["FA"], round(v["genuine_accept"], 4)) for k, v in nc.items()}, flush=True)
    print(mod, "selection %.0fs; selected %d; none: %s" % (t_sel, len(picked), [k for k, v in sel.items() if v == "none"]), flush=True)
    out[mod] = dict(select_seconds=t_sel, n_selected=len(picked), none=[k for k, v in sel.items() if v == "none"],
                    hachimoji=[(k, v) for k, v in sel.items() if k.startswith("Hachimoji")], nc_selection=sel["NC"],
                    conformance=conf, evaluation=evals, nc=nc, log=log)
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("total %.0fs" % (time.time() - t0))
