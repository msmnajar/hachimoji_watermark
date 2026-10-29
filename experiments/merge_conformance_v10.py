# experiments/merge_conformance_v10.py  (revision 10)
"""Merges the final-conformance worker outputs (runs/v10/conf/conformance_final_*.json) into
runs/v10/conformance_final.json.  Every selected configuration of calib_selection_seed{0,1,2}.json must be present;
a configuration tested by two workers must have identical results (the tests are deterministic).
Usage: python experiments/merge_conformance_v10.py <out_dir>"""
import glob, json, os, sys

R = sys.argv[1]
need = {}
for s in (0, 1, 2):
    S = json.load(open(os.path.join(R, f"calib_selection_seed{s}.json")))
    for m, sel in S["selection"].items():
        for k, v in sel.items():
            if isinstance(v, dict) and "cfg" in v:
                need.setdefault(json.dumps([s, m, v["cfg"], v["T"]], sort_keys=True), []).append(k)
have, dup_conflicts, dup_agree, selftests = {}, 0, 0, []
for f in sorted(glob.glob(os.path.join(R, "conf", "conformance_final_*.json"))):
    d = json.load(open(f)); selftests.append(dict(file=os.path.basename(f), grs=d["grs_selftest"], interleave=d["interleave"]))
    for c in d["configs"]:
        k = json.dumps([c["seed"], c["modality"], c["cfg"], c["T"]], sort_keys=True)
        if k in have:
            same = all(have[k][x] == c[x] for x in ("n_attempts", "T1_mismatch", "T2_mismatch", "T3_mismatch", "T3_n",
                                                    "helper_mismatch", "status", "n_pipeline_pairs"))
            dup_agree += same; dup_conflicts += not same
        else:
            have[k] = c
missing = sorted(set(need) - set(have))
configs = [dict(have[k], labels=need[k]) for k in sorted(need) if k in have]
tot = {t: sum(c[t] for c in configs) for t in ("n_attempts", "T1_mismatch", "T2_mismatch", "T3_mismatch", "T3_n", "helper_mismatch")}
out = dict(source_files=[s["file"] for s in selftests], selftests=selftests, n_required=len(need), n_tested=len(configs),
           missing=missing, duplicates_agree=dup_agree, duplicate_conflicts=dup_conflicts, totals=tot, configs=configs,
           selftest_ok=all(all(v == 1.0 for v in s["grs"].values()) and s["interleave"]["mismatches"] == 0 for s in selftests))
json.dump(out, open(os.path.join(R, "conformance_final.json"), "w"), indent=1)
print({k: out[k] for k in ("n_required", "n_tested", "duplicates_agree", "duplicate_conflicts", "totals", "selftest_ok")}, "missing:", len(missing))
sys.exit(1 if (missing or dup_conflicts or not out["selftest_ok"] or any(tot[t] for t in tot if t.endswith("mismatch"))) else 0)
