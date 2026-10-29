# Revision 10: pre-registered evaluation on an untouched cohort

This revision adds the frozen evaluation protocol, its scripts, and the final results reported in the
revision-10 manuscript (decision rules; Section 4.2 and Section 5.5 / decision table).

## Contents
| Path | What |
|---|---|
| `results_v10/PROTOCOL_v10.md` | Frozen protocol (version 7; SHA-256 e34f5059…). `PROTOCOL_v10_PLAN*.md` are the reviewed drafts. |
| `results_v10/freeze_record.json` | SHA-256 of the protocol, manifest, every script and its import closure, the three weights, calibration outputs and the frozen configurations; git commit 9d2dbf46. |
| `results_v10/frozen_v10.json` | Selected configuration of every seed × modality × family × tag length × operating point ("none" = no configuration met the budget). |
| `results_v10/manifest_v10.json`, `*.lst`, `isic.csv`, `check_manifest_output.txt` | Patient partitions (pool, test, calibration cohort, untouched cohort) and the machine check. Identifiers only; no images. |
| `results_v10/calib_selection_seed*.json` | Calibration on the calibration cohort (300 patients per modality). |
| `results_v10/conformance_final.json`, `conf/`, `bench/` | Conformance of every selected configuration against the independent reference (`experiments/ref_v10.py`); benchmarks. |
| `results_v10/U_results_seed*.json`, `U_pairs_seed*.npy` | Final evaluation on the untouched cohort: per-rule counts, statistics, and the accepted ordered pairs (draw indices). |
| `results_v10/verified/` | Verified copies with `SHA256SUMS.txt`, `verify_seed*.txt` (post-run checks by `verify_seed_v10.py`), `final_results_long.json` (one row per seed × modality × rule). |
| `results_v10/revision10_U_final_results.md` | Consolidated results report. |
| `results_v10/DEVIATIONS.md`, `access_log_U.txt`, `logs/` | Incidents (truncated ISIC image; restarts), every guarded access, run logs including failed and interrupted runs. |
| `models/aifnet_v4.pt`, `models/aifnet_v4_seed1.pt`, `models/aifnet_v4_seed2.pt` | AIF-Net weights of training seeds 0, 1, 2 (hashes in the freeze record; seeds 1 and 2 were stored at `runs/multi/seed{1,2}/aifnet_v4.pt` when frozen). |

## Order of operations
`build_manifest_v10.py` → `check_manifest.py` → `calib_v10.py` (seeds 0–2) → `conformance_v10.py final` →
`make_freeze_v10.py` → `eval_final_v10.py` (seeds 0–2; refuses to run unless every hash matches the freeze record)
→ `results_v10/verify_seed_v10.py`.

## Not included
- Medical images (available from the BraTS 2023, NIH ChestX-ray14 and ISIC 2020 providers) and feature caches (`*.npz`).
- Third-party pretrained weights used by the re-implemented baselines (`models/pretrained/`).
- The `reedsolo-1.7.0` source archive hashed in the freeze record (available from PyPI).
