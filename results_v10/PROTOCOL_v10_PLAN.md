# Revision 10: pre-registered evaluation protocol (plan for review)

Status: **draft for review. Nothing below has been executed.**

## 0. What exists already (disclosed)

- **Cohort lists:** written on 26 Oct 2026 (`cohorts_v10.json`). They are built only from file names, patient identifiers and the existing manifests.
- **Image extraction on the laptop:** started, then stopped at your request. So far:
  - BraTS: 1142 middle T1c slices;
  - NIH: 987 of 1100 images resized;
  - ISIC: 0.
- **Nothing else.** No new-cohort image has been copied to the compute machine, passed through AIF-Net, or scored.
- **The 100-patient test sets were seen in revision 9.** Their false acceptances led to the calibration protocol, so they are not confirmatory. The untouched cohort U (Section 1) is the confirmatory set.

## 1. Partitions (patient level, machine-checkable)

A patient is identified as follows:
- BraTS: the 5-digit case number, across both the training and validation releases;
- NIH: the 8-digit patient number;
- ISIC: `patient_id` from the official metadata CSV.

One image per patient:
- BraTS: first time point, middle axial T1c slice;
- NIH: image `_000`;
- ISIC: the first image name of the patient.

| Set | BraTS | NIH | ISIC | Use |
|---|---|---|---|---|
| Training/validation pool (unchanged) | 90 | 100 | 99 | model training only |
| Test T (unchanged, seen in rev 9) | 100 | 100 | 100 | continuity with rev 9 |
| Calibration C | **300 new** (replaces the 8 validation volumes) | 300 (rev-9 200 + patients 301–400) | 300 (rev-9 200 + next 100) | rule selection only |
| Untouched U | 842 (all remaining BraTS patients) | 1000 (patients 401–1400) | 1000 (next 1000) | final evaluation only |

- **BraTS caveat:** U is smaller for BraTS (842) because only 1332 BraTS patients exist in the release.
- **Manifests:** `manifest_v10.json` lists every patient identifier and file per set and per modality.
- **Checker:** `check_manifest.py` asserts pairwise disjointness of all sets at patient level and that every file exists. Its output is part of the artifacts.
- **Feasibility of BraTS C:** 300 patients, 89,700 ordered impostor pairs and 100 × 24 genuine attempts. This is the same size as NIH and ISIC.

## 2. Fixed elements (identical to revision 9 unless stated)

- **Models:** the three trained AIF-Net seeds (0, 1, 2), with no retraining. The weights are identified by their SHA-256 hashes.
- **Features:**
  - 5-view TTA; no rotated views.
  - Reliable-bit selection q ∈ {0.6, 0.7, 0.8, 0.9, 1.0}, a calibrated parameter of each code rule. NC rule: q = 0.7.
- **Operations:** the 24 standard and clinical operations, seeded per operation with `RandomState(11)`.
- **Genuine attempts:**
  - C: the first 100 patients × 24 operations;
  - U: the first 200 patients × 24 operations;
  - T: 100 × 24, as in revision 9.
- **Impostor attempts:** all ordered pairs (i, j), i ≠ j, of clean images within a set and modality:
  - C: 89,700 per modality;
  - U: 999,000 for NIH and ISIC, 708,122 for BraTS;
  - T: 9900.
- **Rule families and grids:** as in revision 9.
  - NC threshold grid: 0.900–1.000 in steps of 0.001.
  - Hachimoji: GF(64) dinucleotide RS, and GF(8) base RS in blocks of 7.
  - Four-letter GF(4) RS in blocks of 3.
  - Binary BCH: length 63 and two shortened (255, k) codes.
  - Each with erasures s and keyed or index bit order.
- **Message:** content bits plus a T-bit tag HMAC(k_h, "hrs-tag" ‖ content), with content ≥ 64 bits and T ∈ {32, 64, 128}. The comparison codes carry the same message, so payloads are matched.

## 3. Selection on C only (two pre-registered operating points)

For each seed, modality, family and tag length, select the configuration with the highest genuine acceptance on C, subject to the false-acceptance count on C:
- **OP-A:** FA_C ≤ ⌊10⁻⁴ × 89,700⌋ = 8. This is the revision-9 target.
- **OP-B:** FA_C = 0. This is a stricter point, chosen because a pair-level target on finite data is optimistic.

Tie-breaks: fewer FA, then larger message. The Hachimoji decoder is whichever of GF(64) and GF(8) gives higher acceptance on C.

Output: `frozen_v10.json`, holding every selected configuration, the full calibration table, and the NC thresholds.

## 4. Freezing procedure (before any U feature exists)

1. Put the code under a local git repository and commit. The commit hash goes into the protocol.
2. Write `PROTOCOL_v10.md` (this document, finalized) and `frozen_v10.json`. Record:
   - their SHA-256 hashes;
   - the hashes of the model weights, the manifests and the evaluation script;
   - a UTC timestamp.

   All of this goes into `freeze_record.json`. The record is also committed to your laptop and the project before U is touched, so it carries an independent timestamp.
3. `eval_final_v10.py` is the only script that reads U images or features. It refuses to run unless:
   - the hashes of `frozen_v10.json`, `PROTOCOL_v10.md` and the script itself match `freeze_record.json`;
   - the git tree is clean.

   It appends every access (time, hashes, command) to `access_log_U.txt`.
4. **Deviations:** any change after U results are seen is written to `DEVIATIONS.md`. Results obtained after a deviation are labelled exploratory. Confirmatory claims would then need a further cohort; NIH and ISIC have unused patients, BraTS does not.

## 5. Final evaluation on U (and T)

For each seed (3) × modality (3) × family (NC, Hachimoji, GF(4), BCH) × tag length (3; not for NC) × operating point (2):

- **Genuine acceptance:**
  - on U: 200 × 24 attempts;
  - on T: 100 × 24 attempts;
  - per operation and overall.
- **Impostor decisions:** the exact errors-and-erasures condition 2e + s ≤ n − K for all codes. For the proposed decoder, the real decoder (`hachimoji_rs.py`, reedsolo 1.7.0) is also run on:
  - every impostor pair that the condition accepts;
  - a random sample of 20,000 rejected pairs (seed 0), with the agreement reported.

  Running the real decoder on all 2.7 million pairs per run is computationally infeasible here (estimated over 50 h). This sample-and-verify design is fixed in advance.
- **Mechanism of every false acceptance:**
  - (a) the impostor decodes to the genuine message: a feature-level codeword collision, which tag length cannot prevent;
  - (b) a different message passes the tag: a tag collision;
  - (c) NC rule: all selected bits agree.
- **Decoding failures among rejected pairs** (real-decoder sample), split into RS failure and tag mismatch.
- **Code properties:** n, K, rate K/n, correction capacity n − K, erasures s, content bits, tag bits.
- **Timing:** decoding time per verification, with library names and versions (reedsolo 1.7.0; galois 0.4.11 for BCH, to be confirmed in the code). Measured on the CPU named in Section 7.

## 6. Statistics

1. **Pair level:** FA count and exact denominator N(N − 1). The pair-level rate is reported as descriptive only. A Clopper–Pearson interval under independence is shown, explicitly labelled as not valid for dependent pairs.
2. **Patient-cluster bootstrap (primary uncertainty):**
   - B = 2000 resamples, seed 0. Each draws N patients with replacement from the N patients of U.
   - The resampled impostor set is all ordered pairs (a, b) of resampled draws whose underlying patients differ. Duplicated patients therefore keep their dependence; a patient never pairs with a copy of itself.
   - FAR\* = FA\* / pairs\*.
   - Reported: 95% percentile interval and one-sided 95% upper bound.
   - Effective sample size: n_eff = p̂(1 − p̂) / Var_boot(FAR\*).
   - Paired bootstrap of acceptance differences between rules, as in revision 9.
3. **Zero false acceptances:** the bootstrap is degenerate (every resample gives 0), so no upper bound is claimed from it. Instead, the paper reports:
   - "0 of M pairs among N patients";
   - the patient-involvement rate (patients involved in at least one FA, out of N), with a Clopper–Pearson interval over patients;
   - the patient-disjoint check below.
4. **Patient-disjoint sanity check:** one fixed random perfect matching of U (seed 0) gives 500 independent pairs per modality (421 for BraTS). This is reported as a limited check only, since zero FAs bound the rate near 6 × 10⁻³.
5. **Claim rule (fixed in advance):** "FAR ≤ 10⁻⁴ (95%)" is stated only if FA > 0 and the cluster-bootstrap one-sided upper bound is ≤ 10⁻⁴. In every other case, the paper states the strongest supported statement, for example "k false acceptances in M pairs among N patients (bootstrap 95% CI …)", and calls 10⁻⁴ the calibration target.

## 7. Reproducibility record

- **Compute:** Intel Xeon @ 2.10 GHz, 2 vCPU, 8 GB RAM, Ubuntu 24.04.5 LTS, no GPU.
- **Software:** `pip freeze`, with reedsolo 1.7.0 and galois 0.4.11 among others.
- **Run records:** every command line, random seed and wall-clock runtime.
- **Artifacts:** feature caches and per-pair decisions, supplied to the repository as hashes plus scripts. Third-party images are not redistributed.

## 8. Paper

- **Self-contained main text.** JAIR's submission page says online appendices "are not included in the review process". So every result a claim rests on stays in the main text: per-seed values in a compact table, U results, and the bootstrap bounds.
- **Online appendix or repository:** only detailed tables (per operation, full calibration grid).
- **Length:** 15 pages plus the checklist, unchanged.
- **Text fixes:**
  - exact partition rule;
  - "calibration target" wording;
  - tag security in a bounded threat model;
  - libraries and code rates;
  - a readable seed table;
  - hardware;
  - checklist updates.

## 9. Compute estimate

- **Features:**
  - C: 900 clean images plus 300 × 24 attacked;
  - U: 2842 clean images plus 600 × 24 attacked, U also covering the 200 genuine patients;
  - × 3 seeds, about 3 h.
- **Calibration grid on C:** about 2 h.
- **U evaluation:** about 3–5 h.
- **Total:** roughly 8–10 h, spread over several 2-hour runs.
