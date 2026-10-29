# Revision 10: final results on the untouched cohort U (seeds 0, 1, 2)

Status: **all three seeds complete and verified** (`verify_seed_v10.py`: FAILURES none, for every seed). Nothing in the manuscript has been changed yet.
Sources: `runs/v10/verified/U_results_seed{0,1,2}.json` (SHA-256 afeae5c5…, 6e3c6573…, 5ff516c4…), long-format extract `verified/final_results_long.json`. Every number below comes from these files.

## 0. Headline findings (actual results)

1. **The pre-specified claim "FAR ≤ 10⁻⁴ (95%)" is not supported anywhere.** The claim rule (Section 6.5) requires M1 ≤ 10⁻⁴. The smallest M1 obtained is 3.84 × 10⁻³ (ISIC, FA = 0). 10⁻⁴ stays the *calibration target* only.
2. **The calibrated false-acceptance control did not carry over from C to U.**
   - OP-A (budget 8 / 89,700 ≈ 8.9 × 10⁻⁵ on C): U rate above that budget in **31 of 78** selected cells, and above 10⁻⁴ in 24.
   - OP-B (0 FA on C): **55 of 66** cells have FA > 0 on U. Only 11 cells reach 0: 8 DNA4 and 3 Hachimoji (NIH seed 0, HRS8, all three tag lengths), with genuine acceptance of 0.31–0.82.
3. **Every false acceptance is mechanism (a)**: the impostor image's features fall inside the decoding radius of the registrant's code word. **FA_b (tag collisions) = 0 in all 144 cells.** The idealized tag-collision expectation E_tag × 2⁻ᵀ is at most 6.4 × 10⁻⁶. So longer tags cannot fix the observed false acceptances; the error is at the feature level.
4. **Strict NC rule.** On C no threshold met the budget for NIH or ISIC; that holds again on U. The uncalibrated τ = 1.00 rule gives FA/M of 3.2–6.5 × 10⁻⁴ on NIH and 0.48–1.4 × 10⁻⁴ on ISIC. On BraTS it gives 0–5 FA (≤ 7.1 × 10⁻⁶).
5. **Family comparison: descriptive only, and confounded.**
   - At OP-A, the Hachimoji decoder has the highest mean genuine acceptance on NIH and ISIC. On BraTS it is tied with BCH (0.933 vs 0.932).
   - It also has the most false acceptances in most cells. For example, NIH: Hachimoji 135–207 vs DNA4 20–91.
   - Because the realized FARs differ, the genuine-acceptance differences are **not comparisons at equal FAR**.
   - The frozen protocol pre-specifies no significance test for family comparisons on U. Any such test would be exploratory.

## 1. Final results table (T = 32 bits. Correction 2026-10-29: T = 64 selected the same configuration as T = 32 in all but 3 cells — BraTS seed 1 DNA4 OP-A and Hachimoji OP-B, BraTS seed 2 DNA4 OP-A. T = 64 and T = 128 values are in `final_results_long.json`.)

How to read each statistic, per the frozen protocol, Section 1.3:
- **BraTS, ISIC (census):** FA/M is the **exact** FAR among all ordered pairs of the remaining eligible patients. M1 applies only under the *assumed* superpopulation (i.i.d.) model, so it is **not** a 95% guarantee.
- **NIH (i.i.d. draws with replacement, P = 30,305):** the M1 column already holds the finite-population form M1/(1 − 1/P). It is a **rigorous** one-sided 95% upper bound on the impostor FAR of that finite population. FA excludes the 42 ordered same-patient draw pairs; those are reported separately.

Other columns:
- Genuine acceptance: 200 patients × 24 operations = 4800 attempts per cell.
- "none": no configuration met the budget on C (no fallback).
- M2 (cluster bootstrap, reported only if FA ≥ 3) is in the JSON; M3 and the disjoint check are descriptive.

### BRATS (N = 842, M = 708,122 ordered pairs)

| Rule | OP | Genuine acceptance on U, mean (range) | FA s0/s1/s2 | FA/M range | M1 95% upper |
|---|---|---|---|---|---|
| Hachimoji | OP-A | 0.933 (0.928–0.937) | 90/80/91 | 1.1e-04–1.3e-04 | 7.68e-03–7.74e-03 |
| Hachimoji | OP-B | 0.907 (0.890–0.927) | 25/11/44 | 1.6e-05–6.2e-05 | 7.20e-03–7.45e-03 |
| DNA4 | OP-A | 0.898 (0.865–0.914) | 11/46/57 | 1.6e-05–8.0e-05 | 7.20e-03–7.53e-03 |
| DNA4 | OP-B | 0.842 (0.824–0.865) | 11/0/6 | 0.0e+00–1.6e-05 | 7.09e-03–7.20e-03 |
| BCH | OP-A | 0.932 (0.921–0.940) | 65/64/97 | 9.0e-05–1.4e-04 | 7.58e-03–7.78e-03 |
| BCH | OP-B | 0.893 (0.869–0.905) | 30/11/15 | 1.6e-05–4.2e-05 | 7.20e-03–7.35e-03 |
| NC (τ=1.00) | uncalibrated | 0.830 (0.823–0.836) | 1/0/5 | 0.0e+00–7.1e-06 | 7.09e-03–7.15e-03 |
| NC (τ=cal.) | OP-A | 0.920 (0.903–0.929) | 47/64/93 | 6.6e-05–1.3e-04 | 7.47e-03–7.75e-03 |
| NC (τ=cal.) | OP-B | 0.866 (0.858–0.879) | 9/3/20 | 4.2e-06–2.8e-05 | 7.13e-03–7.27e-03 |

### NIH (N = 1000, M = 999,000 ordered pairs)

| Rule | OP | Genuine acceptance on U, mean (range) | FA s0/s1/s2 | FA/M range | M1 95% upper |
|---|---|---|---|---|---|
| Hachimoji | OP-A | 0.844 (0.832–0.852) | 135/207/184 | 1.4e-04–2.1e-04 | 6.63e-03–6.90e-03 |
| Hachimoji | OP-B | 0.792 (0.753–0.814) | 0/115/32 | 0.0e+00–1.2e-04 | 5.97e-03–6.55e-03 |
| DNA4 | OP-A | 0.814 (0.801–0.823) | 91/47/20 | 2.0e-05–9.1e-05 | 6.11e-03–6.45e-03 |
| DNA4 | OP-B | 0.770 (0.743–0.817) | 1/22/20 | 1.0e-06–2.2e-05 | 5.98e-03–6.12e-03 |
| BCH | OP-A | 0.824 (0.818–0.829) [2 seeds] | 54/none/138 | 5.4e-05–1.4e-04 | 6.28e-03–6.64e-03 |
| BCH | OP-B | — (none, 3/3 seeds) | — | — | — |
| NC (τ=1.00) | uncalibrated | 0.877 (0.861–0.887) | 321/645/441 | 3.2e-04–6.5e-04 | 7.29e-03–8.25e-03 |
| NC (τ=cal.) | OP-A | — (no τ met the budget on C) | — | — | — |
| NC (τ=cal.) | OP-B | — (no τ met the budget on C) | — | — | — |

### ISIC (N = 1557, M = 2,422,692 ordered pairs)

| Rule | OP | Genuine acceptance on U, mean (range) | FA s0/s1/s2 | FA/M range | M1 95% upper |
|---|---|---|---|---|---|
| Hachimoji | OP-A | 0.818 (0.807–0.832) | 51/123/153 | 2.1e-05–6.3e-05 | 3.97e-03–4.17e-03 |
| Hachimoji | OP-B | 0.759 (0.748–0.777) | 17/23/53 | 7.0e-06–2.2e-05 | 3.89e-03–3.98e-03 |
| DNA4 | OP-A | 0.809 (0.795–0.828) | 19/77/159 | 7.8e-06–6.6e-05 | 3.90e-03–4.18e-03 |
| DNA4 | OP-B | 0.675 (0.570–0.735) | 0/4/4 | 0.0e+00–1.7e-06 | 3.84e-03–3.86e-03 |
| BCH | OP-A | 0.785 (0.780–0.788) | 58/132/137 | 2.4e-05–5.7e-05 | 3.99e-03–4.14e-03 |
| BCH | OP-B | 0.766 (0.766–0.766) [1 seed] | none/none/116 | 4.8e-05–4.8e-05 | 4.10e-03–4.10e-03 |
| NC (τ=1.00) | uncalibrated | 0.846 (0.832–0.858) | 117/334/257 | 4.8e-05–1.4e-04 | 4.11e-03–4.46e-03 |
| NC (τ=cal.) | OP-A | — (no τ met the budget on C) | — | — | — |
| NC (τ=cal.) | OP-B | — (no τ met the budget on C) | — | — | — |


## 2. Deviations and incidents (all logged)

| # | What | Effect |
|---|---|---|
| D1 | Truncated U image `ISIC_5683393.png`, caused by a preparation timeout. Seed 1 stopped with exit 1. The image was regenerated by the unchanged `prep_v10.py` and is pixel-identical to an independent conversion. Seed 1 was rerun from the start with your authorization (`DEVIATIONS.md`). | None on the method. The input is the one the frozen procedure specifies. No ISIC result existed before the fix. |
| D2 | Container restarts interrupted seed 0 and seed 2 twice each (`logs/failed/`). Each was rerun with the same guarded command; there are 8 entries in `access_log_U.txt`. | Decisions and statistics were recomputed in every rerun. **Disclosure:** the frozen `eval_final_v10.py` step 1 reuses U feature files already written (`U_features_seed*_*.npz`), so later reruns reused features extracted by earlier attempts. This is the frozen script's documented behaviour, not added resume logic. Pair counts and N matched in every verification. **I have not independently re-extracted the features to confirm they are bit-identical.** |
| D3 | Pre-freeze defects (F1–F3; protocol Sections 0 and 12) | Fixed before the freeze and part of the approved protocol. They are not post-U deviations. |

- **No change** was made to code, configurations or frozen artifacts after any U result was seen.
- **Exploratory, not pre-specified:**
  - cross-seed means and ranges;
  - counts of cells exceeding the budget;
  - family comparisons;
  - the interpretation in Section 3.

## 3. Interpretation (reasoned, not tested)

- **Selection at the edge of a small sample.** "0 FA in 89,700 C pairs among 300 patients" is compatible with rates far above 10⁻⁴ (the M1 bound at N = 300 is about 2 × 10⁻²). The protocol then picks the configuration with the *highest* genuine acceptance among those meeting the budget. That choice favours configurations near the false-acceptance boundary, so an optimistic bias on U is expected. Patient heterogeneity between C and U may add to it.
- **This is not established here.** Testing it would need a further cohort. NIH and ISIC have unused patients; BraTS does not.
- Revision 9's realized-FAR statements were calibrated on 9900 test pairs. They are superseded by these U numbers.

## 4. Manuscript items to update (after you have checked these numbers)

| Location | Revision 9 statement | Status vs U |
|---|---|---|
| Abstract | "verifies 81–89% … with at most 4 false acceptances in 9900 pairs" | **Contradicted.** U, OP-A: genuine acceptance BraTS 0.933, NIH 0.844, ISIC 0.818, with 51–207 FA per seed out of 0.7–2.4 M pairs. |
| Abstract | "more than four-letter DNA codes on every collection and BCH on two of three" | Genuine acceptance is higher than DNA4 on all three at OP-A. Against BCH: BraTS tied (0.933 vs 0.932), ISIC higher, NIH higher (BCH met the budget for 2 seeds at OP-A and none at OP-B). FARs are **not matched**, so the sentence must be rewritten. |
| Abstract | "strict NC threshold cannot reach 10⁻⁴ on radiographs" | **Consistent.** On U: 3.2–6.5 × 10⁻⁴ (NIH). |
| Contribution list / Introduction | any "controlled FAR" wording | Must say the target was not met on U. |
| Section 4.2 (metrics, calibration) | describes calibration on 200 NIH/ISIC patients and BraTS validation volumes | Replace with the revision 10 protocol: C of 300 patients per modality, OP-A and OP-B, T ∈ {32, 64, 128}, bounded-distance rule, cohort U, M1/M2, and the claim rule. |
| Section 5.5 and Table 3 | revision 9 decision table (9900 test pairs) | **Superseded (F1/F2).** Replace with the Section 1 table (compact: per modality, OP-A/OP-B, Hachimoji/DNA4/BCH/NC; per-seed FA; FA/M; M1). Revision 9 test-set genuine acceptance can stay as continuity: 0.81–0.94 BraTS, 0.43–0.86 NIH, 0.29–0.85 ISIC across cells. |
| Section 5.5 text | "accepts at most 4 of 9900 … up to 4 × 10⁻⁴" | Replace with census/finite-population statements. Add mechanism (a) only, FA_b = 0, and E_tag. |
| Section 5.5 | "RS decoding takes about 1 ms" | U median per genuine attempt: 0.16–4.5 ms over all cells (BCH included). Report per family. |
| Limitations | "realized FAR reached up to four times its target" | Replace: on U, OP-A exceeded its budget in 31 of 78 cells (up to 2.1 × 10⁻⁴, NIH), and OP-B was non-zero in 55 of 66. Add: scope (random non-adversarial impostors, these releases and preprocessing only), census vs superpopulation, and the deviations D1–D2. |
| Conclusion | positive decision claim | Rewrite: exact recovery and error tolerance are supported. FAR control at 10⁻⁴ is **not**. False acceptances are feature-level collisions. |
| Reproducibility checklist and explanation | revision 9 history | Add the pre-registration (protocol hash e34f5059…, freeze record 801fd0cf…, git 9d2dbf46), the single U evaluation, the restarts, D1, and F1–F3. |
| Data availability | — | List the per-pair outputs, the manifest and the scripts. Third-party images are not redistributed; the 20 COVID images stay private. |

## 5. Consistency check: outputs ↔ manuscript (revision 9 text, `hachimoji_watermark_JAIR.tex`)

- **Consistent:** NC rule infeasible on NIH under the budget; no tag collisions in revision 9 (FA_b = 0 here too).
- **Inconsistent (must change):**
  - abstract FA count and 81–89% range;
  - Table 3 and its caption ($3 \times 9900$ pairs);
  - Section 5.5 FA statements;
  - "four times its target" limitation;
  - BCH comparison wording;
  - the calibration description in Section 4.2.
- **Not checkable from the outputs:** NC robustness (0.993–0.995), baselines, volume and TCIA results. These come from revision 9 runs, which are unaffected by revision 10.
- **Internal consistency checked by script:**
  - status counts sum to M;
  - FA + same-patient accepted = accepted draw pairs;
  - FA_a + FA_b = FA;
  - p̂ = FA/M;
  - 4800 genuine attempts per cell;
  - 9900 revision 9 test pairs per cell;
  - 144 frozen cells evaluated (and 18 "none").

## 6. Submission-readiness checklist

| Item | State |
|---|---|
| Methods: revision 10 protocol described in the paper | **To do** |
| Results: U table replaces Table 3; per-seed values in the main text (appendices are not reviewed by JAIR) | **To do** |
| Claim rule respected (no "FAR ≤ 10⁻⁴ at 95%") | Ready to apply |
| Limitations: transfer failure, scope, threat model, census assumption | **To do** |
| Deviations D1–D2 disclosed | **To do** (text drafted above) |
| F1/F2 disclosure (revision 9 Table 3 superseded) | **To do** |
| Hoeffding (1963) cited without a section location (Crossref-verified metadata) | Ready |
| Reproducibility artifacts: freeze record, frozen list, manifest, verified JSONs, SHA256SUMS | Present here; **not yet copied to the laptop or the repository** |
| GitHub push | **Not done; needs your explicit OK.** Exclude third-party weights and the 20 COVID images. |
| Page limit (15 + checklist) after edits | To check after editing |
| Supervisor style rules (no we/our/us, Harvard, acronyms) | To check after editing |
