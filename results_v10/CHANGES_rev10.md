# Revision 10 manuscript changes (working version `jair_v10/`; revision 9 in `jair_v9/` is unchanged)

Every number added below was recomputed by script from `runs/v10/verified/final_results_long.json` (built from the verified `U_results_seed{0,1,2}.json`, SHA-256 afeae5c5…, 6e3c6573…, 5ff516c4…).

## Substantive (scientific) changes
| Location | Change |
|---|---|
| Abstract, Results | Removed "81–89% … at most 4 FA in 9900 pairs" and "more than DNA4 on every collection and BCH on two of three". Added: frozen-protocol evaluation on the untouched cohort; Hachimoji OP-A genuine acceptance 80.7–93.7%; observed FAR 1.1–2.1×10⁻⁴ (BraTS, NIH) and 2.1–6.3×10⁻⁵ (ISIC); all feature-level, no tag collisions; FAR ≤ 10⁻⁴ at 95% not demonstrated; families compared only descriptively; strict NC > 3×10⁻⁴ on radiographs (kept). |
| Abstract, Conclusions | "exact verification decision" → exact payload, but false acceptances on new patients above the calibration target. |
| Contribution 4 | "outperforms four-letter DNA codes" → reports genuine and false acceptances beside DNA4 and BCH under a pre-registered evaluation. |
| Section 3.5 | T-bit tag, T ∈ {32, 64, 128}; bounded-distance rule stated (decoded word kept only within 2e + s ≤ n − K). |
| Section 4.2 | Revision 9 calibration paragraph replaced: frozen protocol, calibration cohort (300 per modality), untouched cohort (BraTS 842 and ISIC 1557 censuses; NIH 1000 i.i.d. draws, P = 30,305), OP-A/OP-B, 1236-configuration grid, ≥ 64 content bits, key-stream order, bounded-distance rule, reference-implementation conformance; statistics (exact census FAR; Hoeffding-type bound p_U, NIH finite-population correction; superpopulation assumption for BraTS/ISIC; claim rule). |
| Section 4.3 | Comparison codes "validation-selected" → selected by the same calibration procedure. |
| Section 5.5 + Table 2 (old Table 3) | Rewritten. New table: per seed, genuine acceptance (%) and FA for every family and OP (T = 32), M per collection, Hachimoji OP-A FA/M and p_U per seed (†: assumption-dependent). Text: budget transfer failure (31/78 OP-A rules above budget rate; 55/66 OP-B rules with FA > 0); bounds 3.84–7.78×10⁻³; mechanism (a) only, FA_b = 0, E_tag ≤ 27,382, ideal-tag expectation ≤ 6.4×10⁻⁶; T = 64/128 changes (3 and 14 of 54 rules); untested explanation labelled as such; descriptive family comparison; NC results; decoding times. Old claims removed: "≤ 4 of 9900", "up to 4×10⁻⁴", paired CIs for family differences, "BCH 0.36 s", "RS about 1 ms". |
| Limitations | "up to four times its target" replaced; added budget-transfer failure, feature-level collisions, sample size needed, scope (releases, image choice, preprocessing, models/rules, random non-adversarial impostors), census vs i.i.d. assumption. |
| Conclusion | Decision claim rewritten (exact payload; target exceeded on BraTS/NIH; FAR ≤ 10⁻⁴ at 95% not demonstrated; NC exceeds on radiographs); future work updated. |
| Checklist explanation | Added: freeze record (git 9d2dbf46, record 801fd0cf…), single evaluation, no post-hoc changes; defects F1–F3; ISIC truncated image and authorized rerun; restarts and reuse of U feature files (bit-identical regeneration NOT independently verified); list of descriptive/non-pre-specified quantities. |
| References | Added Hoeffding (1963), JASA 58(301), 13–30, doi:10.1080/01621459.1963.10500830 (verified on Crossref 2026-10-29); cited without section location. |

## Editorial changes made to save space (no change of meaning)
- Algorithm 2 (verification) removed; its steps are described in the Section 3.1 text with references to Algorithm 1 lines.
- Figure 1 width 0.54 → 0.48 textwidth; Table 1 font footnotesize → scriptsize.
- Section 4.4: duplicated generative-AI sentence replaced by a pointer to the Use of Generative AI section.
- BCH acronym now defined at first body use (Section 4.2).

## Unchanged revision 9 findings (evidence unaffected by revision 10)
NC robustness and ablation (Table 1), same-data baselines, 2026 collections, TCIA key binding, Proposition 1, volumetric results.
