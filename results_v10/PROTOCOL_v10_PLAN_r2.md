# Revision 10: pre-registered evaluation protocol, version 2 (for review)

Status: **draft for review. Not frozen.**
- No feature or score of the calibration set C or the untouched cohort U has been computed.
- The benchmarks in Section 8 used **only revision-9 test data** (already seen) and synthetic data.

Changes from version 1 are marked **[v2]**.

## 0. [v2] Findings from the benchmark that change the method

| Finding | Evidence | Consequence |
|---|---|---|
| **F1.** In revision 9, rule selection and the comparison codes used a fixed pseudo-random bit order (`RandomState(0)`). The deployed Hachimoji decoder uses the key-stream order. | Code inspection: `eval_calibrated.py` vs `hachimoji_rs.py`. | Revision-9 Table 3 compares numbers computed under different bit orders. It must be superseded and the issue disclosed. |
| **F2.** `reedsolo` returns code words *beyond* the bounded-distance radius 2e + s ≤ n − K. Fractions of random words "decoded": 12% for GF(64) (63, 28, s = 32); about 65% for GF(8) blocks of 7. | `bench` script on 3000 random words per code. | The revision-9 decoder sometimes accepted attempts beyond the radius, so its decisions are not characterized by the condition. |
| **Fix.** A **bounded-distance decision** (Section 2.3) and a condition computed from the decoder's own symbols. | **0 mismatches** between condition and real decoder over all attempts on the revision-9 test sets (Section 8). | The condition is now exact for mechanism (a). Mechanism (b) is evaluated by full decoding of every pair (Section 5). |

## 1. Partitions (patient level, machine-checkable): unchanged from version 1

| Set | BraTS | NIH | ISIC | Use |
|---|---|---|---|---|
| Training/validation pool | 90 | 100 | 99 | training only |
| Test T (seen in revision 9) | 100 | 100 | 100 | continuity only, not confirmatory |
| Calibration C | 300 new | 300 | 300 | rule selection only |
| Untouched U | 842 (all remaining) | 1000 | 1000 | final evaluation only |

- **Manifest and checker:** `manifest_v10.json` lists every set, and `check_manifest.py` asserts patient-level disjointness and that every file exists.
- **Patient identifiers:**
  - BraTS: the 5-digit case number, across both the training and validation releases;
  - NIH: the 8-digit patient number;
  - ISIC: `patient_id` from the metadata file.
- **Sizes on C:**
  - Genuine attempts: 100 patients × 24 operations in every modality.
  - Impostor pairs: 300 × 299 = 89,700 per modality.
- **Sizes on U:**
  - Genuine attempts: 200 patients × 24 operations.
  - Impostor pairs: N(N − 1) = 999,000 for NIH and ISIC, 708,122 for BraTS.
- **BraTS comparability:**
  - C is now the same size as NIH and ISIC.
  - U is smaller for BraTS (842) because the release has only 1332 patients.
  - BraTS uses the middle slice, so every impostor pair is at the same relative depth.

## 2. Fixed elements

### 2.1 Unchanged from version 1
- **Models:** the three trained AIF-Net seeds (0, 1, 2), with no retraining. Weights are identified by SHA-256.
- **Features:** 5-view TTA.
- **Operations:** the 24 standard and clinical operations, seeded per operation with `RandomState(11)`.
- **NC rule:** q = 0.7, threshold grid 0.900–1.000 in steps of 0.001.

### 2.2 Families and grids **[v2: GF(64) grid extended]**
- **Selection levels:** q ∈ {0.6, 0.7, 0.8, 0.9, 1.0} for every family.
- **Bit order:** keyed or index order, the same option for every family. **[v2]** The keyed order is the HMAC(k_h, "hrs-stream") key stream for *all* families.

| Family | Code | Message per word | Erasures s |
|---|---|---|---|
| Hachimoji GF(64) | one RS word, n = min(63, ⌊L/6⌋), K ∈ {16, 20, …, 44} (**[v2]** was 16–28) | 6K bits | 0, 4, 8, … ≤ n − K |
| Hachimoji GF(8) | blocks of n = 7, K ∈ {1, …, 4} | 3K per block | 0 … 6 − K |
| Four-letter GF(4) | blocks of n = 3, K ∈ {1, 2} | 2K per block | 0 … 2 − K |
| BCH-63 | (63, k) blocks, k ∈ {36, 30, 24, 18, 16} | k per block | {0, 2, 4, 6} ≤ 2t |
| BCH-255 | 1 or 2 shortened (255, k) words, k ∈ {131, 115, 99, 87, 71} | k − (255 − L_b) | {0, 4, 8, 12, 16} |

- **Message:** content ‖ tag, with tag = first T bits of HMAC-SHA256(k_h, "hrs-tag" ‖ content).
  - T ∈ {32, 64, 128}.
  - content = message − T, and must be at least 64 bits.

### 2.3 **[v2]** Bounded-distance decision (all code families)
1. **Errata positions:** in every block, the s least reliable received symbols (bits, for BCH) are erased. Reliability is the smallest |logit| of the symbol's bits; ties are broken by position.
2. **Decoding:** the block is decoded by:
   - RS: `creedsolo`, the Cython build of reedsolo 1.7.0;
   - BCH: `FastBCH`, a vectorized Berlekamp–Massey decoder for galois 0.4.11's BCH codes. For erasures it uses the fill-with-0 / fill-with-1 rule.
3. **Bound:** an output code word c′ is kept **only if** 2e′ + s ≤ n − K (RS) or 2e′ + s ≤ 2t (BCH), where e′ is the number of non-erased positions in which c′ differs from the received word.
4. **Uniqueness and completeness:**
   - At most one code word satisfies the bound, because 2 × radius < minimum distance.
   - If one exists, the decoder finds it: guaranteed for errors-and-erasures RS decoding, and for binary BCH with the two-fill rule.
5. **Acceptance:** accept iff every block yields such a c′ **and** the tag of the decoded content matches.

**Exact characterization.** An impostor pair (registrant i, probe j) is accepted iff:
- **(a)** the bounded-distance condition holds against i's registered symbols, so the decoder returns i's own message ("feature-level collision"); or
- **(b)** a *different* code word lies within the radius in every block, and its HMAC tag matches ("tag collision").

Revision 9's "≥ 96 bits" rule and its test results are superseded by this definition.

### 2.4 NC rule, exact and vectorized **[v2]**
- **Parity zero-watermark:** par(f) = where(f[p], par1, par0), from Proposition 3.1, followed by the fixed inverse-Arnold permutation.
- **Check:** this is identical to the revision-9 `fast_nc` on 10,050 of 10,050 attempts (Section 8).

## 3. **[v2]** Feasibility (structural, before any data): number of (q, K) choices with content ≥ 64 bits

| T | GF(64) | GF(8) | GF(4) | BCH-63 | BCH-255 |
|---|---|---|---|---|---|
| 32 | 40 (rate 0.25–0.86) | 14 (0.29–0.57) | 10 (0.33–0.67) | 21 (0.25–0.57) | 13 (0.24–0.51) |
| 64 | 30 (0.38–0.86) | 10 (0.29–0.57) | 8 (0.33–0.67) | 14 (0.25–0.57) | 10 (0.28–0.51) |
| 128 | 20 (0.51–0.86); **infeasible under the revision-9 grid** | 5 (0.43–0.57) | 5 (0.67 only) | 6 (0.38–0.57) | 4 (0.39–0.51) |

Rate = K/n. Each (q, K) choice is then crossed with the erasure and bit-order options.

**Matched payloads:**
- **Primary comparison:** every family under the same constraint (content ≥ 64, tag = T).
- **Secondary comparison:** every family restricted to its smallest feasible message ≥ 64 + T. This is the closest possible match, because message granularity differs: 6K bits for GF(64), 3K × blocks, 2K × blocks, and k per BCH block.
- **Reported for every selected configuration:** message, content and tag bits, n, K, rate, correction capacity n − K (2t for BCH), s, bit order and q.

## 4. Selection on C only

### 4.1 Operating points
For each seed × modality × family × T, choose the configuration with the highest genuine acceptance on C, subject to:
- **OP-A:** FA_C ≤ ⌊10⁻⁴ × 89,700⌋ = 8;
- **OP-B:** FA_C = 0.

Tie-breaks, in order: fewer FA_C, larger content, smaller q, index order before keyed order, then grid order.

The Hachimoji decoder is whichever of GF(64) and GF(8) has higher acceptance on C.

### 4.2 Exactness on C **[v2]**
1. Candidates are ranked by the exact condition (mechanism a).
2. The top candidate is then fully decoded on all 89,700 pairs, to add mechanism (b).
3. If the full FA_C breaks the budget, the next candidate is taken and step 2 repeats.

### 4.3 No feasible configuration **[v2]**
- That cell is reported as "none". There is no fallback, no relaxation of the budget, and no configuration is forced.
- **NC rule:** if no τ meets the budget, only the uncalibrated τ = 1.00 is evaluated, labelled "exceeds budget on C".
- **Aggregates:** a mean over seeds is reported only when all 3 seeds are feasible. Otherwise the per-seed values are given, with the number of feasible seeds.
- **Paired comparisons** are computed only for seed/modality cells where both rules are feasible, and the number of such cells is stated.

## 5. Final evaluation on U: exact FA calculation (no sampling) **[v2]**

For each frozen configuration (seed, modality, family, T, OP; identical configurations are evaluated once):

1. **Registration:** for every patient i of U, compute the record with random content. Content is drawn by `RandomState(1)` in manifest order.
2. **Impostor decisions:** for every ordered pair (i, j), i ≠ j, verify the clean image of j against record i with the full decoder of Section 2.3. Every pair gets exactly one status:
   - `accept_genuine`: mechanism (a);
   - `accept_other`: mechanism (b);
   - `reject_fail`: some block has no code word within the radius;
   - `reject_tag`: a different code word lies within the radius, but the tag fails.

   Per-pair results are stored as the sparse lists of non-`reject_fail` pairs plus the counts. Counts satisfy Σ status = N(N − 1), which is checked.
3. **Counts:** FA = #accept_genuine + #accept_other, out of M = N(N − 1). FA_a and FA_b are reported separately.
4. **Tag exposure:** E_tag = #reject_tag + #accept_other, the number of pairs that reached a tag comparison. Its expected number of tag collisions, E_tag × 2⁻ᵀ, is reported. This shows directly whether tag length could have mattered.
5. **NC rule:** exact vectorized NC for all pairs, at the calibrated τ (if any) and at τ = 1.00.
6. **Genuine acceptance:** 200 × 24 attempts on U and 100 × 24 on T, with the status of every attempt. Decoding failures (`reject_fail`) and tag rejections (`reject_tag`) are reported separately.
7. **Timing:** per verification, median over the U genuine attempts.

## 6. Statistics **[v2]**

**Estimand.** p = P(accept | registrant A, probe B), for two different patients drawn at random from the population of the collection. The estimator p̂ = FA/M is an unbiased U-statistic of order 2.

| | Method | Assumptions | Coverage |
|---|---|---|---|
| **M1 (primary, rigorous)** | Hoeffding (1963, §5) bound for U-statistics. The symmetrized kernel g_ij = (h_ij + h_ji)/2 ∈ [0, 1] makes p̂ an average of means of n′ = ⌊N/2⌋ i.i.d. kernels, so exp(−n′·KL(p̂‖p)) bounds P(p̂ ≤ observed \| p). One-sided 95% upper bound p_U solves KL(p̂‖p_U) = ln(20)/n′. | Patients i.i.d. from the population. **No assumption about dependence between pairs that share a patient.** | ≥ 95% by construction. Conservative: with FA = 0 the bound is 1 − 0.05^(1/n′) = **6.0 × 10⁻³ (N = 1000)** and 7.1 × 10⁻³ (N = 842). |
| **M2 (secondary, approximate)** | Patient-cluster bootstrap, B = 2000, seed 0. Weights w ~ Multinomial(N, 1/N); FA\* = Σ over accepted pairs of w_i·w_j; M\* = (Σw)² − Σw². This keeps every pair of copies of *different* patients, so dependence is retained and no patient is paired with its own copy. Reported: percentile 95% CI and one-sided 95% upper bound. | Large-sample validity; asymptotically correct for U-statistics. | **Degenerate when FA = 0** (reported as "not estimable"). Below nominal under strong patient heterogeneity (simulation, Section 8). |
| **M3** | Delete-one-patient jackknife variance, and n_eff = p̂(1 − p̂)/Var_jack. | As M2. | Undefined when FA = 0. |
| Supplementary | One fixed random perfect matching (seed 0): 500 independent pairs (BraTS 421), with exact Clopper–Pearson. | Independent pairs. | Valid but weak; M1 uses all pairs and supersedes it. |

- **[v2] M2 reporting rule:** M2 is reported only when FA ≥ 3, because its coverage collapses with fewer events (Section 8.3). It is always shown next to M1.
- **Reference:** Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *JASA* 58(301), 13–30, doi:10.1080/01621459.1963.10500830.
  - Metadata checked against Crossref.
  - That its Section 5 contains the U-statistic result as stated is **unverified — requires confirmation** against the full text before the paper cites it. The derivation is standard and is reproduced in `stats_v10.py`.
- **Pair-level Clopper–Pearson:** not reported as an interval, because it assumes independent pairs.
- **Patient involvement:** the number of distinct patients involved in false acceptances is a descriptive count only, never a FAR interval.

**Claim rule (fixed before U):**
- "FAR ≤ 10⁻⁴ with 95% confidence" may be stated **only if the M1 bound is ≤ 10⁻⁴**.
- M1 cannot be ≤ 10⁻⁴ at these cohort sizes even with FA = 0. A one-sided 95% bound of 10⁻⁴ needs n′ ≈ 30,000, i.e. about 60,000 independent patients.
- **This protocol therefore pre-commits to not claiming 10⁻⁴.** The paper reports:
  - FA / M and p̂;
  - the M1 bound (rigorous);
  - the M2 interval (labelled approximate, with its simulated coverage);
  - the mechanism split;
  - 10⁻⁴ as the *calibration target only*.

## 7. Freezing and access control (unchanged from version 1, plus items)

- **Recorded in `freeze_record.json`, with a UTC timestamp:**
  - git commit;
  - SHA-256 of `PROTOCOL_v10.md`, `frozen_v10.json`, the manifests, the model weights and every script.
  - The record is also committed to your laptop and to the project.
- **Access guard:** `eval_final_v10.py` is the only reader of U, and checks the hashes plus a clean git tree. Every access is appended to `access_log_U.txt`.
- **Deviations:** logged in `DEVIATIONS.md`. Results after a deviation are labelled exploratory; confirmatory claims would need a new cohort. NIH and ISIC have unused patients; BraTS has none.
- **[v2] Included in the frozen record:** the benchmark outputs of Section 8 and the validation of `creedsolo` against `reedsolo` and of `FastBCH` against `galois`.

## 8. **[v2]** Benchmarks (revision-9 test data and synthetic data only; results in `runs/v10/bench/`)

### 8.1 Correctness checks

| Check | Result |
|---|---|
| `creedsolo` vs pure-Python `reedsolo` | identical outputs on 20,000 random RS decodings (5 codes, 0–31 errors, random erasures) and identical statuses on 49,200 test attempts |
| `FastBCH` vs `galois.BCH.decode` | identical decodability, code word and error count on 15,000 words (BCH (63,36), (63,16), (255,99) shortened to 170, (255,131), (255,71); 0 to t + 3 errors and random words) |
| Condition vs bounded-distance decoder | **0 mismatches in 369,000 attempts**: 72,000 genuine and 297,000 impostor, over 30 seed × set × family cells (all four families, 3 seeds, 3 sets) |
| Mechanism (b), tag collisions | 0 observed. 163 impostor pairs (and some genuine attempts) reached a tag comparison with a different code word, and all were rejected by the tag |
| Message invariance (linearity) | decodability and condition identical for two random messages and for T = 32/64/128: 0 mismatches in 184,500 attempts |
| Vectorized NC vs `fast_nc` | 10,050 of 10,050 identical |

### 8.2 Impact of F1/F2 on the revision-9 Hachimoji results (same configurations)

| Seed | Set | Revision 9 (unbounded) GAR / FA | Bounded-distance GAR / FA |
|---|---|---|---|
| 0 | BraTS / NIH / ISIC | 0.882 / 1 · 0.788 / 0 · 0.792 / 0 | 0.882 / 1 · 0.788 / 0 · 0.790 / 0 |
| 1 | BraTS / NIH / ISIC | 0.897 / 2 · 0.825 / 3 · 0.856 / 0 | 0.897 / 2 · 0.817 / 2 · 0.852 / 0 |
| 2 | BraTS / NIH / ISIC | 0.887 / 2 · 0.820 / 4 · 0.818 / 0 | 0.887 / 2 · 0.817 / 4 · 0.817 / 0 |

- **F2 (unbounded decoding):** lowers acceptance by at most 0.009 and FA by at most 1.
- **F1 (bit order):** affects the revision-9 *condition-based* numbers, i.e. the comparison codes and the selection. For example, in NIH seed 2 the condition gave FA = 1 where the actual decoder gave 4.

### 8.3 Statistics: coverage of the one-sided 95% bounds (300 replicates each)

Model: P(h_ij = 1) = p0·u_i·u_j, with u ~ Gamma(κ, 1/κ). κ = 0 means no patient effect; a smaller κ means stronger heterogeneity.

| N | p0 (expected FA) | M1 coverage | M2 coverage, κ = 0 / 0.2 / 0.05 |
|---|---|---|---|
| 1000 | 10⁻⁶ (≈ 1) | 1.00 | 0.63 / 0.64 / 0.54, **fails** |
| 1000 | 3 × 10⁻⁶ (≈ 3) | 1.00 | 0.96 / 0.95 / 0.93 |
| 1000 | 10⁻⁵ (≈ 10) | 1.00 | 0.99 / 0.98 / 0.96 |
| 1000 | 10⁻⁴ (≈ 100) | 1.00 | 1.00 / 0.98 / 0.93 |
| 1000 | 3 × 10⁻⁴ | 1.00 | 1.00 / 0.94 / 0.87 |
| 842 | 10⁻⁵ | 1.00 | 0.98 / 0.97 / 0.93 |
| 842 | 10⁻⁴ | 1.00 | 1.00 / 0.96 / 0.92 |

- **M1:** never failed.
- **M2:** is unreliable with very few events, and below nominal under strong heterogeneity.
- **Pre-registered consequence:** M2 is reported only when FA ≥ 3, always next to M1 and labelled approximate. With FA ≤ 2, only the counts and M1 are reported.
- **Limitation:** the simulation covers one family of patient effects (two-way multiplicative). Clustered near-duplicate patients could behave differently. M1 does not depend on this.
- **Cost:** the bootstrap with B = 2000 takes under 1 s per cell.

### 8.4 Runtime (measured, 2 vCPU Intel Xeon @ 2.10 GHz, Ubuntu 24.04.5, no GPU)

| Step | Measured cost | Planned work | Estimate |
|---|---|---|---|
| Impostor decision, full decoding | HRS64 0.27–0.62 ms/pair; HRS8 0.03–0.17; GF(4) 0.02–0.10; BCH 0.10–0.92 | ≤ 54 configurations (3 seeds × 3 families × 3 T × 2 OP), identical ones evaluated once, × 2.7 M pairs | ≤ 12 CPU-h, ~6 h wall |
| NC rule (vectorized) | 0.13 ms/pair | 3 seeds × 2.7 M pairs × ≤ 2 thresholds | ~0.6 h |
| Features | ~0.1 s per image-operation (rev 9) | C: 900 + 300 × 24; U: 2842 + 600 × 24; × 3 seeds | ~3 h wall |
| Calibration ranking on C (condition, vectorized) + full decoding of selected | rev-9 calibration timing | grid × 3 modalities × 89,700 pairs | ~2–3 h |
| **Total** | | | **~12–15 h wall**, in several 2-hour runs |

### 8.5 Scripts (to be committed before freezing)

- **Decision rule:** `experiments/v10_decision.py`, which contains `Coder`, `FastBCH` and `nc_batch_fn`.
- **Statistics:** `experiments/stats_v10.py`, with M1, M2, M3 and the coverage simulation.
- **Benchmarks:** `experiments/bench_v10.py`, `bench_v10_tags.py` and `stats_v10_sim2.py`.
- **Still to be written and reviewed before freezing:**
  - `check_manifest.py`;
  - `calib_v10.py` (features and selection on C);
  - `eval_final_v10.py` (access-guarded U evaluation; writes every status count and the sparse list of non-`reject_fail` pairs).
