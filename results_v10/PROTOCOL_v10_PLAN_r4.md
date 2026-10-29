# Revision 10: pre-registered evaluation protocol, version 4 (final, for approval)

Status: **draft for review. Not frozen.**
- No feature or score of the calibration set C or the untouched cohort U has been computed.
- No C or U image has been copied to the compute machine.
- All tests below used **only revision-9 test data** (already seen in revision 9) and synthetic data.

Changes from version 2 are marked **[v3]**. The four targeted changes requested on version 3 are marked **[v4]**, and a consolidated checklist is in Section 11. Version 2 is kept in the project as `claude/revision10_protocol_plan_v2.md`.

---

## 0. Findings that change the method

| | Finding | Status |
|---|---|---|
| F1 (v2) | Revision 9 selected rules and evaluated the comparison codes with a fixed pseudo-random bit order, while the Hachimoji decoder uses the key-stream order. | All families now use the decoder's key-stream order (Section 2). The revision-9 Table 3 is superseded, and this is disclosed in the paper. |
| F2 (v2) | `reedsolo` returns code words beyond the bounded-distance radius. | Bounded-distance rule enforced for every family (Section 2.3). |
| **F3 [v3]** | `creedsolo`, the compiled build of `reedsolo` used for speed, keeps one global set of GF log/antilog tables. It does not restore them when a codec of another field is used. Interleaving GF(4)/GF(8)/GF(64) codecs gave **330 wrong results in 400** decodings. | Found by the independent reference tests while developing the scripts. Fixed: the field tables are re-activated before every RS encode/decode (`activate_field`). A regression test is now part of every conformance run: **0 of 400** wrong. **No reported number is affected**, because every earlier benchmark built a fresh codec immediately before use. |

---

## 1. Partitions and the sampling model **[v3: random sampling, explicit scope]**

### 1.1 Selection rule **[v4: designs of U changed]**
`build_manifest_v10.py` builds the manifest from the release listings only; no image content is read.
- **BraTS:** random permutation (numpy `RandomState(2026)`) of the 1142 eligible patients (every patient not in the pool or the test set). C = first 300; U = the remaining **842 — a census of the remainder**.
- **NIH:**
  - C = the revision-9 calibration patients (200), plus the first 100 of a random permutation (`RandomState(2026)`) of the 30,405 eligible patients.
  - **U = 1000 draws *with replacement*** (`RandomState(2028).choice`) from the remaining **P = 30,305** eligible patients, kept in draw order. The draws are therefore **exactly i.i.d. from that finite population**. 979 distinct patients were drawn; the 21 repeats produce same-patient draw pairs, which are never impostor attempts (Section 6.2).
- **ISIC:**
  - C = the revision-9 calibration patients (200), plus the first 100 of the permutation.
  - **U = all 1557 remaining eligible patients — a census of the remainder.** This replaces version 3's sample of 1000 without replacement.
- **One image per patient:**
  - BraTS: first time point (training release preferred), middle axial T1c slice;
  - NIH: `<patient>_000.png`;
  - ISIC: the patient's first image in name order.

The images extracted on the laptop under earlier lists are discarded; C and U images are prepared from this manifest.

### 1.2 Manifest and machine check
**`check_manifest.py` PASS** on `manifest_v10.json` (SHA-256 `759339fe…5a9a2d`). It checks:
- pairwise patient disjointness of pool/test/C/U in every modality;
- no duplicate patients, except the designed repeats in NIH U;
- **[v4]** that ISIC U and BraTS U are exactly the census of all remaining eligible patients;
- **[v4]** that the NIH U design record (draws, distinct patients) matches the list;
- the expected sizes;
- agreement with `partitions.json` (pool, test) and with the revision-9 calibration lists;
- every file present in the release listing;
- the file ↔ patient identity (BraTS and NIH from the file name; ISIC from the metadata);
- that the BraTS test volumes lie in the test set.

**Negative test:** a manifest with one NIH patient moved from C into U, one ISIC file swapped and one ISIC U patient removed is rejected. All four resulting faults are reported: overlap, size, wrong file, and census violated.

| Set | BraTS | NIH | ISIC |
|---|---|---|---|
| pool | 90 | 100 | 99 |
| test (revision 9) | 100 | 100 | 100 |
| C | 300 | 300 | 300 |
| U | 842 (census) | 1000 i.i.d. draws (979 distinct) | 1557 (census) |

**Image identity.** Laptop-prepared images are pixel-identical to the compute-machine pipeline:
- BraTS slice reader vs `nibabel` + `to_u8`: 5 volumes, maximum difference 0;
- NIH/ISIC 512×512 PNG vs the container's own resize: 6 test images, identical.

### 1.3 **[v4]** What each FAR statement means, by design

For collection c, one trained model and the frozen rule, p_c is the probability that an image of one patient is accepted under the record of a different patient, both from the stated population. The population is the patients of c's public release after the stated image choice and preprocessing:
- BraTS 2023 adult glioma, T1c, middle slice;
- NIH ChestX-ray14, first frontal radiograph;
- ISIC 2020, first dermoscopic image.

| Collection | Design of U | What is rigorous | What rests on an assumption |
|---|---|---|---|
| **NIH** | i.i.d. draws with replacement from the finite population of P = 30,305 remaining eligible patients | **M1 / (1 − 1/P)** is a rigorous one-sided 95% upper bound on the impostor FAR **of that finite population** (Section 6.3, step 8). | Nothing beyond the population definition. |
| **ISIC** | census of all 1557 remaining eligible patients | **FA / M is the exact FAR** among all ordered pairs of these patients. There is no sampling error, so no interval is needed. | Any statement beyond these 1557 patients. M1 is reported **only** as a bound for a hypothetical superpopulation of which these patients would be an i.i.d. sample. That model is an **assumption**, labelled as such, and is **not** presented as a 95% guarantee. |
| **BraTS** | census of all 842 remaining eligible patients | **FA / M is the exact FAR** among these patients. | As ISIC. |

- **Version 3's unverified step is removed.** Version 3 relied on an unverified claim about sampling without replacement. No statement in version 4 depends on that question:
  - the NIH design is exactly i.i.d.;
  - ISIC and BraTS are complete enumerations.
- **No statement extends to:**
  - other scanners, hospitals, protocols, modalities or preprocessing;
  - other models or rules;
  - adversaries who choose probe images (Section 6.4).

  The paper's limitations state this.

---

## 2. Decision rule (version 2 unchanged; F3 fix added)

- Families, grids and messages are as in version 2, Sections 2.1–2.2. The extended GF(64) grid has K ∈ {16, …, 44} and 1236 grid configurations in total.
- **Bounded-distance decision** (version 2, Section 2.3):
  - erasures: the s least reliable symbols per block;
  - decoding: `creedsolo`, now with field activation, or `FastBCH`;
  - an output word is kept only if 2e′ + s ≤ n − K (RS) or ≤ 2t (BCH);
  - acceptance: every block decodes and the HMAC-SHA256 tag of the decoded content matches.
- Exact characterization of acceptance by mechanisms (a) and (b), as in version 2.

## 3. Feasibility: unchanged (version 2, Section 3)
For every selected configuration, the paper reports message, content and tag bits, n, K, rate, correction capacity, s, bit order and q.

## 4. Selection on C: unchanged rules; script `calib_v10.py` **[v3: provided]**

- **Operating points:**
  - OP-A: FA_C ≤ 8 of 89,700;
  - OP-B: FA_C = 0.
- **Ranking:** exact condition table for all 1236 configurations. The ranking engine is identical to `Coder.condition` on 40 configurations × 40 patients, with **0 mismatches**.
- **Full-decoding check [v4]:** the top-ranked configuration is fully decoded on all 89,700 C pairs. If it breaks the budget, the next ranked configuration is checked, **down the whole ranked list, with no cap**, until one passes or all candidates are exhausted. Every candidate checked is logged with its full-decoding FA and acceptance. If all are exhausted the cell is "none", with no fallback.
- **"none":** no fallback and no relaxation.
- **Hachimoji alphabet:** higher acceptance on C; GF(64) on ties.
- **NC rule:** highest acceptance within budget; ties go to fewer FA, then larger τ.
- **[v3] Approval guard:** `calib_v10.py` runs only if `approval_v10.json` records the SHA-256 of the approved `PROTOCOL_v10.md`.

## 5. Final evaluation on U: script `eval_final_v10.py` **[v3: provided]**

As in version 2, Section 5: every ordered pair of U is fully decoded, with no sampling, and every pair gets one of four statuses. The counts must sum to N(N − 1); the script checks this.

**[v3] E_tag:**
- E_tag = #reject_tag + #accept_other is an **observed count**: the pairs that reached a tag comparison with a code word other than the registrant's.
- E_tag × 2⁻ᵀ is reported **only as an idealized theoretical expectation**, under the assumption that truncated HMAC-SHA256 behaves as a random function for an adversary without k_h.
- It is never presented as an observed or estimated number of collisions.
- The *observed* tag collisions are #accept_other.

**Also computed:**
- genuine acceptance on U (200 × 24) and on the revision-9 test set (continuity);
- NC at τ = 1.00 and at the frozen τ;
- M1, M2 (only when FA ≥ 3), M3, and the patient-disjoint check.

## 6. Statistics **[v3: Hoeffding reference corrected, full derivation]**

### 6.1 Setting
- Draws X_1, …, X_N are i.i.d.: exactly so for NIH (with-replacement design); for ISIC and BraTS only under the superpopulation model of Section 1.3. X_i comprises the patient identity, the image, the encounter identifiers (hence the key) and the random content drawn for that draw.
- The directed decision is h(X_i, X_j) ∈ {0, 1}: accept probe j under record i. h is a fixed function of the two draws once the rule is frozen.
- **[v4] Impostor kernel:** h′(x, y) = h(x, y) if x and y are different patients, else 0. Same-patient draw pairs, which occur only in NIH U, are decoded and counted separately, but are never impostor attempts.
- FA = Σ_{i≠j} h′_ij, M = N(N − 1), p̂ = FA/M, and p′ = E h′(X_1, X_2).

### 6.2 Symmetrized directed-pair kernel
- g(x, y) = ½[h′(x, y) + h′(y, x)] ∈ {0, ½, 1}. It is symmetric, and E g(X_1, X_2) = p′.
- Since Σ_{i≠j} h′_ij = Σ_{i<j} [h′_ij + h′_ji] = 2 Σ_{i<j} g_ij:

  p̂ = U := (N choose 2)⁻¹ Σ_{i<j} g(X_i, X_j),

  which is the U-statistic of order 2 with kernel g.

### 6.3 Inequality and derivation (Hoeffding 1963)
- **Reference:** Hoeffding, W. (1963). Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association* 58(301), 13–30. doi:10.1080/01621459.1963.10500830.
  - Metadata verified against Crossref.
  - **Section:** the U-statistic result is in **Section 4, subsection (a)**, per the reviewer's correction. Version 2 said Section 5, which was wrong and was already marked unverified.
  - The full text could not be accessed here (JSTOR), so **the section and equation numbering are unverified — requires confirmation** against the full text before the paper cites them.
  - The derivation below does not depend on the numbering.

1. **Representation (Hoeffding's device).** Let k = ⌊N/2⌋. For a permutation π of {1, …, N}, let

   W_π = (1/k) Σ_{r=1}^{k} g(X_{π(2r−1)}, X_{π(2r)}).

   Averaging over all N! permutations gives U = (1/N!) Σ_π W_π, because every unordered pair appears equally often.
2. **Independence inside W_π.** The k pairs in W_π are disjoint, so W_π is a mean of k **independent**, identically distributed variables in [0, 1], each with mean p. This uses only the i.i.d. patients.
3. **Moment generating function.** For s > 0, exp(−sU) = exp(−s · avg_π W_π) ≤ avg_π exp(−sW_π) by convexity (Jensen). Since every W_π has the same distribution, E e^{−sU} ≤ E e^{−sW}, where W is a mean of k i.i.d. [0, 1] variables.
4. **Chernoff–Hoeffding step.** Write p for p′ in steps 4–7. For a variable G ∈ [0, 1] with mean p, convexity gives e^{−sG} ≤ 1 − G + G·e^{−s}, so E e^{−sG} ≤ 1 − p + p·e^{−s}. This is the bound behind Hoeffding's Theorem 1. Hence E e^{−sW} ≤ (1 − p + p e^{−s})^k, and

   inf_{s>0} e^{s(p−t)} (1 − p + p e^{−s})^k = exp(−k · KL(p − t ‖ p)), 0 < t < p,

   with KL(a ‖ b) = a ln(a/b) + (1 − a) ln((1 − a)/(1 − b)). This is Theorem 1 applied to 1 − g, i.e. the lower tail. By Markov's inequality and step 3,

   **P(U ≤ p − t) ≤ exp(−k · KL(p − t ‖ p)).**
5. **One-sided (1 − α) upper confidence bound.** For an observed u = p̂:

   p_U(u) = max{ q ≥ u : k · KL(u ‖ q) ≤ ln(1/α) }   (computed by bisection in `stats_v10.m1_upper`).

   **Coverage:**
   - The bound is monotone in u, and KL(x ‖ p) is decreasing in x on [0, p].
   - Therefore p_U(U) < p ⇔ U < x\*(p), where x\* < p solves k·KL(x\* ‖ p) = ln(1/α).
   - Hence P(p_U(U) < p) ≤ P(U ≤ x\*) ≤ exp(−k·KL(x\* ‖ p)) = α.
6. **Why dependence between pairwise decisions does not invalidate the bound.**
   - Decisions sharing a patient (h_ij, h_ik, h_ji, …) are dependent, and their dependence structure is unknown.
   - The derivation never uses independence between such decisions. It uses only:
     - (i) i.i.d. patients, giving independence of disjoint pairs in step 2;
     - (ii) convexity in step 3;
     - (iii) boundedness of g.
   - The price is that the effective sample size is k = ⌊N/2⌋, not N(N − 1). That is why the bound is conservative.
7. **Values.** With FA = 0, p_U = 1 − α^{1/k}:
   - 5.97 × 10⁻³ for N = 1000;
   - 7.09 × 10⁻³ for N = 842;

   - 3.84 × 10⁻³ for N = 1557;
   - NIH finite-population form, 5.97 × 10⁻³ / (1 − 1/30,305) = 5.97 × 10⁻³.

   A bound of 10⁻⁴ would need k ≈ 29,960 (≈ 60,000 patients).
8. **[v4] Finite-population bound for NIH.** p′ = (1 − 1/P)·p_NIH: two i.i.d. draws are different patients with probability 1 − 1/P, and h′ is 0 otherwise. So p_NIH ≤ p_U / (1 − 1/P) with the same 95% coverage, rigorously, for the finite population of P = 30,305 patients.
9. **[v4] Code ↔ protocol consistency** (`stats_v10.py`):
   - `m1_upper` solves k·KL(u ‖ q) = ln(1/α) by bisection with k = ⌊N/2⌋. Unit checks for N ∈ {300, 842, 1000, 1557} and FA ∈ {0, 1, 3, 10, 100}:
     - closed form 1 − α^{1/k} at FA = 0 (to 10⁻¹²);
     - the KL equation holds at the returned bound (to 10⁻⁹);
     - monotone in FA (0–50);
     - **all pass**.
   - `far_report` applies h′ through the patient identities; it divides by (1 − 1/P) only for the with-replacement design and labels census results as exact.
   - The coverage simulation (version 2, Section 8.3) found M1 coverage 100% in every setting.

### 6.4 Other quantities: unchanged from version 2, with labels
- **M2, patient-cluster bootstrap:** approximate. Reported only if FA ≥ 3. Simulated coverage: 54–64% with one expected event; 87–100% otherwise.
- **M3, jackknife and n_eff:** descriptive.
- **Patient-disjoint check:** a limited sanity check (500 / 421 pairs, exact Clopper–Pearson), superseded by M1.
- **Patient-involvement count:** descriptive only, never an interval.
- **Threat model:** the bound covers **random, non-adversarial** impostor images from the stated population. It says nothing about an attacker who searches for an image that verifies. That is a separate, cryptographic question, discussed only qualitatively.

### 6.5 Claim rule (unchanged)
"FAR ≤ 10⁻⁴ (95%)" only if M1 ≤ 10⁻⁴, which is impossible at these sizes. **10⁻⁴ is the calibration target only.**

---

## 7. Conformance testing **[v3]**: condition vs complete decoder vs independent reference

Three different tests, each reported separately:

| Test | Compares | What it can detect |
|---|---|---|
| **T1, theoretical condition** | The bounded-distance condition 2e + s ≤ bound, computed against the registrant's own symbols, vs the evaluated pipeline's `accept_genuine`. | That the pipeline accepts exactly when the genuine code word lies within the radius. This checks the theory–implementation link. It is not a test of an independent implementation. |
| **T2, independent complete pipeline** | The evaluated pipeline (`v10_decision.Coder`) vs `ref_v10.py`, an **independent implementation of the full acceptance pipeline**. The decision compared is accept/reject, plus whether the returned content equals the registered content. | Errors in any step: erasure selection, decoding, code-word reconstruction, content extraction, HMAC verification, and in registration (helper data compared bit for bit). |
| **T3, exhaustive search** | Brute-force nearest code word over all (2^m)^K code words vs the reference's decoder, for GF(4) and GF(8) blocks. | Errors in the reference decoder itself (no decoding assumptions). |

**`ref_v10.py` shares no decoding code with the pipeline:**
- **Symbols, key-stream order, Hachimoji rules and registration:** the revision-7 `HachimojiRS` class, which uses the pure-Python `reedsolo` encoder.
- **RS decoding:** a **Berlekamp–Welch** decoder (linear algebra over `galois.GF`) applied to the generalized-RS form of the `reedsolo` codes. This is a different algorithm from creedsolo's Berlekamp–Massey decoder.
  - The GRS form is c_j = v_j f(β_j), β_j = α⁻ʲ, v_j = α^(−j·nsym) ∏_{l=n}^{N−1}(β_j − β_l).
  - It was checked on 20 encodings each of 7 codes. A Berlekamp–Welch self-test corrects ⌊(n − K)/2⌋ errors in 30 of 30 trials for each of 8 codes.
- **BCH decoding:** `galois.BCH.decode` (independent of FastBCH), with the fill-0 / fill-1 erasure rule re-implemented.
- **Erasure selection, content extraction and the HMAC tag check:** re-implemented.
- **Field-table regression test** (F3): 400 interleaved decodings compared with pure-Python `reedsolo`.

**Coverage of configurations:**
- **Before approval (now):** a stratified grid sample on revision-9 test data. For each tag length T ∈ {32, 64, 128} × family (GF(64), GF(8), GF(4), BCH-63, BCH-255) × with/without erasures, 3 random feasible configurations: **87 configurations**, 6 patients each, 24 genuine + 99 impostor attempts per patient. Results: Section 8.
- **Before freezing (mandatory, `conformance_v10.py final`):** **every configuration selected on C**, for every seed, modality, family, T and OP, tested on C with:
  - all genuine attempts;
  - random impostor pairs (numpy `RandomState(0)`): **10,000 for RS configurations and 2,000 for BCH configurations**. The BCH reference (`galois`) costs about 5–20 ms per word and block, so 10,000 pairs per BCH configuration would take days;
  - **every impostor pair whose pipeline status is not reject_fail** (all 89,700 pairs pass through the pipeline first). This includes every accepted pair and every pair that reached a tag comparison. The test is therefore exhaustive for every pair the pipeline did *not* reject at decoding.

  `make_freeze_v10.py` refuses to freeze unless T1, T2, T3 and the registration checks all show 0 mismatches.

---

## 8. Test and benchmark results (revision-9 test data and synthetic data only)

### 8.1 Earlier checks (version 2): unchanged
- `creedsolo` = `reedsolo`.
- `FastBCH` = `galois`.
- T1-type agreement: 0 mismatches in 369,000 attempts.
- Message invariance.
- Vectorized NC.
- Coverage simulation.

### 8.2 **[v3]** New script checks

| Check | Result |
|---|---|
| `check_manifest.py` on the real manifest / on a corrupted manifest | PASS / FAIL with both faults reported |
| Calibration ranking engine vs `Coder.condition` (40 configurations × 40 patients, NIH test) | 0 mismatches |
| Vectorized NC table vs revision-9 `fast_nc` | 0 of 1560 mismatches (plus 10,050 of 10,050 in v2) |
| Field-table regression (F3) | 0 of 400 with the fix (330 of 400 without) |
| T1 / T2 / T3 on the 87-configuration grid (every family × T × with/without erasures; GF(4) with erasures is infeasible at T = 128) | **64,206 attempts** (10,980 accepted genuine, 53,177 rejected, 49 reaching a tag comparison with another code word, all tag-rejected). **T1: 0, T2: 0, T3: 0 of 1188, registration: 0 mismatches** |
| **[v4]** End-to-end dry run on the revision-9 test sets, **complete for BraTS, NIH and ISIC** (`dry_run_v10.py`; logs `dry_run_{brats,nih,isic}.log/.json`). It runs selection, then final conformance, then full evaluation with pair-count checks, then statistics. NIH includes 3 repeated draws to exercise the with-replacement design | 3 × 24 selected cells (all 4 families × T 32/64/128 × OP-A/OP-B), **70 distinct configurations**, **583,832 conformance attempts: T1 0, T2 0, T3 0 of 10,200, registration 0**. **707,544 evaluated pairs, every pair-count check passed.** The 144 accepted same-patient draw pairs (NIH) were correctly excluded from FA. NC OP-B was "none" for NIH (no τ with 0 FA). It was reported as such, with no fallback. No failures, exceptions or tracebacks |

### 8.3 Runtime
- **Calibration grid:** 36 s per 100 patients per modality, so about 5 min per modality and seed on C.
- **Final conformance on C** (reference implementation): about 2–15 min per RS configuration and about 5–30 min per BCH configuration. With at most about 100 distinct selected configurations over 3 seeds, that is **about 15–25 h**, the dominant cost.
- **Total after approval:** about 30–40 h wall, in several 2-hour runs. This replaces the 12–15 h of version 2, which did not include final conformance.
- **[v4] ISIC census:** U has 1557 patients, i.e. 2.42 M ordered pairs instead of 0.999 M. Total U pairs per seed are 4.13 M instead of 2.71 M, which adds about 3–5 h of full decoding. **Total about 35–45 h wall.**

---

## 9. Order of operations after approval

1. Authors write `approval_v10.json` with the SHA-256 of the approved `PROTOCOL_v10.md` (this document, finalized).
2. **C images only:** prepared on the laptop from the manifest, and copied.
3. `calib_v10.py` for seeds 0, 1, 2.
4. `conformance_v10.py final` on every selected configuration.
5. `make_freeze_v10.py`. It writes `frozen_v10.json` and `freeze_record.json` with:
   - **the protocol and approval record;**
   - **the frozen configuration list** (`frozen_v10.json`);
   - **the manifest** and the `check_manifest` output;
   - **the calibration selections and C feature files;**
   - **the conformance outputs** (grid and final);
   - **the benchmark outputs** (`runs/v10/bench`);
   - **every script** under `experiments/`, `models/`, `watermarking/`, `core/`, `evaluation/`, `training/`, and the root modules (`config.py`);
   - **[v4] the local import closure** (`dependency_closure.txt`, 28 modules). The freeze script refuses to run if any module in it is not hashed;
   - **[v4] third-party decoders:** the compiled `creedsolo` module, the `reedsolo` 1.7.0 source archive it was built from, and the pure-Python `reedsolo` used by the reference; `galois` is covered by its version and `pip freeze`;
   - the release listings the manifest was built from;
   - **the three model weights;**
   - **SHA-256 of each item above;**
   - the git commit, UTC time, `pip freeze`, Python version, `uname` and CPU model.

   The record is committed to your laptop and the project **before** step 6.
6. **U images:** prepared on the laptop and copied; the transfer is logged.
7. `eval_final_v10.py` for seeds 0, 1, 2. Its guard checks every hash and the git commit, and logs every access.
8. Any deviation goes to `DEVIATIONS.md`, and results after it are labelled exploratory.

## 10. Files for review

| File | Role |
|---|---|
| `experiments/build_manifest_v10.py`, `check_manifest.py` | partitions and machine check |
| `experiments/features_v10.py` | feature extraction for C/U |
| `experiments/calib_v10.py` | approval guard, calibration on C, selection |
| `experiments/conformance_v10.py`, `ref_v10.py` | T1/T2/T3 tests; independent reference |
| `experiments/eval_final_v10.py` | freeze guard, access log, final evaluation |
| `experiments/make_freeze_v10.py` | frozen list and freeze record |
| `experiments/v10_decision.py` | evaluated decision rule (Coder, FastBCH, vectorized NC) |
| `experiments/stats_v10.py`, `stats_v10_sim2.py` | M1/M2/M3, disjoint check, coverage simulation |
| `experiments/bench_v10.py`, `bench_v10_tags.py`, `dry_run_v10.py` | benchmarks and the dry run |
| `runs/v10/manifest_v10.json` | patient manifest (built, checked; SHA-256 `759339fe…5a9a2d`) |
| `runs/v10/dependency_closure.txt`, `freeze_coverage_check.txt` | **[v4]** local import closure of the pipeline, and the pre-freeze coverage self-test of the freeze record |
| `runs/v10/conf/dry_run_{brats,nih,isic}.log/.json` | **[v4]** complete dry-run logs |


---

## 11. **[v4]** Checklist of the four targeted changes and the final checks

| # | Requirement | Change | Check | Status |
|---|---|---|---|---|
| 1 | ISIC sampling: no unverified without-replacement assumption | ISIC U = **census of all 1557** remaining eligible patients; FA / M is the exact FAR of that set. M1 for ISIC and BraTS is labelled a superpopulation-model bound, not a 95% guarantee. For consistency, NIH U = **i.i.d. draws with replacement** from P = 30,305, giving a **rigorous** finite-population bound M1 / (1 − 1/P). Same-patient draw pairs are excluded through the kernel h′ | `check_manifest.py` PASS, including the census checks; negative test FAIL with all 4 faults; dry run on NIH with repeated draws (144 same-patient acceptances excluded) | **PASS** |
| 2 | No 50-candidate limit | `calib_v10.select_modality` walks the whole ranked list until a configuration passes the full-decoding budget or all are exhausted. Every candidate is logged; "none" otherwise, with no fallback | code (`for i in order:`); dry run logs the steps per cell | **PASS** |
| 3 | Complete dry run, BraTS, NIH, ISIC | `dry_run_v10.py` on all three modalities | 70 configurations, 583,832 attempts: T1/T2/T3/registration 0 mismatches; 707,544 pairs, all pair-count checks passed; no exceptions | **PASS** |
| 4a | Hoeffding derivation ↔ implementation | Derivation (Section 6.3, steps 1–8) and `stats_v10.m1_upper` / `far_report` describe the same procedure: k = ⌊N/2⌋, KL inversion, h′, and division by (1 − 1/P) for NIH only | unit checks: closed form at FA = 0, the KL equation at the returned bound, monotonicity (N ∈ {300, 842, 1000, 1557}); coverage simulation M1 = 100% | **PASS** |
| 4b | Hoeffding section number | Section 4(a), per the reviewer | full text not accessible here | **Unverified — requires confirmation** (does not affect the derivation) |
| 4c | All scripts and dependencies in the freeze record | `make_freeze_v10.py` hashes every module of the import closure, the decoders, the listings, the weights, the protocol, the manifest, the benchmark, conformance and calibration outputs, and the frozen list; it refuses if the closure is not covered | `--check-coverage`: 112 files, 107 present now; the 5 missing are produced by the pipeline before freezing (protocol, approval, frozen list, manifest-check output, final conformance) | **PASS** |

**Not done, and still prohibited until explicit approval:**
- freezing;
- C image transfer and C features;
- any U feature or score.
