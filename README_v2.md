# hachimoji_watermark v2

Corrected implementation used for the results in `hachimoji_watermark_paper_rev2.tex`.
The v1 folder (`hachimoji_watermark`) is left untouched.

## What changed and why

| # | v1 problem | v2 fix |
|---|---|---|
| 1 | `generate_index_sequence(1, 384)` min-max normalised a single value -> always 0, so every image used Hachimoji rule 0 and the chaotic system had no effect on the MWK | Integers taken as `floor(frac(abs(s)*1e6)*m)`; one rule per LL2 block and per group of 8 global bits |
| 2 | 6D system `(10, 28, 8/3, 1, 1, 1)`: measured Lyapunov spectrum (1e6 RK4 steps) (-0.000, -0.004, -0.015, -0.333, -9.832, -27.816): no positive exponent, i.e. not chaotic | System of Wang et al. 2019 used by Wu et al. 2024 (Phys. Scr. 99, 045252, Eq. 1): a=10, b=8/3, c=28, d=-1, e=8, r=3. Measured spectrum (1e6 RK4 steps): (0.774, 0.312, -0.001, -0.410, -0.644, -14.699) -> two positive exponents |
| 3 | 1000 warm-up steps of 1e-3: a 1e-15 change of x0 left the output unchanged | RK4, h = 5e-3, 10,000 warm-up steps (50 time units) |
| 4 | Key = SHA-256(DICOM fields): no secret | HMAC-SHA256 with a 256-bit institutional key K_inst held by the trusted authority |
| 5 | Hachimoji pairs B-Z, P-S | B-S, P-Z (Hoshika et al. 2019) |
| 6 | ROI 7-stage pipeline computed and discarded; "57.5% saving" was a formula | ROI-block and global positions pass add/xor/sub/inversion inside the MWK; background-block positions one XOR; saving measured |
| 7 | BraTS/USC-SIPI StudyUID from Python `hash()` (salted per run) | SHA-256-derived UID |
| 8 | Features gave NC ~0.93 between different patients' slices | 255 global DWT-DCT sign bits added (cross-patient NC 0.73) |
| 9 | Volume check: single NC >= 0.999 rule (flags JPEG) | Robust layer (NC >= tau) + fragile SHA-256 chain over pixel data |
| 10 | TCIA: first 20 files = 20 slices of one series of one patient | 26 non-blank slices from 13 series, 4 studies, 2 patients |

## Run
```
pip install numpy scipy pywavelets scikit-image nibabel pydicom pillow
python experiments/run_all_v2.py <datasets_root> <out_dir>
python experiments/e3_rerun.py        # TCIA subset with blank slices excluded
```
`<datasets_root>` must contain brats2023/, TCIA/RIDER_Breast_MRI/, USC_SIPI/, NIH_ChestXray14/images/, ISIC2020/Training_JPEG/train/.
Results: results/results_v2.json, results/lyapunov_spectrum.txt.
