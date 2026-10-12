# hachimoji_watermark v4

v4 is the implementation behind the revision-4 manuscripts (EAAI and JAIR versions).

## Changes from v3

1. **AIF-Net v4 local code fixed.** In v3, the 3 local bits per block were the same for every image. Their sign depended only on the block position, because the local logits were normalized per channel over all positions. As a result, about one third of the watermark positions carried no image information, which made the v3 NC values look somewhat better than the network really was.
   - v4 outputs 512 bits: 256 global bits and 4 bits per cell of an 8×8 grid.
   - Each local bit is normalized and trained on its own (`models/feature_net_v4.py`, `training/train_feature_net_v4.py`).
   - Retrained for 3000 steps (`models/aifnet_v4.pt`, SHA-256 `1afc5aba22a708dcff0c825272422915bb2c8ae825fe05dd9af05dcc37f2a295`).
2. **Test-time augmentation.** The logits are averaged over 5 mild views of the image (`feature_net_v4.tta_views`).
3. **Reliability-aware bit selection.** The watermark is bound only to the fraction q of feature bits with the largest |logit|.
   - The selected indices are stored in the record (`ZeroWatermarkV2(..., select_q=q)`).
   - q = 0.7 was chosen on validation data by equal error rate (`experiments/select_q_v4.py`).
4. **Threshold.** The robust-layer threshold, selected on validation volumes, is τ = 1.00 (`experiments/select_threshold_v4.py`).
5. **Re-implemented baselines** (`models/baselines.py`): VGG19+LBP after Gharib et al. (2025) and FocalNet-T after Li et al. (2026), with real ImageNet weights (see `models/pretrained/README.md`). They run in exactly the same pipeline. The handcrafted DWT–DCT–SVD features serve as the third baseline.
6. **Prior-study settings.** Extra attacks reproduce the settings used in prior papers: Gaussian noise with variance 0.01 and 0.1, rotation 25°, and median 7×7.
7. **Tables and figures from results.** All tables and figures are generated from the results files (`experiments/make_tables_v4.py`, `figures/make_figures_v4.py`).

## Main results (BraTS 2023, 20 test slices, mean NC over 24 operations)

| Features | BraTS | NIH | ISIC | USC-SIPI | Different slices (lower is better) |
|---|---|---|---|---|---|
| Handcrafted DWT–DCT–SVD | 0.940 | 0.899 | 0.864 | 0.919 | 0.730 |
| VGG19 + LBP (Gharib-style), with selection | 0.901 | 0.916 | 0.903 | 0.937 | 0.879 |
| FocalNet-T (Li-style), with selection | 0.918 | 0.928 | 0.923 | 0.966 | 0.883 |
| **AIF-Net v4 + TTA + selection (proposed)** | **0.995** | **0.992** | **0.993** | **0.980** | **0.711** |

## Reproduce

```
pip install numpy==2.4.4 scipy==1.17.1 PyWavelets==1.8.0 scikit-image==0.26.0 pydicom==3.0.2 nibabel==5.4.2 pillow torch matplotlib timm h5py
python training/train_feature_net_v4.py <train_set.npz> models 3000
python experiments/select_q_v4.py <train_set.npz> results_v4/q_selection.json
python experiments/select_threshold_v4.py <train_root> results_v4/threshold_selection_v4.json learned_v4 0.7
python experiments/run_all_v4.py <datasets_root> results_v4 learned_v4 0.7 1      # proposed (with volumes)
python experiments/run_all_v4.py <datasets_root> results_v4 learned_v4 none 0     # ablations
python experiments/run_all_v4.py <datasets_root> results_v4 learned_v4_notta none 0
python experiments/run_all_v4.py <datasets_root> results_v4 learned_v4_notta 0.7 0
python experiments/run_all_v4.py <datasets_root> results_v4 handcrafted none 0    # baselines
python experiments/run_all_v4.py <datasets_root> results_v4 handcrafted_rel 0.7 0
python experiments/run_all_v4.py <datasets_root> results_v4 vgg19_lbp none 0
python experiments/run_all_v4.py <datasets_root> results_v4 vgg19_lbp 0.7 0
python experiments/run_all_v4.py <datasets_root> results_v4 focalnet none 0
python experiments/run_all_v4.py <datasets_root> results_v4 focalnet 0.7 0
python experiments/make_tables_v4.py results_v4 results_v4/tables
python figures/make_figures_v4.py results_v4 figures/paper_figures_v4
python experiments/summarize_v4.py results_v4 1.0
```

Notes:
- `NPROC=1` runs a single worker process.
- `experiments/tune_baselines_v4.py` checks other binarization rules for the baselines on validation data. The paper keeps the per-image median.
- The v3 files are unchanged and still run. The handcrafted results are identical in v3 and v4.

## Test collections of 2026 studies (added 12 Oct 2026)

`experiments/run_2026_v4.py` runs a feature configuration on 8 public collections used by 2026 studies, with 20 test images each:

| Collection | Used by (2026) | Source |
|---|---|---|
| Brain tumor MRI (Nickparvar) | Taj; Niharika | Kaggle |
| CT kidney | Niharika | Kaggle |
| Brain MRI (Navoneel) | George | Kaggle |
| TCGA-LGG FLAIR (Buda et al.) | George | Kaggle |
| MIAS mammography | Suzgen | Kaggle |
| Bone break X-ray | Suzgen | Kaggle |
| Retina fundus | Suzgen | Kaggle |
| COVID-19 chest X-ray (Cohen) | Anand | GitHub `ieee8023/covid-chestxray-dataset` @ 78543292 |

How the subsets were made:
- The image list and SHA-256 checksums are in `datasets2026/manifest.json` and `datasets2026/manifest_covid.json`.
- The Kaggle subsets were extracted with HTTP range requests. They are the first files per class in sorted order.
- All images are converted to gray and resized to 512×512 (Lanczos). For TCGA-LGG the FLAIR channel is used.

```
python experiments/run_2026_v4.py <datasets2026_dir> results_2026 learned_v4 0.7
# repeat for: handcrafted none, handcrafted_rel 0.7, vgg19_lbp none/0.7, focalnet none/0.7
python experiments/make_tables_2026.py results_2026 results_2026/tables
```

Results, mean NC over the 24 operations, averaged over the 8 collections:

| Configuration | Mean NC |
|---|---|
| Handcrafted | 0.925 |
| VGG19+LBP with selection | 0.928 |
| FocalNet-T with selection | 0.952 |
| **Proposed** | **0.990** (0.987–0.994 per collection) |

Not included:
- Xiong 2026: private data.
- Li 2026 (MedPix): no bulk download.
- Zhu 2026 (MSD): 7–27 GB, and its brain task comes from BraTS.
- Jagadeesh 2026 and Hung 2026: image sources not named.
