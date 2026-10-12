# hachimoji_watermark v3

v3 is the implementation for the EAAI manuscript (revision 3). The changes from v2:

- **Learned features (AIF-Net).** The handcrafted DWT–DCT/SVD features are replaced by a learned, attack-invariant feature hashing network, trained without labels.
- **Same pipeline.** The encounter-bound HMAC key, 6D hyperchaotic key stream, per-block Hachimoji coding, ROI-dependent diffusion and two-layer volume chain are unchanged from v2.
- **Ablation.** The v2 features remain available as `features="handcrafted"`, so the two extractors can be compared.

## What is new in v3

| Part | File | Description |
|---|---|---|
| AIF-Net | `models/feature_net.py` | Residual CNN with 4.6 M parameters on a 256×256 standardized input. It outputs 256 global logits and 3 local logits per LL2 block (sampled at each 32×32 block centre). The bits are the signs of the batch-normalized logits. |
| Weights | `models/aifnet_v3.pt` | Trained weights. Their SHA-256 is written to the results file and must be stored with the trusted authority. |
| Training set | `training/build_training_set.py` | BraTS cases 101–190 (test: 1–20), plus 100 NIH and 100 ISIC images that are disjoint from the test images. TCIA and USC-SIPI are never used for training. |
| Training | `training/train_feature_net.py` | Self-supervised training, no labels. Loss terms: invariance, quantization, per-domain variance, covariance and InfoNCE. Hard negatives are different patients at the same depth and nearby slices of the same patient. Uses 12 attack families with random strengths. |
| Feature switch | `watermarking/zero_watermark.py` | `ZeroWatermarkV2(seed, arnold, features="learned" \| "handcrafted")`. The record stores which extractor was used, and verification refuses a mismatch. |
| Threshold | `experiments/select_threshold.py` | Selects the robust-layer threshold on validation volumes (AIF-Net 0.98, handcrafted 0.92). |
| Experiments | `experiments/run_all_v3.py` | All paper experiments (E1–E6), plus 6 attacks whose type or strength is outside the training augmentation. |
| Figures | `figures/make_figures_v3.py` | Fig. 1–8 of the manuscript, with embedded fonts. |
| Config | `config.py` | Now documents the parameters actually used. The stale v1 values are removed. |

## Reproduce
```
pip install numpy==2.4.4 scipy==1.17.1 PyWavelets==1.8.0 scikit-image==0.26.0 pydicom==3.0.2 nibabel==5.4.2 pillow torch matplotlib
# 1. training set (paths as in the thesis datasets folder)
python training/build_training_set.py <train_root> train_set.npz
# 2. train AIF-Net (CPU, ~2 h on 2 cores)
python training/train_feature_net.py train_set.npz models 2500
# 3. experiments (the handcrafted run reproduces the v2 numbers exactly)
python experiments/run_all_v3.py <datasets_root> results learned
python experiments/run_all_v3.py <datasets_root> results handcrafted
# 4. robust-layer threshold, chosen on validation volumes only (BraTS cases 183-190)
python experiments/select_threshold.py <train_root> results/threshold_selection.json
# 5. figures and a printout of every number used in the paper
python figures/make_figures_v3.py results figures/paper_figures_v3
python experiments/summarize.py results/results_v3_learned.json
```
`<datasets_root>` must contain `brats2023/`, `TCIA/RIDER_Breast_MRI/`, `USC_SIPI/`, `NIH_ChestXray14/images/` and `ISIC2020/Training_JPEG/train/`, with the same files as in v2.

## Main results (BraTS 2023, 20 test slices, mean NC)

| Attack | Handcrafted (v2) | AIF-Net (v3) |
|---|---|---|
| JPEG Q50 | 0.9895 | 0.9961 |
| Gaussian σ = 20 | 0.9445 | 0.9886 |
| Salt-and-pepper 1% | 0.9246 | 0.9933 |
| Rotation 5° | 0.9040 | 0.9916 |
| Rotation 10° | 0.8367 | 0.9808 |
| Histogram equalization | 0.8853 | 0.9552 |
| Mean of 6 unseen attacks | 0.907 | 0.954 |

- **Reproducibility:** the handcrafted run (`results/results_v3_handcrafted.json`) reproduces every value in `results/results_v2.json` exactly.
- **Weights:** the SHA-256 of `models/aifnet_v3.pt` is `a9582c1d894f21b9e3984ef341bb01a7d9af0ea6829c65611e3df485dd34b503`.
