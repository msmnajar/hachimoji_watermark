# Pretrained weights for the re-implemented baselines

These third-party weights are not part of the repository. Download them into this folder before running the `vgg19_lbp` or `focalnet` baselines.

```
# VGG19 ImageNet weights, Keras version (no top). Converted to PyTorch on first use.
curl -L -o vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5 \
  https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
# FocalNet-T (SRF) ImageNet checkpoint, official release
curl -L -o focalnet_tiny_srf.pth \
  https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_tiny_srf.pth
```

Expected SHA-256 values (the experiments store them in the results files):

| File | SHA-256 |
|---|---|
| `vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5` | `bde7da0dfd621cbf86e11ce04e4946a9a42634c2cfd21318bba090d860699e35` |
| `focalnet_tiny_srf.pth` | `336b9a912dbf1e5b909d0580c4df884b8f56aff3b0ad3af394516762c12784dc` |

Requirements: `pip install timm h5py`. The runs used timm 1.0.30 and h5py 3.16.0.
