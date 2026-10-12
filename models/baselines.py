# models/baselines.py  (v4)
"""
Re-implemented feature extractors of prior zero-watermarking schemes, used as
same-data baselines. Each extractor follows the published description as far as
it is specified; where a paper leaves a detail open (pooling, binarization rule,
input size), the choice made here is stated in the docstring and in the paper.

All extractors return real-valued features v and a binarization threshold
(the median of each feature part), so that the zero-watermark pipeline can use
them exactly like the AIF-Net logits: bit = v > threshold, and the reliability
of a bit is |v - threshold| divided by the median absolute deviation of its part.

  vgg19_lbp  Gharib et al. (2025): pre-trained VGG19 CONV5-4 features fused with
             local binary pattern (LBP) features. Here: 224 x 224 input (gray image
             replicated to three channels, Caffe/ImageNet mean subtraction), CONV5-4
             after ReLU, global average pooling (512 values); uniform LBP (P = 8,
             R = 1) histograms (59 bins) over 4 x 4 regions of the 256 x 256 image
             (944 values). Each part is binarized at its median: 1456 bits.
  focalnet   Li et al. (2026): DWT pre-processing followed by FocalNet features.
             Here: level-1 Haar LL band, rescaled to 0-255 and resized to 224 x 224,
             ImageNet-pretrained FocalNet-T (SRF); global average pooling of the
             last two stages (768 + 384 values), each binarized at its median:
             1152 bits. The attention pre-processing of Li et al. is not reproduced.
Pre-trained weights: Keras VGG19 ImageNet weights (no top), converted to PyTorch,
and the official FocalNet-T (SRF) ImageNet checkpoint.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

WEIGHTS_DIR = Path(__file__).resolve().parent / "pretrained"
_M = {}


# ------------------------------------------------------------------ VGG19 (Keras weights)
class VGG19Conv54(nn.Module):
    CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]

    def __init__(self):
        super().__init__()
        layers, c = [], 3
        for v in self.CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]
                c = v
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


def _vgg():
    if "vgg" not in _M:
        torch.set_num_threads(1)
        net = VGG19Conv54()
        pt = WEIGHTS_DIR / "vgg19_conv54.pt"
        if pt.exists():
            net.load_state_dict(torch.load(pt, map_location="cpu"))
        else:
            import h5py
            f = h5py.File(WEIGHTS_DIR / "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", "r")
            names = [n for n in f.attrs["layer_names"].astype(str) if "conv" in n]
            convs = [m for m in net.features if isinstance(m, nn.Conv2d)]
            assert len(names) == len(convs) == 16
            for n, m in zip(names, convs):
                g = f[n]
                W = np.array(g[f"{n}_W_1:0"]); b = np.array(g[f"{n}_b_1:0"])
                m.weight.data = torch.from_numpy(W.transpose(3, 2, 0, 1).copy())
                m.bias.data = torch.from_numpy(b)
            torch.save(net.state_dict(), pt)
        net.eval()
        _M["vgg"] = net
    return _M["vgg"]


def _lbp_uniform_hist(img256):
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(img256, 8, 1, method="nri_uniform")   # 59 codes
    out = []
    s = img256.shape[0] // 4
    for i in range(4):
        for j in range(4):
            h, _ = np.histogram(lbp[i*s:(i+1)*s, j*s:(j+1)*s], bins=59, range=(0, 59))
            out.append(h / h.sum())
    return np.concatenate(out)


@torch.no_grad()
def features_vgg19_lbp(image):
    net = _vgg()
    im = np.asarray(Image.fromarray(image).resize((224, 224), Image.LANCZOS), np.float32)
    x = np.stack([im - 103.939, im - 116.779, im - 123.68])[None]   # BGR mean (gray: same values)
    f = net(torch.from_numpy(x)).mean(dim=(2, 3))[0].numpy()
    lbp = _lbp_uniform_hist(np.asarray(Image.fromarray(image).resize((256, 256), Image.LANCZOS)))
    return [f, lbp]


# ------------------------------------------------------------------ FocalNet-T
def _focal():
    if "focal" not in _M:
        import timm
        from timm.models import focalnet as tf
        torch.set_num_threads(1)
        m = timm.create_model("focalnet_tiny_srf", pretrained=False)
        ck = torch.load(WEIGHTS_DIR / "focalnet_tiny_srf.pth", map_location="cpu", weights_only=False)["model"]
        m.load_state_dict(tf.checkpoint_filter_fn(ck, m))
        m.eval()
        _M["focal"] = m
    return _M["focal"]


@torch.no_grad()
def features_focalnet(image):
    import pywt
    m = _focal()
    LL = pywt.dwt2(image.astype(np.float64), "haar")[0]
    LL = ((LL - LL.min()) / (LL.max() - LL.min() + 1e-8) * 255).astype(np.uint8)
    im = np.asarray(Image.fromarray(LL).resize((224, 224), Image.LANCZOS), np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
    x = torch.from_numpy(((np.stack([im] * 3) - mean) / std)[None])
    x = m.stem(x)
    outs = []
    for layer in m.layers:
        x = layer(x)
        outs.append(x)
    s3 = outs[2].mean(dim=(2, 3))[0].numpy()
    s4 = m.norm(outs[3]).mean(dim=(2, 3))[0].numpy()
    return [s4, s3]


BASELINES = {"vgg19_lbp": features_vgg19_lbp, "focalnet": features_focalnet}


def weights_sha256(name):
    fn = {"vgg19_lbp": "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", "focalnet": "focalnet_tiny_srf.pth"}[name]
    return hashlib.sha256((WEIGHTS_DIR / fn).read_bytes()).hexdigest()
