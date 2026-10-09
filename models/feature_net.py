# models/feature_net.py  (v3)
"""
Learned attack-invariant feature hashing network (AIF-Net).

Input: an 8-bit gray-level image, resized to 256 x 256 (Lanczos) and
standardized per image (zero mean, unit variance), so that the network sees
every modality at the same scale and is insensitive to global gain/offset.

A small residual CNN (five stride-2 stages, 32-256 channels, 4.6 M
parameters in total) produces an 8 x 8 x 256 feature map, from which
  * the global head (4 x 4 adaptive average pooling, two linear layers) gives
    G = 256 global logits, and
  * the local head (two 1 x 1 convolutions) gives 3 logits per cell of an
    8 x 8 grid; this map is sampled bilinearly at the center of every LL2
    block footprint (32 x 32 pixels of the original image), so every block of
    the zero-watermark has 3 local logits.
Both heads end in a batch normalization without affine parameters, so every
logit is standardized over the training distribution; the binary features
used by the zero-watermark are the signs of these standardized logits.

The network is trained without labels (training/train_feature_net.py).
Inference is deterministic: float32, eval mode, one thread, fixed weights.
The SHA-256 of the weight file is part of the method configuration and is
stored with the trusted authority, so that registration and verification use
exactly the same feature extractor.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

N_GLOBAL = 256          # global feature bits
N_LOCAL = 3             # bits per LL2 block
IN_SIZE = 256           # network input size
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "aifnet_v3.pt"


def conv_bn(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(x + self.b2(self.c2(y)), inplace=True)


class AIFNet(nn.Module):
    """Input (B,1,256,256) in [0,1]; output global logits (B,256) and local logits (B,3,8,8)."""

    def __init__(self, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(conv_bn(1, w, 2), ResBlock(w))                 # 128
        self.s2 = nn.Sequential(conv_bn(w, 2 * w, 2), ResBlock(2 * w))           # 64
        self.s3 = nn.Sequential(conv_bn(2 * w, 4 * w, 2), ResBlock(4 * w))       # 32
        self.s4 = nn.Sequential(conv_bn(4 * w, 4 * w, 2), ResBlock(4 * w))       # 16
        self.s5 = nn.Sequential(conv_bn(4 * w, 8 * w, 2), ResBlock(8 * w))       # 8
        self.local_head = nn.Sequential(nn.Conv2d(8 * w, 64, 1), nn.ReLU(inplace=True),
                                        nn.Conv2d(64, N_LOCAL, 1),
                                        nn.BatchNorm2d(N_LOCAL, affine=False))
        self.global_head = nn.Sequential(nn.Flatten(), nn.Linear(8 * w * 16, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, N_GLOBAL), nn.BatchNorm1d(N_GLOBAL, affine=False))

    def forward(self, x):
        m = x.mean(dim=(2, 3), keepdim=True)
        s = x.std(dim=(2, 3), keepdim=True) + 1e-3
        x = (x - m) / s
        f = self.s5(self.s4(self.s3(self.s2(self.stem(x)))))
        return self.global_head(F.adaptive_avg_pool2d(f, 4)), self.local_head(f)


def prepare(image: np.ndarray) -> np.ndarray:
    """uint8 image of any size -> float32 256 x 256 in [0, 1] (Lanczos resampling)."""
    im = Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8))
    if im.size != (IN_SIZE, IN_SIZE):
        im = im.resize((IN_SIZE, IN_SIZE), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def to_block_grid(local_logits: torch.Tensor, grid, shape):
    """(B,3,8,8) -> (B,3,gh,gw): bilinear sample of the local map at the center of
    every 32 x 32-pixel LL2 block footprint of the original H x W image."""
    gh, gw = grid
    H, W = shape
    cy = (torch.arange(gh, dtype=torch.float32) * 32 + 16) / H * 2 - 1
    cx = (torch.arange(gw, dtype=torch.float32) * 32 + 16) / W * 2 - 1
    yy, xx = torch.meshgrid(cy, cx, indexing="ij")
    g = torch.stack([xx, yy], -1)[None].expand(local_logits.shape[0], -1, -1, -1)
    return F.grid_sample(local_logits, g, mode="bilinear", padding_mode="border", align_corners=False)


# ---------------------------------------------------------------- inference
_MODEL = {}


def weights_sha256(path=DEFAULT_WEIGHTS):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path=DEFAULT_WEIGHTS):
    path = str(path)
    if path not in _MODEL:
        torch.set_num_threads(1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = AIFNet(ck.get("width", 32))
        net.load_state_dict(ck["state_dict"])
        net.eval()
        _MODEL[path] = net
    return _MODEL[path]


@torch.no_grad()
def logits(image: np.ndarray, grid, path=DEFAULT_WEIGHTS):
    """image: 2-D uint8 array; grid: (gh, gw) LL2 block grid.
    Returns global (256,) and local (3, gh, gw) float32 logits."""
    net = load(path)
    x = torch.from_numpy(prepare(image))[None, None]
    g, l = net(x)
    return g[0].numpy(), to_block_grid(l, grid, image.shape)[0].numpy()
