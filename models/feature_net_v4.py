# models/feature_net_v4.py  (v4)
"""
AIF-Net v4: learned attack-invariant feature hashing network with a corrected
local code.

Input: an 8-bit gray-level image, resized to 256 x 256 (Lanczos) and
standardized per image (zero mean, unit variance).

A residual CNN (five stride-2 stages, 32-256 channels) produces an
8 x 8 x 256 feature map, from which
  * the global head (4 x 4 adaptive average pooling, two linear layers) gives
    G = 256 global logits, and
  * the local head (two 1 x 1 convolutions) gives 4 logits for each of the
    8 x 8 cells of the image (L = 256 local logits).
Both codes end in a batch normalization without affine parameters. In v4 the
local code is normalized per cell and channel (BatchNorm1d over the flattened
256 local logits), so that every local bit has its own threshold. In v3 the
local logits were normalized per channel over all cells, which let the cell
position alone decide the sign; the v3 local bits were therefore constant
across images and carried no image information (see the v4 README).

The binary features are the signs of the 512 standardized logits; their
magnitudes are used for reliability-aware bit selection
(watermarking/zero_watermark.py).
Inference is deterministic: float32, eval mode, one thread, fixed weights.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

N_GLOBAL = 256          # global feature bits
N_LOCAL = 4             # bits per cell of the 8 x 8 grid
GRID = 8
IN_SIZE = 256           # network input size
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "aifnet_v4.pt"


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
    """Input (B,1,256,256) in [0,1]; output global logits (B,256) and local logits (B,256),
    the local logits ordered cell-major (cell c = row * 8 + col, 4 logits per cell)."""

    def __init__(self, width=32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(conv_bn(1, w, 2), ResBlock(w))                 # 128
        self.s2 = nn.Sequential(conv_bn(w, 2 * w, 2), ResBlock(2 * w))           # 64
        self.s3 = nn.Sequential(conv_bn(2 * w, 4 * w, 2), ResBlock(4 * w))       # 32
        self.s4 = nn.Sequential(conv_bn(4 * w, 4 * w, 2), ResBlock(4 * w))       # 16
        self.s5 = nn.Sequential(conv_bn(4 * w, 8 * w, 2), ResBlock(8 * w))       # 8
        self.local_head = nn.Sequential(nn.Conv2d(8 * w, 64, 1), nn.ReLU(inplace=True),
                                        nn.Conv2d(64, N_LOCAL, 1), nn.Flatten())
        self.local_bn = nn.BatchNorm1d(N_LOCAL * GRID * GRID, affine=False)
        self.global_head = nn.Sequential(nn.Flatten(), nn.Linear(8 * w * 16, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, N_GLOBAL), nn.BatchNorm1d(N_GLOBAL, affine=False))

    def forward(self, x):
        m = x.mean(dim=(2, 3), keepdim=True)
        s = x.std(dim=(2, 3), keepdim=True) + 1e-3
        x = (x - m) / s
        f = self.s5(self.s4(self.s3(self.s2(self.stem(x)))))
        loc = self.local_head(f)                                   # (B, 4*64), channel-major
        loc = loc.view(-1, N_LOCAL, GRID * GRID).transpose(1, 2).reshape(-1, N_LOCAL * GRID * GRID)
        return self.global_head(F.adaptive_avg_pool2d(f, 4)), self.local_bn(loc)


def prepare(image: np.ndarray) -> np.ndarray:
    """uint8 image of any size -> float32 256 x 256 in [0, 1] (Lanczos resampling)."""
    im = Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8))
    if im.size != (IN_SIZE, IN_SIZE):
        im = im.resize((IN_SIZE, IN_SIZE), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


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


TTA_VIEWS = ("identity", "blur", "rot+2", "rot-2", "scale")


def tta_views(image: np.ndarray):
    """Mild views averaged at registration and verification (test-time augmentation)."""
    from scipy.ndimage import gaussian_filter, rotate
    f = image.astype(np.float32)
    h, w = image.shape
    small = Image.fromarray(image).resize((int(w * 0.85), int(h * 0.85)), Image.LANCZOS).resize((w, h), Image.LANCZOS)
    views = [image,
             np.clip(gaussian_filter(f, 0.7), 0, 255).astype(np.uint8),
             np.clip(rotate(f, 2, reshape=False, order=1, mode="constant"), 0, 255).astype(np.uint8),
             np.clip(rotate(f, -2, reshape=False, order=1, mode="constant"), 0, 255).astype(np.uint8),
             np.asarray(small)]
    # optional extra rotated views (revision 9 study; default: none). AIF_TTA_ROT="10,20" adds +-10 and +-20 deg
    import os
    extra = os.environ.get("AIF_TTA_ROT", "")
    for a in [float(x) for x in extra.split(",") if x.strip()]:
        for sgn in (1, -1):
            views.append(np.clip(rotate(f, sgn * a, reshape=False, order=1, mode="constant"), 0, 255).astype(np.uint8))
    return views


@torch.no_grad()
def logits(image: np.ndarray, path=DEFAULT_WEIGHTS, tta=True):
    """image: 2-D uint8 array. Returns the 512 standardized logits
    (256 global, then 256 local in cell-major order), averaged over the TTA views."""
    net = load(path)
    views = tta_views(image) if tta else [image]
    x = torch.from_numpy(np.stack([prepare(v) for v in views]))[:, None]
    g, l = net(x)
    return torch.cat([g, l], 1).mean(0).numpy()
