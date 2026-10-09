# training/train_feature_net.py  (v3)
"""
Self-supervised training of AIF-Net (models/feature_net.py).

Every mini-batch mixes 12 BraTS slices and 6 chest-radiograph / dermoscopy
images. The BraTS slices come from 6 volumes at the same relative depth
(different patients, similar anatomy) and, for each volume, a second slice
3-15 slices away (same patient, nearby anatomy); both act as hard negatives. For every image x a random attack chain A (one or two operations from
rand_attack, applied at the native resolution) gives a positive copy A(x).
With u = tanh(1.5 z(x)) and v = tanh(1.5 z(A(x))), z the batch-normalized
logits, the loss combines

  invariance    mean((u - v)^2)                            same image -> same bits
  quantization  mean((|u| - 1)^2)                          bits away from the sign boundary
  variance      mean_k relu(0.9 - std_k(u)), per domain    every bit varies within each domain
  covariance    sum_{k != l} cov_kl^2 / K                  decorrelated bits
  InfoNCE       each x must identify its own A(x) among the batch (T = 0.1)

for the 256 global bits. The 3 local bits of the 8 x 8 cell map use the first
four terms, with every cell of every image as a sample.

Usage: python training/train_feature_net.py <train_set.npz> <out_dir> [n_steps]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feature_net import AIFNet, prepare
from evaluation import attacks as A

W = dict(inv=10.0, quant=1.0, var=2.0, cov=0.04, nce=0.5)
N_BRATS_VOLS, N_BIG = 6, 6


# ---------------------------------------------------------------- augmentation
def rand_attack(img, rs):
    """One training attack with a random strength."""
    k = rs.randint(13)
    if k == 0:
        return A.gaussian_noise(img, rs.uniform(2, 45))
    if k == 1:
        return A.salt_and_pepper(img, rs.uniform(0.002, 0.10))
    if k == 2:
        return A.jpeg_compression(img, int(rs.uniform(15, 96)))
    if k == 3:
        return A.median_filter(img, int(rs.choice([3, 5])))
    if k == 4:
        return A.rotation(img, rs.uniform(-35, 35))
    if k == 5:
        return A.scaling(img, rs.uniform(0.3, 0.9))
    if k == 6:
        return A.cropping(img, rs.uniform(0.02, 0.3))
    if k == 7:
        return A.histogram_equalization(img)
    if k == 8:   # gamma / display-window change
        return (255 * (img / 255.0) ** rs.uniform(0.6, 1.6)).astype(np.uint8)
    if k == 9:   # Gaussian blur
        from scipy.ndimage import gaussian_filter
        return np.clip(gaussian_filter(img.astype(np.float32), rs.uniform(0.5, 2.0)), 0, 255).astype(np.uint8)
    if k == 10:  # small translation
        dy, dx = rs.randint(-8, 9, 2)
        return np.roll(img, (dy, dx), (0, 1))
    if k == 11:
        return A.telemedicine_transcoding(img, 2)
    return img.copy()


def attack_chain(img, rs):
    out = rand_attack(img, rs)
    if rs.rand() < 0.35:
        out = rand_attack(out, rs)
    return out


def to_tensor(batch):
    return torch.from_numpy(np.stack([prepare(im) for im in batch]))[:, None]


# ---------------------------------------------------------------- losses
def vic_terms(u, v, groups=None):
    """u, v: (N, K) codes of originals and attacked copies."""
    n, K = u.shape
    inv = ((u - v) ** 2).mean()
    quant = (((u.abs() - 1) ** 2).mean() + ((v.abs() - 1) ** 2).mean()) / 2
    groups = groups or [torch.arange(n)]
    var = 0.0
    for gi in groups:
        for x in (u[gi], v[gi]):
            std = torch.sqrt(x.var(dim=0) + 1e-4)
            var = var + torch.relu(0.9 - std).mean() / (2 * len(groups))
    cov = 0.0
    for x in (u, v):
        xc = x - x.mean(0)
        c = (xc.t() @ xc) / max(n - 1, 1)
        cov = cov + ((c - torch.diag(torch.diag(c))) ** 2).sum() / K / 2
    return inv, quant, var, cov


def info_nce(u, v, T=0.1):
    un, vn = F.normalize(u, dim=1), F.normalize(v, dim=1)
    logits = un @ vn.t() / T
    tgt = torch.arange(len(u))
    return (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt)) / 2


def total_loss(net, xa, xb, groups):
    ga, la = net(xa)
    gb, lb = net(xb)
    u, v = torch.tanh(1.5 * ga), torch.tanh(1.5 * gb)
    inv, quant, var, cov = vic_terms(u, v, groups)
    nce = info_nce(u, v)
    lu, lv = torch.tanh(1.5 * la), torch.tanh(1.5 * lb)
    lu2 = lu.permute(0, 2, 3, 1).reshape(-1, lu.shape[1])
    lv2 = lv.permute(0, 2, 3, 1).reshape(-1, lv.shape[1])
    linv, lquant, lvar, lcov = vic_terms(lu2, lv2)
    L = (W["inv"] * (inv + linv) + W["quant"] * (quant + lquant) + W["var"] * (var + lvar)
         + W["cov"] * (cov + lcov) + W["nce"] * nce)
    return L, dict(inv=round(inv.item(), 5), quant=round(quant.item(), 4), var=round(float(var), 4),
                   cov=round(float(cov), 3), nce=round(nce.item(), 4), linv=round(linv.item(), 5),
                   lvar=round(float(lvar), 4))


# ---------------------------------------------------------------- validation
VAL_TESTS = {"JPEG50": lambda im: A.jpeg_compression(im, 50), "Gauss20": lambda im: A.gaussian_noise(im, 20),
             "Rot5": lambda im: A.rotation(im, 5), "Rot10": lambda im: A.rotation(im, 10),
             "SP1": lambda im: A.salt_and_pepper(im, 0.01), "HistEq": A.histogram_equalization,
             "Median3": lambda im: A.median_filter(im, 3)}


@torch.no_grad()
def validate(net, imgs, seed):
    """Global-bit agreement after fixed attacks and between different images."""
    net.eval()
    rs = np.random.RandomState(seed)
    s0 = net(to_tensor(imgs))[0] > 0
    out = {}
    for k, fn in VAL_TESTS.items():
        np.random.seed(rs.randint(1 << 30))
        s = net(to_tensor([fn(im) for im in imgs]))[0] > 0
        out[k] = round((s == s0).float().mean().item(), 4)
    agree = (s0[:, None, :] == s0[None, :, :]).float().mean(-1)
    off = ~torch.eye(len(imgs), dtype=torch.bool)
    out["diff_mean"] = round(agree[off].mean().item(), 4)
    out["diff_max"] = round(agree[off].max().item(), 4)
    net.train()
    return out


def main(npz, out_dir, n_steps=3000):
    torch.manual_seed(0)
    torch.set_num_threads(2)
    rs = np.random.RandomState(0)
    D = np.load(npz)
    brats, vol, big = D["brats"], D["vol_id"], D["big"]
    n_vol = int(D["n_volumes"])
    val_vols = set(range(n_vol - 8, n_vol))           # last 8 volumes: validation only
    by_vol = {}
    for i in range(len(brats)):
        if vol[i] not in val_vols:
            by_vol.setdefault(int(vol[i]), []).append(i)
    va_idx = [i for i in range(len(brats)) if vol[i] in val_vols]
    val_brats = [brats[i] for i in va_idx[::8]][:24]
    n_nih = 100                                          # big = 100 NIH images, then 100 ISIC images
    big_va_idx = list(range(n_nih - 10, n_nih)) + list(range(len(big) - 10, len(big)))
    big_tr_idx = [i for i in range(len(big)) if i not in big_va_idx]
    val_big = [big[i] for i in big_va_idx]

    net = AIFNet(32)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=n_steps, pct_start=0.1)
    log, t0 = [], time.time()
    os.makedirs(out_dir, exist_ok=True)
    for step in range(1, n_steps + 1):
        batch = []
        depth = rs.uniform(0, 1)       # same relative depth in all volumes: different patients, similar anatomy
        for v in rs.choice(list(by_vol), N_BRATS_VOLS, replace=False):
            ids = by_vol[v]
            i = int(np.clip(round(depth * (len(ids) - 1)) + rs.randint(-1, 2), 0, len(ids) - 1))
            j = min(len(ids) - 1, max(0, i + rs.choice([-1, 1]) * rs.randint(1, 6)))  # 3-15 slices apart
            batch += [brats[ids[i]], brats[ids[j]]]
        batch += [big[k] for k in rs.choice(big_tr_idx, N_BIG, replace=False)]
        groups = [torch.arange(2 * N_BRATS_VOLS), torch.arange(2 * N_BRATS_VOLS, len(batch))]
        att = [attack_chain(im, rs) for im in batch]
        L, parts = total_loss(net, to_tensor(batch), to_tensor(att), groups)
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            parts.update(step=step, loss=round(L.item(), 4), t=round(time.time() - t0))
            print(json.dumps(parts), flush=True)
        if step % 250 == 0 or step == n_steps:
            v = dict(step=step, brats=validate(net, val_brats, 1), big=validate(net, val_big, 2))
            print("VAL", json.dumps(v), flush=True)
            log.append(v)
            torch.save(dict(state_dict=net.state_dict(), width=32, step=step, loss_weights=W,
                            val=log), os.path.join(out_dir, "aifnet_v3.pt"))
    json.dump(log, open(os.path.join(out_dir, "training_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3000)
