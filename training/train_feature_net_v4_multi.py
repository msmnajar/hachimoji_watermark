# training/train_feature_net_v4.py  (v4)
"""
Self-supervised training of AIF-Net v4 (models/feature_net_v4.py).

Changes from v3:
  * the code has 512 bits: 256 global and 4 per cell of an 8 x 8 grid, and the
    local bits are normalized per cell and channel (no position-only bits);
  * all loss terms (invariance, quantization, per-domain variance, covariance,
    InfoNCE) act on the full 512-bit code, so every local bit must vary across
    images and help to identify each image among its hard negatives;
  * InfoNCE weight 1.0 (v3: 0.5) and 8 BraTS volumes per batch (v3: 6).

Every mini-batch mixes 16 BraTS slices and 6 chest-radiograph / dermoscopy
images. The BraTS slices come from 8 volumes at the same relative depth
(different patients, similar anatomy) and, for each volume, a second slice
3-15 slices away (same patient, nearby anatomy). For every image x a random
attack chain A (one or two operations from rand_attack) gives a positive copy.

Usage: python training/train_feature_net_v4.py <train_set.npz> <out_dir> [n_steps]
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feature_net_v4 import AIFNet, prepare, N_GLOBAL
from evaluation import attacks as A

W = dict(inv=10.0, quant=1.0, var=2.0, cov=0.04, nce=1.0)
N_BRATS_VOLS, N_BIG = 8, 6


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
    u = torch.tanh(1.5 * torch.cat([ga, la], 1))
    v = torch.tanh(1.5 * torch.cat([gb, lb], 1))
    inv, quant, var, cov = vic_terms(u, v, groups)
    nce = info_nce(u, v)
    L = W["inv"] * inv + W["quant"] * quant + W["var"] * var + W["cov"] * cov + W["nce"] * nce
    G = N_GLOBAL
    return L, dict(inv=round(inv.item(), 5), inv_g=round(((u[:, :G] - v[:, :G]) ** 2).mean().item(), 5),
                   inv_l=round(((u[:, G:] - v[:, G:]) ** 2).mean().item(), 5), quant=round(quant.item(), 4),
                   var=round(float(var), 4), cov=round(float(cov), 3), nce=round(nce.item(), 4))


# ---------------------------------------------------------------- validation
VAL_TESTS = {"JPEG50": lambda im: A.jpeg_compression(im, 50), "Gauss20": lambda im: A.gaussian_noise(im, 20),
             "Rot5": lambda im: A.rotation(im, 5), "Rot10": lambda im: A.rotation(im, 10),
             "SP1": lambda im: A.salt_and_pepper(im, 0.01), "HistEq": A.histogram_equalization,
             "Median3": lambda im: A.median_filter(im, 3)}


@torch.no_grad()
def validate(net, imgs, seed):
    """Bit agreement after fixed attacks and between different images, for the
    global and the local code separately; lstd = mean std of local logits across images."""
    net.eval()
    rs = np.random.RandomState(seed)
    g0, l0 = net(to_tensor(imgs))
    out = {"lstd": round(l0.std(0).mean().item(), 3), "gstd": round(g0.std(0).mean().item(), 3)}
    for k, fn in VAL_TESTS.items():
        np.random.seed(rs.randint(1 << 30))
        g, l = net(to_tensor([fn(im) for im in imgs]))
        out[k] = (round(((g > 0) == (g0 > 0)).float().mean().item(), 4), round(((l > 0) == (l0 > 0)).float().mean().item(), 4))
    off = ~torch.eye(len(imgs), dtype=torch.bool)
    for nm, s0 in (("g", g0 > 0), ("l", l0 > 0)):
        agree = (s0[:, None, :] == s0[None, :, :]).float().mean(-1)
        out[f"diff_{nm}"] = (round(agree[off].mean().item(), 4), round(agree[off].max().item(), 4))
    net.train()
    return out


def main(npz, out_dir, n_steps=3000, seed=0, hard_neg=True):
    """seed: torch and numpy seed (paper model: 0). hard_neg=False: ablation in which the
    16 BraTS slices of a batch are drawn uniformly at random from the training pool
    (random volumes and depths), i.e. without same-depth and neighbouring-slice negatives."""
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rs = np.random.RandomState(seed)
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
    start = 1
    ck = os.path.join(out_dir, "resume.pt")
    if os.path.exists(ck):                         # resume: model, optimizer, scheduler and all RNG states
        R = torch.load(ck, weights_only=False)
        net.load_state_dict(R["net"]); opt.load_state_dict(R["opt"]); sched.load_state_dict(R["sched"])
        rs.set_state(R["rs"]); torch.set_rng_state(R["torch_rng"]); np.random.set_state(R["np_global"])
        log, start = R["log"], R["step"] + 1
        print("RESUME", start, flush=True)
    for step in range(start, n_steps + 1):
        batch = []
        if hard_neg:
            depth = rs.uniform(0, 1)       # same relative depth in all volumes: different patients, similar anatomy
            for v in rs.choice(list(by_vol), N_BRATS_VOLS, replace=False):
                ids = by_vol[v]
                i = int(np.clip(round(depth * (len(ids) - 1)) + rs.randint(-1, 2), 0, len(ids) - 1))
                j = min(len(ids) - 1, max(0, i + rs.choice([-1, 1]) * rs.randint(1, 6)))  # 3-15 slices apart
                batch += [brats[ids[i]], brats[ids[j]]]
        else:
            pool = [i for ids in by_vol.values() for i in ids]
            batch += [brats[i] for i in rs.choice(pool, 2 * N_BRATS_VOLS, replace=False)]
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
            torch.save(dict(state_dict=net.state_dict(), width=32, step=step, loss_weights=W, version="v4",
                            val=log), os.path.join(out_dir, "aifnet_v4.pt"))
        if step % 50 == 0:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), rs=rs.get_state(),
                            torch_rng=torch.get_rng_state(), np_global=np.random.get_state(), log=log, step=step),
                       ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    json.dump(log, open(os.path.join(out_dir, "training_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3000,
         int(sys.argv[4]) if len(sys.argv) > 4 else 0, (sys.argv[5] != "nohn") if len(sys.argv) > 5 else True)
