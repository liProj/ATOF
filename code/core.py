"""Building blocks of the proposed pipeline (all label-free except `fit_linear`).
Everything runs on the GPU on whole-scene tensors; one seed takes a few seconds."""
import numpy as np, torch, torch.nn.functional as F
from common import *

FD = f"{J}/results/feats"
import os
DEV = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
DINO_DEFAULT = "vit_base_u8_k3"


def load_blocks(ds, dino=DINO_DEFAULT, names=("ms", "dino")):
    """Each block is scaled to unit mean squared norm so that no block dominates the linear head."""
    out = {}
    for n in names:
        if n == "dino": v = np.load(f"{FD}/{ds}_{dino}_p128.npy")
        elif n in ("ms", "spec"): v = np.load(f"{FD}/{ds}_{n}.npy")
        else: v = np.load(f"{FD}/{ds}_{n}_p128.npy")
        v = torch.tensor(v, device=DEV)
        out[n] = v / v.pow(2).sum(1).mean().sqrt()
    return out


def fit_linear(Z, y, nc, C=100.0, w=None, iters=100, W0=None):
    """Multinomial logistic regression, objective  C * sum_i w_i CE_i + 0.5 ||W||^2  (bias not penalised). L-BFGS."""
    n, d = Z.shape; dev0 = Z.device
    if n < 20000:  # tiny problems are much faster on the CPU than on a shared GPU
        Z = Z.cpu().double(); y = y.cpu(); w = None if w is None else w.cpu().double(); W0 = None if W0 is None else W0.cpu().double()
    W = torch.zeros(d + 1, nc, device=Z.device, dtype=Z.dtype) if W0 is None else W0.clone()
    W.requires_grad_(True); w = torch.ones(n, device=Z.device, dtype=Z.dtype) if w is None else w
    opt = torch.optim.LBFGS([W], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-6)
    soft = y.dim() == 2

    def closure():
        opt.zero_grad(); lg = Z @ W[:-1] + W[-1]
        ce = -(y * F.log_softmax(lg, 1)).sum(1) if soft else F.cross_entropy(lg, y, reduction="none")
        loss = C * (w * ce).sum() + 0.5 * W[:-1].pow(2).sum(); loss.backward(); return loss
    opt.step(closure)
    return W.detach().float().to(dev0)


def predict_proba(Z, W, bs=200000):
    return torch.cat([F.softmax(Z[i:i + bs] @ W[:-1] + W[-1], 1) for i in range(0, len(Z), bs)])


def make_kernel(G, H, Wd, r=3, sigma=0.3):
    """Label-free lattice kernel: k_ij = exp((g_i.g_j - 1) / sigma) for j in the (2r+1)^2 window of i, row-normalised.
    G: (H*W, d) guidance features with unit-norm rows. Returns (n_off, H, W) and the offsets. Computed once per scene."""
    Gm = G.T.reshape(1, -1, H, Wd); offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    Gp = F.pad(Gm, (r, r, r, r), mode="reflect")
    K = torch.cat([torch.exp(((Gm * Gp[:, :, r + dy:r + dy + H, r + dx:r + dx + Wd]).sum(1) - 1) / sigma) for dy, dx in offs], 0)
    return K / K.sum(0, keepdim=True), offs, r


def diffuse(P, kern, iters=2, keep=0.0):
    """Feature-guided diffusion of class probabilities on the image lattice:
    p_i <- keep * p_i + (1 - keep) * sum_j k_ij p_j."""
    K, offs, r = kern; _, H, Wd = K.shape; nc = P.shape[1]; Pm = P.T.reshape(1, nc, H, Wd)
    for _ in range(iters):
        Pp = F.pad(Pm, (r, r, r, r), mode="reflect"); acc = torch.zeros_like(Pm)
        for k, (dy, dx) in enumerate(offs): acc += K[k] * Pp[:, :, r + dy:r + dy + H, r + dx:r + dx + Wd]
        Pm = keep * Pm + (1 - keep) * acc
    return Pm.reshape(nc, -1).T


def select_pseudo(P, nc, frac=0.2, exclude=None, prior=None):
    """Class-balanced confident pseudo-labels: for every predicted class keep its `frac` most confident pixels."""
    conf, lab = P.max(1); keep = torch.zeros(len(P), dtype=torch.bool, device=P.device)
    for c in range(nc):
        idx = torch.nonzero(lab == c).squeeze(1)
        if len(idx) == 0: continue
        k = max(1, int(frac * len(idx))); keep[idx[torch.topk(conf[idx], k).indices]] = True
    if exclude is not None: keep[exclude] = False
    return keep, lab, conf


def unit(v): return v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
