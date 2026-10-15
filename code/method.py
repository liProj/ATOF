"""Proposed classifier: a bank of label-free feature blocks, alignment-tempered block fusion, convex linear head.

Blocks (all computed once per scene without labels, see feat_v2.py / build_feats.py / build_guided.py):
  ms    multi-scale box mean / std of 15 principal components
  gms   token-guided spectral pooling (pooling weights from foundation-model token similarity)
  tok   frozen DINOv3 patch tokens, one block per (false-colour composite, layer, up-sampling factor)

Fusion: every block b gets the weight  w_b = (a_b / max_b a_b) ** gamma, where a_b is the centred kernel-target
alignment of the block's linear kernel with the training labels. gamma = 0 is plain concatenation; larger gamma
concentrates the head on the blocks that agree with the few labels. The exponent gamma is
chosen by cross-validated likelihood on the training + validation pixels (the reference method uses the same
validation pixels for checkpoint selection); the head itself is fitted on the training pixels only.
"""
import os, numpy as np, torch
from core import *

GAMMAS = (0.0, 1.0, 2.0, 4.0, 8.0)
C_DEFAULT = 100.0
BALANCED = os.environ.get('BALANCED', '1') == '1'


def cw(y, nc):
    """Class-balanced sample weights  n / (n_classes_present * n_c)  (mean 1)."""
    if not BALANCED: return None
    cnt = torch.bincount(y, minlength=nc).float(); return (len(y) / ((cnt > 0).sum() * cnt.clamp_min(1)))[y]
NFOLD = 5


def expand(item):
    """'tok:<short>:u<UP>:c<a>-<b>:l<l1,l2>[:tta]' -> block names; anything else is a plain block name."""
    if not item.startswith("tok:"): return [item]
    p = item.split(":"); short, up, cs, ls = p[1:5]; suf = "_tta" if len(p) > 5 and p[5] == "tta" else ""
    a, b = cs[1:].split("-") if "-" in cs else (cs[1:], cs[1:])
    return [f"{short}_{up}_c{k}_l{l}{suf}" for k in range(int(a), int(b) + 1) for l in ls[1:].split(",")]


def block_names(spec): return [n for it in spec.split("+") for n in expand(it)]


class Bank:
    """Feature blocks restricted to an index set (default: labelled pixels); each block has unit mean squared norm,
    with the scale estimated on the whole scene (label-free)."""

    def __init__(self, ds, spec, index=None, dev=DEV):
        self.ds, self.spec, self.names, self.dev = ds, spec, block_names(spec), dev
        self.blocks = []
        for n in self.names:
            p = f"{FD}/{ds}_{n}.npy"; p = p if os.path.exists(p) else f"{FD}/{ds}_{n}_p128.npy"
            v = np.load(p, mmap_mode="r"); sc = np.sqrt((np.asarray(v[::7]).astype(np.float64) ** 2).sum(1).mean())
            a = np.asarray(v) if index is None else np.asarray(v[index])
            self.blocks.append(torch.tensor(a / sc, device=dev, dtype=torch.float32))
        self.dims = [b.shape[1] for b in self.blocks]

    def alignment(self, tr, y, nc):
        """Centred kernel-target alignment of each block on the training pixels."""
        Y = torch.nn.functional.one_hot(y[tr], nc).float(); Y = Y - Y.mean(0); Ky = Y @ Y.T; al = []
        for b in self.blocks:
            z = b[tr] - b[tr].mean(0); Kb = z @ z.T
            al.append(((Kb * Ky).sum() / (Kb.norm() * Ky.norm())).clamp_min(1e-4))
        return torch.stack(al)

    def fused(self, al, gamma, rows=None):
        w = (al / al.max()) ** gamma if gamma > 0 else torch.ones_like(al)
        w = w * (len(w) / (w ** 2).sum()).sqrt()          # keep the total energy of the concatenation constant
        return torch.cat([(b if rows is None else b[rows]) * wi for b, wi in zip(self.blocks, w)], 1), w


def cv_nll(bank, pool, y, nc, gamma, C, seed=0, nfold=NFOLD):
    """Cross-validated negative log-likelihood on the labelled pool (alignment is re-estimated inside every fold)."""
    fold = torch.tensor(np.random.RandomState(seed).permutation(len(pool)) % nfold, device=pool.device); nll = 0.0; acc = 0.0
    for f in range(nfold):
        itr, ite = pool[fold != f], pool[fold == f]; al = bank.alignment(itr, y, nc)
        Ztr, _ = bank.fused(al, gamma, itr); Zte, _ = bank.fused(al, gamma, ite)
        P = predict_proba(Zte, fit_linear(Ztr, y[itr], nc, C=C, w=cw(y[itr], nc)))
        nll -= float(torch.log(P[torch.arange(len(ite)), y[ite]].clamp_min(1e-12)).sum()); acc += float((P.argmax(1) == y[ite]).sum())
    return nll / len(pool), acc / len(pool)


def fit_predict(bank, tr, va, y, nc, gammas=GAMMAS, C=C_DEFAULT, seed=0, rule="cv_nll", return_proba=False):
    """tr, va: index tensors into the bank rows; y: labels (0..nc-1) for all bank rows (only tr / va are read).
    gamma is chosen by 5-fold cross-validated NLL on train + validation pixels (rule='cv_nll') or by validation
    accuracy (rule='val_acc'); the final head is always fitted on the training pixels only.
    Returns hard predictions for every bank row, the selected gamma, block weights and the selection table."""
    table = {}
    if len(gammas) == 1: g = gammas[0]
    elif rule == "cv_nll":
        pool = torch.cat([tr, va])
        for gm in gammas: table[gm] = cv_nll(bank, pool, y, nc, gm, C, seed)[0]
        g = min(gammas, key=lambda k: table[k])
    else:
        al = bank.alignment(tr, y, nc)
        for gm in gammas:
            P = predict_proba(bank.fused(al, gm, va)[0], fit_linear(bank.fused(al, gm, tr)[0], y[tr], nc, C=C, w=cw(y[tr], nc))); table[gm] = -float((P.argmax(1) == y[va]).float().mean())
        g = min(gammas, key=lambda k: table[k])
    al = bank.alignment(tr, y, nc); Z, w = bank.fused(al, g); W = fit_linear(Z[tr], y[tr], nc, C=C, w=cw(y[tr], nc)); P = predict_proba(Z, W)
    out = dict(pred=P.argmax(1), gamma=g, C=C, val=float((P[va].argmax(1) == y[va]).float().mean()), weights=w.cpu().numpy(),
               align=al.cpu().numpy(), table=table)
    if return_proba: out["proba"] = P
    return out
