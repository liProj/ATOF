"""Shared data handling for the GRSL round (target: LDMAN, doi 10.1109/LGRS.2026.3740946)."""
import os, sys, json, numpy as np, scipy.io as sio
from pathlib import Path
J = str(Path(os.environ.get("ATOF_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve())
# Table I of the article: per-class training / validation counts (testing = the rest).
TABLE1 = {
    "HUS": dict(train=[13, 12, 7, 12, 13, 3, 12, 13, 12, 12, 12, 12, 5, 5, 7],
                val=[12, 13, 7, 13, 12, 3, 13, 12, 13, 12, 13, 13, 4, 4, 6],
                test=[1226, 1229, 683, 1219, 1217, 319, 1243, 1219, 1227, 1203, 1210, 1208, 460, 419, 647]),
    "PU": dict(train=[13, 37, 4, 6, 3, 10, 3, 7, 2], val=[14, 38, 4, 6, 2, 10, 2, 8, 2],
               test=[6604, 18574, 2091, 3052, 1340, 5009, 1325, 3667, 943]),
}
# Published LDMAN results (Tables II, III): OA, AA, Kappa x100 (mean, sd) and per-class accuracy.
PUB = {
    "HUS": dict(OA=(91.89, 0.89), AA=(92.12, 0.71), K=(91.24, 0.96),
                cls=[92.60, 95.03, 98.88, 93.16, 99.50, 81.52, 88.07, 71.00, 86.11, 96.07, 96.34, 93.73, 90.11, 99.66, 100.00]),
    "PU": dict(OA=(91.16, 0.86), AA=(88.10, 1.26), K=(88.24, 1.18),
               cls=[90.88, 97.46, 76.86, 82.18, 100.00, 87.45, 93.78, 77.34, 86.98]),
}
# Best published competitor per data set (for reference lines in figures).
PUB_OTHERS = {
    "HUS": {"SSEFN": (87.61, 88.07, 87.76), "DCTN": (87.66, 88.20, 86.66), "MASSFormer": (88.57, 89.22, 87.72),
            "MambaHSI": (88.16, 88.69, 87.21), "3DSSMamba": (87.70, 88.41, 86.71), "HSI-MFormer": (89.20, 89.06, 88.32),
            "DSNet": (84.45, 83.18, None)},
    "PU": {"DSNet": (81.90, None, None), "SSEFN": (88.56, 85.29, 86.07), "DCTN": (89.54, 85.13, 85.95),
           "MASSFormer": (88.66, 82.92, 84.82), "MambaHSI": (88.13, 80.94, 84.16), "3DSSMamba": (88.04, 81.74, 84.12),
           "HSI-MFormer": (88.74, 81.17, 84.93)},
}
NAMES = {
    "HUS": ["Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water", "Residential", "Commercial",
            "Road", "Highway", "Railway", "Parking lot 1", "Parking lot 2", "Tennis court", "Running track"],
    "PU": ["Asphalt", "Meadows", "Gravel", "Trees", "Metal sheets", "Bare soil", "Bitumen", "Bricks", "Shadows"],
}


def load(ds):
    """Returns cube (H, W, B) float32 and ground truth (H, W) int, 0 = unlabelled."""
    if ds == "PU":
        X = sio.loadmat(f"{J}/data/PU/PaviaU.mat")["paviaU"].astype(np.float32)
        gt = sio.loadmat(f"{J}/data/PU/PaviaU_gt.mat")["paviaU_gt"].astype(np.int64)
    else:
        m = sio.loadmat(f"{J}/data/HUS/dfc2013.mat")
        X = m["data_hs"].astype(np.float32)
        gt = m["train"].astype(np.int64) + m["test"].astype(np.int64)  # contest train + test maps are disjoint
    return X, gt


def official_houston():
    m = sio.loadmat(f"{J}/data/HUS/dfc2013.mat")
    return m["train"].astype(np.int64), m["test"].astype(np.int64)


def split(ds, gt, seed):
    """Random per-class split with exactly the counts of Table I. Returns flat pixel indices."""
    rng = np.random.RandomState(1000 + seed)
    g = gt.ravel(); tr, va, te = [], [], []
    for c, (ntr, nva) in enumerate(zip(TABLE1[ds]["train"], TABLE1[ds]["val"]), start=1):
        idx = rng.permutation(np.flatnonzero(g == c))
        tr.append(idx[:ntr]); va.append(idx[ntr:ntr + nva]); te.append(idx[ntr + nva:])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def split_n(gt, seed, n_per_class, n_val=None):
    """Fixed number of labelled pixels per class (label-budget study)."""
    rng = np.random.RandomState(5000 + seed)
    g = gt.ravel(); tr, va, te = [], [], []
    nv = n_per_class if n_val is None else n_val
    for c in range(1, g.max() + 1):
        idx = rng.permutation(np.flatnonzero(g == c))
        tr.append(idx[:n_per_class]); va.append(idx[n_per_class:n_per_class + nv]); te.append(idx[n_per_class + nv:])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def pca(X, k, whiten=True):
    H, W, B = X.shape
    Z = X.reshape(-1, B).astype(np.float64); Z = Z - Z.mean(0)
    C = Z.T @ Z / (len(Z) - 1)
    w, V = np.linalg.eigh(C); o = np.argsort(w)[::-1][:k]
    P = Z @ V[:, o]
    if whiten: P = P / np.sqrt(w[o])
    return P.reshape(H, W, k).astype(np.float32), w[o] / w.sum()


def metrics(y, p, nc):
    """y, p in 0..nc-1. Returns OA, AA, Kappa (x100) and per-class accuracy."""
    cm = np.zeros((nc, nc), np.int64); np.add.at(cm, (y, p), 1)
    n = cm.sum(); oa = np.trace(cm) / n
    pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return dict(OA=100 * oa, AA=100 * pc.mean(), K=100 * (oa - pe) / (1 - pe), cls=(100 * pc).tolist(), cm=cm.tolist())


def get_split(ds, gt, seed, mode="table1"):
    """mode: 'table1' (article protocol) | 'n<k>' (k labelled pixels per class, k for validation) |
    'official' (Houston contest: spatially disjoint train / test maps; 10% of the train map held out for validation)."""
    if mode == "table1": return split(ds, gt, seed)
    if mode.startswith("n"): return split_n(gt, seed, int(mode[1:]))
    if mode == "official":
        assert ds == "HUS"; trm, tem = official_houston(); rng = np.random.RandomState(9000 + seed); tr, va = [], []
        for c in range(1, 16):
            idx = rng.permutation(np.flatnonzero(trm.ravel() == c)); nv = max(1, len(idx) // 10); va.append(idx[:nv]); tr.append(idx[nv:])
        return np.concatenate(tr), np.concatenate(va), np.flatnonzero(tem.ravel() > 0)
    raise ValueError(mode)
