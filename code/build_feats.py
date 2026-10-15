"""Label-free feature blocks, cached once per scene.
  {DS}_ms.npy          multi-scale mean / std of 15 principal components (165-d), standardised
  {DS}_spec.npy        the 15 standardised principal components of the centre pixel
  {name}_p{k}.npy      PCA-reduced, standardised copy of a dense foundation-model feature file
Usage: build_feats.py DS [featname ...] [k=128]"""
import sys, os, numpy as np, torch
from scipy.ndimage import uniform_filter
from common import *
ds = sys.argv[1]; names = [a for a in sys.argv[2:] if not a.isdigit()]; k = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 128
FD = f"{J}/results/feats"; os.makedirs(FD, exist_ok=True)


def ms_block(P, scales=(3, 5, 9, 13, 21)):
    f = [P]
    for s in scales:
        m = uniform_filter(P, size=(s, s, 1), mode="reflect"); f.append(m)
        f.append(np.sqrt(np.maximum(uniform_filter(P ** 2, size=(s, s, 1), mode="reflect") - m ** 2, 0)))
    F = np.concatenate(f, -1).reshape(-1, P.shape[-1] * (1 + 2 * len(scales)))
    return ((F - F.mean(0)) / (F.std(0) + 1e-6)).astype(np.float32)


if not os.path.exists(f"{FD}/{ds}_ms.npy"):
    X, gt = load(ds); P, _ = pca(X, 15, whiten=False); P = P / P[..., 0].std()
    np.save(f"{FD}/{ds}_ms.npy", ms_block(P))
    np.save(f"{FD}/{ds}_spec.npy", ((P - P.mean((0, 1))) / P.std((0, 1))).reshape(-1, 15).astype(np.float32))
    print("saved ms / spec", flush=True)


def reduce(D, k):
    """standardise columns, project on the top-k principal axes (covariance accumulated in chunks on the GPU)"""
    D = D.reshape(-1, D.shape[-1]); n, d = D.shape; dev = "cuda"
    s1 = torch.zeros(d, dtype=torch.float64, device=dev); s2 = torch.zeros(d, d, dtype=torch.float64, device=dev)
    for i in range(0, n, 50000):
        x = torch.tensor(D[i:i + 50000].astype(np.float32), device=dev).double(); s1 += x.sum(0); s2 += x.T @ x
    mu = s1 / n; C = s2 / n - torch.outer(mu, mu); sd = C.diag().clamp_min(1e-12).sqrt()
    R = C / torch.outer(sd, sd); w, V = torch.linalg.eigh(R); V = V[:, torch.argsort(w, descending=True)[:k]]
    out = np.empty((n, k), np.float32)
    for i in range(0, n, 50000):
        x = (torch.tensor(D[i:i + 50000].astype(np.float32), device=dev).double() - mu) / sd
        out[i:i + 50000] = (x @ V).float().cpu().numpy()
    return out / out.std(0).mean(), float(torch.sort(w, descending=True)[0][:k].sum() / w.sum())


for name in names:
    o = f"{FD}/{name}_p{k}.npy"
    if os.path.exists(o): continue
    F, ev = reduce(np.load(f"{FD}/{name}.npy", mmap_mode="r"), k)
    np.save(o, F); print("saved", o, F.shape, f"explained {ev:.3f}", flush=True)
