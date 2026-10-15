"""Token-guided spectral pooling (label-free). For every pixel the principal-component spectra of its neighbours are
averaged with weights given by the similarity of the frozen foundation-model tokens:
    k_ij = exp((t_i . t_j - 1) / sigma),  j in a (2r+1)^2 window,
so that the pooling region follows objects instead of a square box. Mean and standard deviation at several radii.
Output: results/feats/{DS}_{out}.npy
Usage: build_guided.py DS guidance_spec out [sigma=0.2] [npc=15]      e.g.  build_guided.py PU tok:vit_base:u4:c0-2:l12 gms"""
import sys, os, numpy as np, torch, torch.nn.functional as F
from method import *
ds, gspec, outn = sys.argv[1], sys.argv[2], sys.argv[3]; sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2; npc = int(sys.argv[5]) if len(sys.argv) > 5 else 15
tag = f"{ds}_{outn}"
if os.path.exists(f"{FD}/{tag}.npy"): sys.exit(0)
X, gt = load(ds); H, W, _ = X.shape
P, _ = pca(X, npc, whiten=False); P = P / P[..., 0].std()
Pm = torch.tensor(P, device=DEV).permute(2, 0, 1)[None]                      # (1, npc, H, W)
T = unit(torch.cat(Bank(ds, gspec).blocks, 1)).T.reshape(1, -1, H, W)   # guidance tokens, unit norm per pixel
R = 6; Tp = F.pad(T, (R, R, R, R), mode="reflect"); Pp = F.pad(Pm, (R, R, R, R), mode="reflect")
feats = [Pm[0]]; radii = (1, 2, 4, 6)
acc = {r: [torch.zeros_like(Pm[0]), torch.zeros_like(Pm[0]), torch.zeros(1, H, W, device=DEV)] for r in radii}
for dy in range(-R, R + 1):
    for dx in range(-R, R + 1):
        k = torch.exp(((T * Tp[:, :, R + dy:R + dy + H, R + dx:R + dx + W]).sum(1) - 1) / sigma)   # (1, H, W)
        q = Pp[0, :, R + dy:R + dy + H, R + dx:R + dx + W]
        for r in radii:
            if max(abs(dy), abs(dx)) <= r: acc[r][0] += k * q; acc[r][1] += k * q * q; acc[r][2] += k
for r in radii:
    m = acc[r][0] / acc[r][2]; feats += [m, (acc[r][1] / acc[r][2] - m * m).clamp_min(0).sqrt()]
Fm = torch.cat(feats, 0).reshape(-1, H * W).T
Fm = ((Fm - Fm.mean(0)) / (Fm.std(0) + 1e-6)).cpu().numpy().astype(np.float32)
np.save(f"{FD}/{tag}.npy", Fm); print("saved", tag, Fm.shape)
