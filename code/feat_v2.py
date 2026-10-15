"""Dense frozen foundation-model tokens for a hyperspectral scene, one small block per (composite, layer).

The cube is reduced to principal components; consecutive PC triplets are rendered as false-colour images,
up-sampled by UP, tiled (512 px tiles, stride 384) and passed through a frozen DINOv3 ViT. Patch tokens of the
requested layers are averaged over overlapping tiles (and, with TTA=1, over the four flips), resampled to the
pixel grid, and reduced to Q principal directions (label-free). No label is used anywhere.

Output: results/feats/{DS}_{short}_u{UP}_c{k}_l{layer}[_tta].npy   (H*W, Q) float32
Usage: feat_v2.py DS arch UP K layers(comma, 1-based) [Q=64]        env TTA=1 enables flip averaging"""
import sys, os, time, numpy as np, torch, timm, torch.nn.functional as F
from common import *

ds, arch, UP, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
layers = [int(a) for a in sys.argv[5].split(",")]; Q = int(sys.argv[6]) if len(sys.argv) > 6 else 64
TTA = os.environ.get("TTA", "0") == "1"
short = arch.split(".")[0].replace("_patch16", "").replace("_dinov3", "") + ("_sat" if "sat" in arch else "")
FDIR = f"{J}/results/feats"; os.makedirs(FDIR, exist_ok=True)
name = lambda k, l: f"{FDIR}/{ds}_{short}_u{UP}_c{k}_l{l}{'_tta' if TTA else ''}" + ("" if Q == 64 else f"_q{Q}") + ".npy"
todo = [(k, l) for k in range(K) for l in layers if not os.path.exists(name(k, l))]
if not todo: sys.exit(0)
dev = "cuda"; X, gt = load(ds); H, W, _ = X.shape
P, _ = pca(X, 3 * K, whiten=False)
FP32 = os.environ.get("FP32", "0") == "1"   # ViT-L with the LVD weights overflows in half precision
model = timm.create_model(arch, pretrained=True, dynamic_img_size=True).eval().to(dev)
if not FP32: model = model.half()
cfg = model.pretrained_cfg
mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
T, S, ps = 512, 384, 16; D = model.num_features; t0 = time.time()


def reduce(f, q):
    """f: (D, H, W) on GPU -> (H*W, q) standardised principal projections"""
    x = f.reshape(f.shape[0], -1).T.double(); mu = x.mean(0); x = x - mu; sd = x.std(0).clamp_min(1e-8); x = x / sd
    C = x.T @ x / len(x)
    try: w, V = torch.linalg.eigh(C)
    except Exception:      # the GPU solver occasionally fails on near-singular covariances; redo on the CPU
        wn, Vn = np.linalg.eigh(C.cpu().numpy() + 1e-10 * np.eye(C.shape[0])); w, V = torch.tensor(wn, device=x.device), torch.tensor(Vn, device=x.device)
    o = torch.argsort(w, descending=True)[:q]
    z = (x @ V[:, o]).float(); return (z / z.std(0).mean()).cpu().numpy(), float(w[o].sum() / w.sum())


for k in sorted({k for k, _ in todo}):
    img = P[..., 3 * k:3 * k + 3].copy()
    for c in range(3):
        lo, hi = np.percentile(img[..., c], [1, 99]); img[..., c] = np.clip((img[..., c] - lo) / (hi - lo + 1e-8), 0, 1)
    im0 = F.interpolate(torch.tensor(img).permute(2, 0, 1)[None].to(dev), scale_factor=UP, mode="bicubic", align_corners=False).clamp(0, 1)
    Hu, Wu = im0.shape[2:]
    sums = {l: torch.zeros(D, H, W, device=dev) for l in layers}
    flips = [(), (2,), (3,), (2, 3)] if TTA else [()]
    for fl in flips:
        im = torch.flip(im0, fl) if fl else im0
        im = F.pad(im, (0, (-Wu) % ps, 0, (-Hu) % ps), mode="reflect"); Hp, Wp = im.shape[2:]
        acc = {l: torch.zeros(D, Hp // ps, Wp // ps, device=dev) for l in layers}; cnt = torch.zeros(1, Hp // ps, Wp // ps, device=dev)
        ys = sorted(set(list(range(0, max(Hp - T, 0) + 1, S)) + [max(Hp - T, 0)])); xs = sorted(set(list(range(0, max(Wp - T, 0) + 1, S)) + [max(Wp - T, 0)]))
        tiles = [(y, x) for y in ys for x in xs]
        with torch.no_grad():
            for b in range(0, len(tiles), 8):
                batch = torch.cat([im[:, :, y:y + T, x:x + T] for y, x in tiles[b:b + 8]])
                outs = model.forward_intermediates(((batch - mean) / std) if FP32 else ((batch - mean) / std).half(), indices=[l - 1 for l in layers], norm=True, output_fmt="NCHW", intermediates_only=True)
                hh, ww = outs[0].shape[2:]
                for l, o in zip(layers, outs):
                    for (y, x), t in zip(tiles[b:b + 8], o.float()): acc[l][:, y // ps:y // ps + hh, x // ps:x // ps + ww] += t
                for (y, x) in tiles[b:b + 8]: cnt[:, y // ps:y // ps + hh, x // ps:x // ps + ww] += 1
        for l in layers:
            f = F.interpolate((acc[l] / cnt)[None], size=(-(-Hp // UP), -(-Wp // UP)), mode="bilinear", align_corners=False)[0, :, :H, :W]
            sums[l] += torch.flip(f, [d - 1 for d in fl]) if fl else f
        del acc
    for l in layers:
        if os.path.exists(name(k, l)): continue
        z, ev = reduce(sums[l] / len(flips), Q)
        if not np.isfinite(z).all(): raise SystemExit(f"non-finite tokens for {ds} {short} composite {k} layer {l}: rerun with FP32=1")
        np.save(name(k, l), z)
        print(ds, short, f"u{UP} composite {k} layer {l} explained {ev:.3f} {time.time() - t0:.0f}s", flush=True)
    del sums; torch.cuda.empty_cache()
