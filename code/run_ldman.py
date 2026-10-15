"""Re-run of the authors' released LDMAN network under the article's protocol.
Stated in the article: patch 13x13, batch 64, cross-entropy, Adam lr 5e-4, no weight decay, no augmentation,
1%/1%/98% (HUS) and 0.2%/0.2%/99.6% (PU) random split, five runs.
Not stated (our choices, see data_check.md): 30 principal components with whitening (the released ldman() builder
hard-codes n_bands=30 for both scenes), reflect padding, number of epochs, checkpoint = best validation OA.
Usage: run_ldman.py DS SEED [EPOCHS] [TAG]"""
import sys, os, time, json, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
from common import *
sys.path.insert(0, f"{J}/third_party/LDMAN"); import LDMAN

ds, seed = sys.argv[1], int(sys.argv[2]); EP = int(sys.argv[3]) if len(sys.argv) > 3 else 200
tag = sys.argv[4] if len(sys.argv) > 4 else f"ldman_ep{EP}"
out = f"{J}/results/{tag}/{ds}_s{seed}.json"; os.makedirs(os.path.dirname(out), exist_ok=True)
if os.path.exists(out): sys.exit(0)
dev = "cuda"; torch.manual_seed(seed); np.random.seed(seed)
X, gt = load(ds); nc = int(gt.max()); H, W, _ = X.shape
PRE = os.environ.get("PRE", "whiten")
if PRE == "whiten": P, _ = pca(X, 30)
elif PRE == "nowhiten": P, _ = pca(X, 30, whiten=False); P = P / P.std()
elif PRE == "std":  # band-wise standardisation, then PCA without whitening
    Xs = (X - X.mean((0, 1))) / X.std((0, 1)); P, _ = pca(Xs, 30, whiten=False)
elif PRE == "minmax":  # global min-max to [0,1], then PCA without whitening
    Xs = (X - X.min()) / (X.max() - X.min()); P, _ = pca(Xs, 30, whiten=False)
r = 6
Pp = torch.tensor(np.pad(P, ((r, r), (r, r), (0, 0)), mode="reflect")).permute(2, 0, 1).to(dev)  # (30, H+12, W+12)
tr, va, te = get_split(ds, gt, seed, os.environ.get('SPLIT', 'table1')); y = torch.tensor(gt.ravel() - 1).to(dev)

def patches(idx):
    i, j = np.divmod(idx, W)
    return torch.stack([Pp[:, a:a + 13, b:b + 13] for a, b in zip(i, j)]).unsqueeze(1)  # (n, 1, 30, 13, 13)

def predict(model, idx, bs=2048):
    model.eval(); o = []
    with torch.no_grad():
        for k in range(0, len(idx), bs): o.append(model(patches(idx[k:k + bs])).argmax(1))
    return torch.cat(o).cpu().numpy()

model = LDMAN.ldman(ds.lower(), 13).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=5e-4); ce = nn.CrossEntropyLoss()
Xtr, ytr = patches(tr), y[torch.tensor(tr, device=dev)]; best, best_ep, state = -1, 0, None; t0 = time.time(); curve = []
for ep in range(EP):
    model.train(); perm = torch.randperm(len(tr), device=dev)
    for k in range(0, len(tr), 64):
        b = perm[k:k + 64]
        if len(b) < 2: continue
        opt.zero_grad(); ce(model(Xtr[b]), ytr[b]).backward(); opt.step()
    if ep % 5 == 4 or ep == EP - 1:
        acc = float((predict(model, va) == (gt.ravel()[va] - 1)).mean()); curve.append((ep + 1, acc))
        if acc >= best: best, best_ep, state = acc, ep + 1, {k: v.clone() for k, v in model.state_dict().items()}
ttrain = time.time() - t0
last = metrics(gt.ravel()[te] - 1, predict(model, te), nc)
model.load_state_dict(state); t0 = time.time(); pte = predict(model, te); ttest = time.time() - t0
m = metrics(gt.ravel()[te] - 1, pte, nc)
m.update(val=best, best_ep=best_ep, train_s=ttrain, test_s=ttest, last=dict(OA=last["OA"], AA=last["AA"], K=last["K"]),
         curve=curve, n_params=sum(p.numel() for p in model.parameters()))
if os.environ.get("MAP") == "1":   # whole-scene label map for the figures
    os.makedirs(f"{J}/results/maps", exist_ok=True); np.save(f"{J}/results/maps/ldman_{ds}_s{seed}.npy", predict(model, np.arange(H * W)).reshape(H, W).astype(np.uint8))
json.dump(m, open(out, "w")); print(ds, seed, EP, f"OA {m['OA']:.2f} AA {m['AA']:.2f} K {m['K']:.2f} | last OA {last['OA']:.2f} | best_ep {best_ep} | {ttrain:.0f}s")
