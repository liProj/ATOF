"""Runner for the proposed classifier and its ablations (CPU, one JSON per seed).
Usage: run_ours.py DS TAG SPEC [--seeds 10-19] [--split table1|n<k>|official] [--gammas 0,1,2,4,8] [--rule cv_nll|val_acc]
                   [--C 100] [--head linear|ncm|svm] [--post none|diffuse|selftrain|both] [--balanced 1|0]"""
import sys, os, json, time, argparse, numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("tag"); ap.add_argument("spec")
ap.add_argument("--seeds", default="10-19"); ap.add_argument("--split", default="table1")
ap.add_argument("--gammas", default="0,1,2,4,8"); ap.add_argument("--rule", default="cv_nll"); ap.add_argument("--C", type=float, default=100.0)
ap.add_argument("--head", default="linear"); ap.add_argument("--post", default="none"); ap.add_argument("--balanced", default="1")
a = ap.parse_args(); os.environ["BALANCED"] = a.balanced
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import method
from method import *

ds = a.ds; lo, hi = (a.seeds.split("-") + [a.seeds])[:2]; seeds = range(int(lo), int(hi) + 1)
gammas = tuple(float(g) for g in a.gammas.split(","))
outdir = f"{J}/results/ours/{a.tag}"; os.makedirs(outdir, exist_ok=True)
todo = [s for s in seeds if not os.path.exists(f"{outdir}/{ds}_s{s}.json")]
if not todo: sys.exit(0)
X, gt = load(ds); H, W, _ = X.shape; nc = int(gt.max()); g0 = gt.ravel()
whole = a.post != "none"                                  # post-processing needs the whole lattice
L = np.arange(g0.size) if whole else np.flatnonzero(g0 > 0)
pos = np.full(g0.size, -1); pos[L] = np.arange(len(L))
y = torch.tensor(np.maximum(g0[L] - 1, 0), device=DEV)
t0 = time.time(); bank = Bank(ds, a.spec, None if whole else L); t_bank = time.time() - t0
kern = None
if whole:
    G = unit(torch.cat(bank.blocks, 1)); kern = make_kernel(G, H, W, r=3, sigma=0.3); del G

for seed in todo:
    tr, va, te = [torch.tensor(pos[i], device=DEV) for i in get_split(ds, gt, seed, a.split)]
    t0 = time.time(); extra = {}
    if a.head == "linear":
        o = fit_predict(bank, tr, va, y, nc, gammas=gammas, C=a.C, seed=seed, rule=a.rule, return_proba=whole)
        pred = o["pred"]; extra = dict(gamma=o["gamma"], weights=o["weights"].tolist(), align=o["align"].tolist(), val=o["val"],
                                       table={str(k): v for k, v in o["table"].items()})
        if whole:
            P = o["proba"]; al = torch.tensor(o["align"], device=DEV)
            if a.post in ("selftrain", "both"):       # one round of class-balanced confident pseudo-labels, then refit
                keep, lab, conf = select_pseudo(P, nc, frac=0.3, exclude=tr); idx = torch.nonzero(keep).squeeze(1)
                if len(idx) > 20000: idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))[:20000]]
                Z, _ = bank.fused(al, o["gamma"]); Zs = torch.cat([Z[tr], Z[idx]]); ys = torch.cat([y[tr], lab[idx]])
                ws = torch.cat([torch.ones(len(tr)), torch.full((len(idx),), len(tr) / len(idx))])
                P = predict_proba(Z, fit_linear(Zs, ys, nc, C=a.C, w=ws, iters=60)); del Z
            if a.post in ("diffuse", "both"): P = diffuse(P, kern, iters=2)
            pred = P.argmax(1)
    else:
        al = bank.alignment(tr, y, nc); Z, _ = bank.fused(al, gammas[0])
        if a.head == "ncm":
            mu = torch.stack([Z[tr][y[tr] == c].mean(0) for c in range(nc)]); pred = torch.cdist(Z, mu).argmin(1)
        elif a.head == "svm":
            from sklearn.svm import SVC
            pred = torch.tensor(SVC(C=10, gamma="scale", class_weight="balanced").fit(Z[tr].numpy(), y[tr].numpy()).predict(Z[te].numpy()))
            full = torch.zeros(len(Z), dtype=torch.long); full[te] = pred; pred = full
    dt = time.time() - t0; p = pred.cpu().numpy(); ten = te.cpu().numpy(); yy = y.cpu().numpy()
    m = metrics(yy[ten], p[ten], nc); m.update(extra); m.update(fit_s=dt, bank_s=t_bank, names=bank.names, dims=bank.dims, n_train=len(tr), spec=a.spec, args=vars(a))
    json.dump(m, open(f"{outdir}/{ds}_s{seed}.json", "w"))
    print(a.tag, ds, seed, f"OA {m['OA']:.2f} AA {m['AA']:.2f} K {m['K']:.2f} gamma {extra.get('gamma')} {dt:.1f}s", flush=True)
