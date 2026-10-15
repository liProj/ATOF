"""Whole-scene label map of the proposed classifier for one seed (CPU). Usage: map_ours.py DS SEED SPEC"""
import sys, os, numpy as np, torch
os.environ.setdefault("DEV", "cpu")
torch.set_num_threads(4)
from method import *
ds, seed, spec = sys.argv[1], int(sys.argv[2]), sys.argv[3]
X, gt = load(ds); H, W, _ = X.shape; nc = int(gt.max()); g0 = gt.ravel(); y = torch.tensor(np.maximum(g0 - 1, 0))
bank = Bank(ds, spec); tr, va, te = [torch.tensor(i) for i in get_split(ds, gt, seed)]
o = fit_predict(bank, tr, va, y, nc, seed=seed, return_proba=True)
os.makedirs(f"{J}/results/maps", exist_ok=True)
np.save(f"{J}/results/maps/ours_{ds}_s{seed}.npy", o["pred"].numpy().reshape(H, W).astype(np.uint8))
np.save(f"{J}/results/maps/ours_conf_{ds}_s{seed}.npy", o["proba"].max(1).values.numpy().reshape(H, W).astype(np.float16))
m = metrics(g0[te.numpy()] - 1, o["pred"].numpy()[te.numpy()], nc); print(ds, seed, "OA %.2f" % m["OA"], "gamma", o["gamma"])
