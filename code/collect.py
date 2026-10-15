"""Aggregate every per-seed JSON into results/summary.json (the single source for tables, figures, paper and report)."""
import json, glob, os, re, numpy as np
from scipy import stats
from common import *

EVAL = list(range(10, 20)); DEV = list(range(0, 10))


def load_runs(d, ds, seeds=None):
    out = {}
    for f in glob.glob(f"{d}/{ds}_s*.json"):
        s = int(re.search(r"_s(\d+)\.json", f).group(1))
        if seeds is None or s in seeds: out[s] = json.load(open(f))
    return dict(sorted(out.items()))


def agg(runs):
    if not runs: return None
    a = {k: np.array([r[k] for r in runs.values()]) for k in ("OA", "AA", "K")}
    cls = np.array([r["cls"] for r in runs.values()])
    o = dict(n=len(runs), seeds=list(runs), cls_mean=cls.mean(0).tolist(), cls_sd=cls.std(0, ddof=1).tolist() if len(runs) > 1 else [0] * cls.shape[1],
             cm=np.sum([r["cm"] for r in runs.values()], 0).tolist())
    for k, v in a.items(): o[k] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]; o[k + "_all"] = v.tolist()
    for k in ("gamma", "fit_s", "train_s", "test_s", "val", "best_ep"):
        if k in next(iter(runs.values())): o[k + "_all"] = [r[k] for r in runs.values()]
    r0 = next(iter(runs.values()))
    if "weights" in r0: o["weights_mean"] = np.mean([r["weights"] for r in runs.values()], 0).tolist(); o["align_mean"] = np.mean([r["align"] for r in runs.values()], 0).tolist(); o["names"] = r0["names"]; o["dims"] = r0["dims"]
    if "table" in r0 and r0["table"]: o["cv_table_mean"] = {k: float(np.mean([r["table"][k] for r in runs.values() if k in r["table"]])) for k in r0["table"]}
    if "n_params" in r0: o["n_params"] = r0["n_params"]
    return o


def paired(a, b):
    """a, b: dicts seed -> run. Paired comparison of OA / AA / K on common seeds."""
    s = sorted(set(a) & set(b)); out = dict(n=len(s))
    if len(s) < 3: return out
    for k in ("OA", "AA", "K"):
        d = np.array([a[i][k] - b[i][k] for i in s]); t = stats.ttest_rel([a[i][k] for i in s], [b[i][k] for i in s])
        try: w = stats.wilcoxon(d).pvalue
        except ValueError: w = float("nan")
        se = d.std(ddof=1) / np.sqrt(len(d)); tc = stats.t.ppf(0.975, len(d) - 1)
        out[k] = dict(diff=float(d.mean()), lo=float(d.mean() - tc * se), hi=float(d.mean() + tc * se), p_t=float(t.pvalue), p_w=float(w), wins=int((d > 0).sum()))
    return out


S = dict(datasets={}, pub=PUB, pub_others=PUB_OTHERS, names=NAMES, table1=TABLE1)
for ds in ("HUS", "PU"):
    D = {}
    # reference re-runs
    for tag in sorted(os.path.basename(p) for p in glob.glob(f"{J}/results/ldman_*") + glob.glob(f"{J}/results/pilot_*")):
        if tag == "ldman_main":
            D["ldman_main"] = agg(load_runs(f"{J}/results/{tag}", ds, EVAL)); D["ldman_main_dev"] = agg(load_runs(f"{J}/results/{tag}", ds, DEV))
        else:
            a = agg(load_runs(f"{J}/results/{tag}", ds))
            if a: D[tag] = a
    for p in sorted(glob.glob(f"{J}/results/ours/*")):
        tag = os.path.basename(p)
        if tag == "main":
            D["main"] = agg(load_runs(p, ds, EVAL)); D["main_dev"] = agg(load_runs(p, ds, DEV))
        else:
            a = agg(load_runs(p, ds))
            if a: D[tag] = a
    for tag in ("main", "abl_nobal", "head_ncm"):   # same five splits as the slow variants (10-14)
        D[tag + "_e5"] = agg(load_runs(f"{J}/results/ours/{tag}", ds, range(10, 15)))
    D = {k: v for k, v in D.items() if v}
    # paired tests against the re-run reference on identical splits
    T = {}
    pairs = [("main", "ldman_main", EVAL), ("main", "ldman_main", DEV)] + [(f"n{k}", f"ldman_n{k}", None) for k in (5, 10, 20, 40)] + [("official", "ldman_official", None)]
    for a, b, seeds in pairs:
        ra = load_runs(f"{J}/results/ours/{a}", ds, seeds); rb = load_runs(f"{J}/results/{b}", ds, seeds)
        if ra and rb: T[f"{a}_vs_{b}" + ("_dev" if seeds is DEV else "")] = paired(ra, rb)
    for tag in D:
        if tag.startswith(("abl_", "bb_", "fus_", "C_", "head_", "post_")):
            ra = load_runs(f"{J}/results/ours/main", ds, EVAL); rb = load_runs(f"{J}/results/ours/{tag}", ds, EVAL)
            if ra and rb: T[f"main_vs_{tag}"] = paired(ra, rb)
    # comparison with the printed numbers: Welch test from summary statistics (article: 5 runs)
    W = {}
    if "main" in D:
        for k in ("OA", "AA", "K"):
            m, s, n = D["main"][k][0], D["main"][k][1], D["main"]["n"]; pm, ps = PUB[ds][k]
            t = stats.ttest_ind_from_stats(m, s, n, pm, ps, 5, equal_var=False)
            W[k] = dict(ours=m, ours_sd=s, pub=pm, pub_sd=ps, diff=m - pm, p=float(t.pvalue), seeds_above=int((np.array(D["main"][k + "_all"]) > pm).sum()))
        cls = np.array(D["main"]["cls_mean"]); pc = np.array(PUB[ds]["cls"])
        W["cls_wins"] = int((cls > pc + 1e-9).sum()); W["cls_ties"] = int((np.abs(cls - pc) <= 1e-9).sum()); W["cls_n"] = len(pc)
        W["cls_diff"] = (cls - pc).tolist()
    S["datasets"][ds] = dict(runs=D, tests=T, vs_pub=W)

# overall tally against the printed LDMAN cells (3 summary metrics + per-class accuracies, both scenes)
tot = win = tie = 0
for ds in ("HUS", "PU"):
    W = S["datasets"][ds]["vs_pub"]
    if not W: continue
    tot += 3 + W["cls_n"]; win += sum(W[k]["diff"] > 0 for k in ("OA", "AA", "K")) + W["cls_wins"]; tie += W["cls_ties"]
S["tally"] = dict(cells=tot, wins=win, ties=tie)
json.dump(S, open(f"{J}/results/summary.json", "w"), indent=1)

for ds in ("HUS", "PU"):
    D = S["datasets"][ds]["runs"]; print(f"== {ds}  published LDMAN OA {PUB[ds]['OA'][0]} AA {PUB[ds]['AA'][0]} K {PUB[ds]['K'][0]}")
    for k, v in D.items(): print(f"  {k:22s} n={v['n']:2d} OA {v['OA'][0]:.2f}±{v['OA'][1]:.2f} AA {v['AA'][0]:.2f}±{v['AA'][1]:.2f} K {v['K'][0]:.2f}±{v['K'][1]:.2f}")
    for k, v in S["datasets"][ds]["tests"].items():
        if "OA" in v and not k.startswith("main_vs_"): print(f"  test {k}: dOA {v['OA']['diff']:+.2f} [{v['OA']['lo']:+.2f},{v['OA']['hi']:+.2f}] p_t={v['OA']['p_t']:.2g} p_w={v['OA']['p_w']:.2g} wins {v['OA']['wins']}/{v['n']}")
print("tally vs printed LDMAN cells:", S["tally"])
