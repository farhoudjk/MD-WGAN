"""Aggregation and significance testing across seeds.

The headline numbers in the submitted paper (RMSE 1.029 vs 1.032 vs 1.044) are
reported without any measure of spread, so a reviewer cannot tell whether the
ordering is real.  Everything here is built to answer that: paired tests across
common seeds, bootstrap confidence intervals, and effect sizes.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

METRICS = ["rmse", "mae", "mode_coverage", "js_proxy"]
RESOURCE = ["cpu_usage", "memory_usage", "latency", "network_traffic", "comm_latency"]


def load_suite(run_root, suite):
    """Load every (arm, seed) result for one suite."""
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(run_root, suite, "*.json"))):
        base = os.path.basename(path)[:-5]
        name, seed = base.split("__seed")
        with open(path) as f:
            out[name][int(seed)] = json.load(f)
    return dict(out)


def arm_metrics(result):
    """Reduce one run to the scalar metrics used in the tables."""
    final = result.get("final", {})
    rounds = result.get("rounds", [])
    tail = rounds[max(0, int(len(rounds) * 0.8)):] or rounds
    m = {k: final.get(k, np.nan) for k in METRICS}
    for k in RESOURCE:
        m[k] = float(np.mean([r[k] for r in tail])) if tail else np.nan
    m["objective_cost"] = result.get("objective_cost", np.nan)
    m["equilibrium_rate"] = result.get("equilibrium_rate", np.nan)
    m["wall_seconds"] = result.get("wall_seconds", np.nan)
    # Convergence speed: first round reaching within 2% of the run's best RMSE.
    evals = result.get("evals", [])
    if evals:
        r = np.array([e["rmse"] for e in evals])
        rounds_ax = np.array([e["round"] for e in evals])
        best = np.nanmin(r)
        hit = np.where(r <= best * 1.02)[0]
        m["rounds_to_converge"] = float(rounds_ax[hit[0]]) if len(hit) else np.nan
        m["final_rmse"] = float(r[-1])
        m["best_rmse"] = float(best)
        # Stability: std of RMSE over the last 20% of evaluations.
        m["rmse_stability"] = float(np.std(r[max(0, int(len(r) * 0.8)):]))
    return m


def bootstrap_ci(x, n=10000, alpha=0.05, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return (float(x.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_test(a, b):
    """Paired comparison of two arms over their common seeds.

    Paired (rather than independent) because both arms see the same seeds, and
    the seed effect is large relative to the arm effect.  Reports the Wilcoxon
    signed-rank p (no normality assumption, appropriate at n=5) alongside the
    paired t-test and Cohen's d_z.
    """
    seeds = sorted(set(a) & set(b))
    xa = np.array([a[s] for s in seeds], float)
    xb = np.array([b[s] for s in seeds], float)
    keep = np.isfinite(xa) & np.isfinite(xb)
    xa, xb = xa[keep], xb[keep]
    if len(xa) < 2:
        return {"n": len(xa), "p_t": np.nan, "p_wilcoxon": np.nan,
                "cohen_dz": np.nan, "mean_diff": np.nan}
    d = xa - xb
    t_p = float(stats.ttest_rel(xa, xb).pvalue)
    try:
        w_p = float(stats.wilcoxon(xa, xb).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w_p = np.nan
    sd = d.std(ddof=1)
    return {
        "n": int(len(xa)),
        "mean_diff": float(d.mean()),
        "p_t": t_p,
        "p_wilcoxon": w_p,
        "cohen_dz": float(d.mean() / sd) if sd > 1e-12 else np.nan,
    }


def holm_bonferroni(pvals):
    """Holm step-down adjustment for a family of comparisons."""
    items = [(p, i) for i, p in enumerate(pvals) if np.isfinite(p)]
    items.sort()
    adj = [np.nan] * len(pvals)
    m = len(items)
    prev = 0.0
    for rank, (p, i) in enumerate(items):
        val = min(1.0, (m - rank) * p)
        val = max(val, prev)      # enforce monotonicity
        adj[i] = val
        prev = val
    return adj


def summarise(run_root, suite, reference="full"):
    """Per-arm means with CIs, plus paired tests against the reference arm."""
    data = load_suite(run_root, suite)
    per_arm = {name: {seed: arm_metrics(res) for seed, res in seeds.items()}
               for name, seeds in data.items()}

    rows = {}
    for name, seeds in per_arm.items():
        keys = set().union(*[set(m) for m in seeds.values()]) if seeds else set()
        rows[name] = {"n_seeds": len(seeds)}
        for k in keys:
            mean, lo, hi = bootstrap_ci([m.get(k, np.nan) for m in seeds.values()])
            rows[name][k] = mean
            rows[name][k + "_ci"] = (lo, hi)
            rows[name][k + "_std"] = float(np.nanstd(
                [m.get(k, np.nan) for m in seeds.values()], ddof=1)) if len(seeds) > 1 else 0.0

    tests = {}
    if reference in per_arm:
        for metric in ("rmse", "mae", "objective_cost", "cpu_usage",
                       "memory_usage", "latency", "network_traffic"):
            ref = {s: m.get(metric, np.nan) for s, m in per_arm[reference].items()}
            names, raw = [], []
            for name in per_arm:
                if name == reference:
                    continue
                cur = {s: m.get(metric, np.nan) for s, m in per_arm[name].items()}
                res = paired_test(cur, ref)
                tests[(name, metric)] = res
                names.append(name)
                raw.append(res["p_wilcoxon"])
            for name, adj in zip(names, holm_bonferroni(raw)):
                tests[(name, metric)]["p_holm"] = adj
    return rows, tests, per_arm
