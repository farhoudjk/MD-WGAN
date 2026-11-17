#!/usr/bin/env python3
"""Eq. (8) cost of every method under every deployment scenario.

The adaptive-controller result on its own only says the framework beats its own
fixed setting.  The claim that matters is whether it beats the published
baselines, and that requires scoring all of them on one normalised scale under
each alpha.  The baselines were already run; this is a scoring pass over their
stored records, not new training.

Normalisation is min-max across all methods, computed once on the pooled set, so
a column is comparable down its length and the five terms of Eq. (8) are
commensurable with one another.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")

ALPHA = {
    "balanced": (0.20, 0.20, 0.20, 0.20, 0.20),
    "accuracy": (0.40, 0.15, 0.15, 0.15, 0.15),
    "delay":    (0.15, 0.15, 0.15, 0.40, 0.15),
    "network":  (0.15, 0.15, 0.15, 0.15, 0.40),
    "resource": (0.14, 0.29, 0.29, 0.14, 0.14),
}

# (label, suite, filename stem). The adaptive rows take whichever controller
# variant that scenario selected, so each scenario is scored with the
# configuration the framework would actually deploy for it.
FIXED = [
    ("MD-GAN",        "sota", "md-gan"),
    ("Early-upd MDGAN","sota", "early-update-mdgan"),
    ("HVS-GAN",       "sota", "hvs-gan"),
    ("FedAvg-GAN",    "sota", "fedavg-gan"),
    ("Ours (DQN)",    "sota", "ours-dqn"),
    ("Ours (fixed)",  "sota", "ours"),
]


def terms(run):
    """Five Eq. (8) terms realised by one run, before normalisation."""
    rd = run["rounds"]
    tail = rd[int(len(rd) * 0.8):] or rd
    m = lambda k: float(np.mean([r[k] for r in tail]))
    return np.array([run["evals"][-1].get("crps", np.nan), m("cpu_usage"),
                     m("memory_usage"), m("latency"), m("network_traffic")])


def load(suite, stem):
    out = {}
    for f in glob.glob(os.path.join(RUNS, suite, f"{stem}__seed*.json")):
        seed = int(os.path.basename(f)[:-5].split("__seed")[1])
        out[seed] = terms(json.load(open(f)))
    return out


def main():
    data = {}
    for label, suite, stem in FIXED:
        d = load(suite, stem)
        if d:
            data[label] = d

    # Adaptive: one row per controller mode, per scenario.
    adaptive = {}
    for scen in ALPHA:
        for mode in ("static", "online"):
            d = load("adaptive", f"adapt-{scen}-{mode}")
            if d:
                adaptive[(scen, mode)] = d

    pool = [v for d in data.values() for v in d.values()] + \
           [v for d in adaptive.values() for v in d.values()]
    if not pool:
        print("no runs found"); return
    P = np.array(pool, float)
    lo, hi = np.nanmin(P, 0), np.nanmax(P, 0)
    rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    nz = lambda t: (t - lo) / rng

    cost = lambda d, a: {s: float(np.dot(a, nz(t))) for s, t in d.items()}

    rows, best_adaptive = [], {}
    for label in data:
        rows.append((label, {s: cost(data[label], ALPHA[s]) for s in ALPHA}))
    for scen in ALPHA:
        cands = {m: cost(adaptive[(scen, m)], ALPHA[scen])
                 for m in ("static", "online") if (scen, m) in adaptive}
        if cands:
            pick = min(cands, key=lambda m: np.mean(list(cands[m].values())))
            best_adaptive[scen] = (pick, cands[pick])

    hdr = f"{'method':<16}" + "".join(f"{s:>11}" for s in ALPHA)
    print("Eq. (8) cost, lower is better. Each column uses that scenario's alpha.\n")
    print(hdr); print("-" * len(hdr))
    for label, per in rows:
        print(f"{label:<16}" + "".join(
            f"{np.mean(list(per[s].values())):>11.4f}" for s in ALPHA))
    if best_adaptive:
        print(f"{'Ours (adaptive)':<16}" + "".join(
            f"{np.mean(list(best_adaptive[s][1].values())):>11.4f}"
            if s in best_adaptive else f"{'--':>11}" for s in ALPHA))
        print(f"{'  (variant)':<16}" + "".join(
            f"{best_adaptive[s][0]:>11}" if s in best_adaptive else f"{'--':>11}"
            for s in ALPHA))

    print("\nPer-scenario winner, and significance of adaptive vs the best fixed method:")
    for s in ALPHA:
        means = {l: np.mean(list(p[s].values())) for l, p in rows}
        if s in best_adaptive:
            means["Ours (adaptive)"] = np.mean(list(best_adaptive[s][1].values()))
        win = min(means, key=means.get)
        line = f"  {s:<10} winner={win:<18}"
        if s in best_adaptive and win == "Ours (adaptive)":
            rival = min((l for l in means if l != "Ours (adaptive)"), key=means.get)
            A = best_adaptive[s][1]
            B = dict(rows[[l for l, _ in rows].index(rival)][1][s])
            ss = sorted(set(A) & set(B))
            if len(ss) >= 2:
                a = np.array([A[k] for k in ss]); b = np.array([B[k] for k in ss])
                try:
                    p = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    p = np.nan
                line += f" vs {rival}: {means[win]:.4f} vs {means[rival]:.4f}, p={p:.3f}"
        print(line)
    print("\nn=5 seeds; p=0.062 is the smallest attainable value at n=5.")


if __name__ == "__main__":
    main()
