#!/usr/bin/env python3
"""Score the adaptive arms on Eq. (8), each under its own weighting.

The point of the adaptive controller is that a deployment states what it cares
about and the framework moves to suit it.  Scoring every arm under one common
weighting would therefore miss the whole claim.  For each scenario we ask only:
under *this* scenario's alpha, does adapting beat a fixed configuration?

Terms are min-max normalised across all arms so the five components of Eq. (8)
are commensurable, exactly as in the reported comparison with the baselines.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

SCEN = ["balanced", "accuracy", "delay", "network", "resource"]
MODES = ["none", "static", "online"]
ALPHA = {
    "balanced": (0.20, 0.20, 0.20, 0.20, 0.20),
    "accuracy": (0.40, 0.15, 0.15, 0.15, 0.15),
    "delay":    (0.15, 0.15, 0.15, 0.40, 0.15),
    "network":  (0.15, 0.15, 0.15, 0.15, 0.40),
    "resource": (0.14, 0.29, 0.29, 0.14, 0.14),
}


def raw_terms(run):
    """The five Eq. (8) terms realised by one run, before normalisation."""
    rd = run["rounds"]
    tail = rd[int(len(rd) * 0.8):] or rd
    m = lambda k: float(np.mean([r[k] for r in tail]))
    last = run["evals"][-1]
    # Training term: CRPS is the strictly proper accuracy score, so it is the
    # quantity the alpha_G weight should act on.
    return np.array([last.get("crps", np.nan), m("cpu_usage"),
                     m("memory_usage"), m("latency"), m("network_traffic")])


def main(run_dir):
    runs = {}
    for f in glob.glob(os.path.join(run_dir, "adapt-*.json")):
        base, seed = os.path.basename(f)[:-5].split("__seed")
        _, scen, mode = base.split("-")
        runs.setdefault((scen, mode), {})[int(seed)] = raw_terms(json.load(open(f)))
    if not runs:
        print("no adaptive runs in", run_dir); return

    # Normalise each term across every arm so the weighted sum is comparable.
    allv = np.array([v for d in runs.values() for v in d.values()], float)
    lo, hi = np.nanmin(allv, 0), np.nanmax(allv, 0)
    rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    norm = lambda t: (t - lo) / rng

    print(f"{'scenario':<10}{'none':>9}{'static':>9}{'online':>9}"
          f"{'best':>10}{'gain%':>8}{'p':>8}")
    print("-" * 63)
    rows = []
    for s in SCEN:
        cost = {}
        for mode in MODES:
            d = runs.get((s, mode))
            if not d:
                continue
            cost[mode] = {sd: float(np.dot(ALPHA[s], norm(t)))
                          for sd, t in d.items()}
        if "none" not in cost:
            continue
        mean = {m: float(np.mean(list(v.values()))) for m, v in cost.items()}
        best = min(mean, key=mean.get)
        gain = 100.0 * (mean["none"] - mean[best]) / max(mean["none"], 1e-9)
        p = np.nan
        if best != "none":
            ss = sorted(set(cost[best]) & set(cost["none"]))
            a = np.array([cost[best][k] for k in ss])
            b = np.array([cost["none"][k] for k in ss])
            if len(ss) >= 2 and np.any(a != b):
                try:
                    p = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    pass
        print(f"{s:<10}" + "".join(f"{mean.get(m, float('nan')):>9.4f}" for m in MODES)
              + f"{best:>10}{gain:>7.1f}%"
              + (f"{p:>8.3f}" if np.isfinite(p) else f"{'--':>8}")
              + ("  *" if np.isfinite(p) and p < 0.05 else ""))
        rows.append((s, mean, best, gain, p))

    wins = sum(1 for _, _, b, _, _ in rows if b != "none")
    print(f"\nAdaptive beats fixed in {wins}/{len(rows)} scenarios.")
    print("Each row scored under its OWN alpha. * = p<0.05 (paired Wilcoxon, n=5;")
    print("note 0.062 is the smallest attainable p at n=5, so absence of * at")
    print("n=5 is weak evidence of no effect, not evidence of no effect.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "runs", "adaptive"))
