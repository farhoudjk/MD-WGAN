#!/usr/bin/env python3
"""Which knob carries the adaptive result?

Each arm enables exactly one knob of the controller, with the others left at
their defaults, so the Eq. (8) improvement attributable to each is separable.
Scored under the scenario's own alpha, against the no-control arm.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "knobs")
ALPHA = {"network": (0.15, 0.15, 0.15, 0.15, 0.40),
         "resource": (0.14, 0.29, 0.29, 0.14, 0.14)}
KNOBS = ["none", "top_k", "E_G", "sync_every", "lambda_sup", "all"]


def terms(run):
    rd = run["rounds"]
    tail = rd[int(len(rd) * 0.8):] or rd
    m = lambda k: float(np.mean([r[k] for r in tail]))
    return np.array([run["evals"][-1].get("crps", np.nan), m("cpu_usage"),
                     m("memory_usage"), m("latency"), m("network_traffic")])


def main():
    data = {}
    for f in glob.glob(os.path.join(RUNS, "knob-*.json")):
        stem, seed = os.path.basename(f)[:-5].split("__seed")
        _, scen, knob = stem.split("-", 2)
        data.setdefault((scen, knob), {})[int(seed)] = terms(json.load(open(f)))
    if not data:
        print("no knob runs yet"); return

    P = np.array([v for d in data.values() for v in d.values()], float)
    lo, hi = np.nanmin(P, 0), np.nanmax(P, 0)
    rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)

    for scen, a in ALPHA.items():
        rows = []
        for knob in KNOBS:
            d = data.get((scen, knob))
            if not d:
                continue
            c = {s: float(np.dot(a, (t - lo) / rng)) for s, t in d.items()}
            rows.append((knob, c))
        if not rows or rows[0][0] != "none":
            base = dict(rows[0][1]) if rows else {}
        else:
            base = dict(rows[0][1])
        print(f"\n### {scen}-weighted   (Eq. 8 cost, lower better)")
        print(f"{'knob enabled':<14}{'cost':>9}{'vs none':>10}{'p':>8}")
        print("-" * 41)
        for knob, c in rows:
            mean = float(np.mean(list(c.values())))
            if knob == "none":
                print(f"{knob:<14}{mean:>9.4f}{'--':>10}{'--':>8}")
                continue
            gain = 100.0 * (np.mean(list(base.values())) - mean) / \
                max(np.mean(list(base.values())), 1e-9)
            ss = sorted(set(c) & set(base))
            p = np.nan
            if len(ss) >= 2:
                x = np.array([c[k] for k in ss]); y = np.array([base[k] for k in ss])
                if np.any(x != y):
                    try:
                        p = stats.wilcoxon(x, y).pvalue
                    except ValueError:
                        pass
            star = " *" if np.isfinite(p) and p < 0.05 else ""
            print(f"{knob:<14}{mean:>9.4f}{gain:>9.1f}%"
                  + (f"{p:>8.3f}" if np.isfinite(p) else f"{'--':>8}") + star)
    print("\nn=5 seeds; 0.062 is the smallest attainable p at n=5.")


if __name__ == "__main__":
    main()
