#!/usr/bin/env python3
"""Plain-text attribution summary: what each component is actually worth.

Prints, for every ablation arm, the change in each metric relative to the full
framework together with the paired test, so the contribution of each mechanism
can be read off directly.  Intended to be run first, before any paper text is
written, so the framing follows the evidence.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from analysis.stats import summarise   # noqa: E402

LOWER_IS_BETTER = {"rmse", "mae", "js_proxy", "cpu_usage", "memory_usage",
                   "latency", "network_traffic", "objective_cost",
                   "rounds_to_converge", "rmse_stability"}


def verdict(p):
    if not np.isfinite(p):
        return "n/a"
    return "significant" if p < 0.05 else "n.s."


def report(run_root, suite, reference, metrics):
    rows, tests, per_arm = summarise(run_root, suite, reference=reference)
    if reference not in rows:
        print(f"[{suite}] reference arm '{reference}' not found; have: "
              f"{sorted(rows)}")
        return

    print(f"\n{'=' * 78}\n{suite.upper()}  (reference = {reference})\n{'=' * 78}")
    ref = rows[reference]
    print(f"{reference} baseline: " + ", ".join(
        f"{m}={ref.get(m, float('nan')):.4g}" for m in metrics))
    print(f"seeds per arm: {ref.get('n_seeds')}\n")

    for metric in metrics:
        print(f"-- {metric} " + "-" * (74 - len(metric)))
        base = ref.get(metric, np.nan)
        entries = []
        for name, r in rows.items():
            if name == reference:
                continue
            val = r.get(metric, np.nan)
            if not np.isfinite(val) or not np.isfinite(base):
                continue
            t = tests.get((name, metric), {})
            delta = val - base
            pct = 100.0 * delta / abs(base) if abs(base) > 1e-12 else np.nan
            entries.append((name, val, r.get(metric + "_std", 0.0), delta, pct,
                            t.get("p_wilcoxon", np.nan),
                            t.get("p_holm", np.nan),
                            t.get("cohen_dz", np.nan)))
        # Largest degradation relative to the full framework first: that is the
        # component whose removal hurts most, i.e. the one carrying the claim.
        sign = 1 if metric in LOWER_IS_BETTER else -1
        entries.sort(key=lambda e: -sign * e[3])
        print(f"{'arm':<22}{'value':>11}{'± sd':>9}{'Δ':>10}{'Δ%':>9}"
              f"{'p':>8}{'p_holm':>9}{'d_z':>7}  verdict")
        for n, v, sd, d, pct, p, ph, dz in entries:
            print(f"{n:<22}{v:>11.4g}{sd:>9.3g}{d:>+10.4g}{pct:>+8.1f}%"
                  f"{p:>8.3f}{ph:>9.3f}{dz:>7.2f}  {verdict(ph)}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(HERE), "runs"))
    ap.add_argument("--metrics", nargs="+",
                    default=["rmse", "mae", "cpu_usage", "memory_usage",
                             "latency", "network_traffic", "objective_cost",
                             "mode_coverage"])
    a = ap.parse_args()

    for suite, ref in (("ablation", "full"), ("baseline", "sel-dqn"),
                       ("noniid", "noniid-natural")):
        if os.path.isdir(os.path.join(a.runs, suite)):
            report(a.runs, suite, ref, a.metrics)


if __name__ == "__main__":
    main()
