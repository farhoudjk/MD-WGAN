#!/usr/bin/env python3
"""One table per metric: mean +- s.d. over seeds, sorted best-first."""
import glob, json, os, sys
import numpy as np
from scipy import stats

TH = 0.90
METRICS = [("rmse", "RMSE", False), ("crps", "CRPS", False),
           ("min_ade", "minADE", False), ("picp", "PICP (ideal 0.90)", None),
           ("mode_coverage", "Mode coverage", True),
           ("conv", "Rounds to target", False), ("work", "Compute to target", False),
           ("latency", "Latency (s)", False), ("traffic", "Traffic (MB)", False)]


def load(run_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        name, seed = os.path.basename(f)[:-5].split("__seed")
        d = json.load(open(f)); ev, rd = d["evals"], d["rounds"]
        last = ev[-1]
        hit = [e["round"] for e in ev if e["rmse"] <= TH]
        tail = rd[int(len(rd) * 0.8):]
        out.setdefault(name, {})[int(seed)] = {
            "rmse": last.get("rmse"), "crps": last.get("crps"),
            "min_ade": last.get("min_ade"), "picp": last.get("picp"),
            "mode_coverage": last.get("mode_coverage"),
            "conv": hit[0] if hit else np.nan,
            "work": sum(x["cpu_usage"] for x in rd if not hit or x["round"] <= hit[0]),
            "latency": np.mean([x["latency"] for x in tail]),
            "traffic": np.mean([x["network_traffic"] for x in tail]) / 1e6}
    return out


def main(run_dir, ref=None):
    D = load(run_dir)
    if not D:
        print("no runs in", run_dir); return
    names = list(D)
    ref = ref if ref in names else None
    for key, title, hi in METRICS:
        rows = []
        for n in names:
            v = np.array([D[n][s][key] for s in sorted(D[n])], float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            p = np.nan
            if ref and n != ref:
                ss = sorted(set(D[n]) & set(D[ref]))
                a = np.array([D[n][s][key] for s in ss], float)
                b = np.array([D[ref][s][key] for s in ss], float)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= 2 and np.any(a[ok] != b[ok]):
                    try:
                        p = stats.wilcoxon(a[ok], b[ok]).pvalue
                    except ValueError:
                        p = np.nan
            rows.append((n, v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v), p))
        if not rows:
            continue
        keyf = (lambda r: abs(r[1] - 0.90)) if hi is None else (lambda r: -r[1] if hi else r[1])
        rows.sort(key=keyf)
        print(f"\n### {title}" + ("   (closer to 0.90 is better)" if hi is None
                                 else "   (higher better)" if hi else "   (lower better)"))
        print(f"{'arm':<13}{'mean':>10}{'sd':>9}{'n':>4}" + (f"{'p vs '+ref:>12}" if ref else ""))
        print("-" * (36 + (12 if ref else 0)))
        for n, m, sd, c, p in rows:
            line = f"{n:<13}{m:>10.4f}{sd:>9.4f}{c:>4d}"
            if ref:
                line += f"{'--' if n == ref else (f'{p:.3f}' if np.isfinite(p) else 'n/a'):>12}"
                if np.isfinite(p) and p < 0.05 and n != ref:
                    line += " *"
            print(line)
    if ref:
        print(f"\nPaired Wilcoxon vs '{ref}'. * = p<0.05.")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "mdwgan/runs_setsel"
    main(d, sys.argv[2] if len(sys.argv) > 2 else None)
