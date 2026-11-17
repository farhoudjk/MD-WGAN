#!/usr/bin/env python3
"""Single human-readable summary of everything the sweep has established.

Written to be the first thing read after an unattended run: what the best
configuration is, what each component is worth, and which claims survive
significance testing.
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from analysis.stats import paired_test, holm_bonferroni   # noqa: E402

RUNS = os.path.join(os.path.dirname(HERE), "runs")
TARGET = 0.86          # fixed RMSE threshold for time-to-target


def load():
    d = {}
    for f in glob.glob(os.path.join(RUNS, "*", "*.json")):
        name, seed = os.path.basename(f)[:-5].split("__seed")
        try:
            d.setdefault(name, {})[int(seed)] = json.load(open(f))
        except Exception:
            pass
    return d


def metrics(run):
    ev = run.get("evals", [])
    if not ev:
        return None
    last = ev[-1]
    rr = [e["rmse"] for e in ev]
    rn = [e["round"] for e in ev]
    hit = [rn[i] for i, v in enumerate(rr) if v <= TARGET]
    conv = hit[0] if hit else np.nan
    work = sum(r["cpu_usage"] for r in run["rounds"]
               if not hit or r["round"] <= conv)
    tail = run["rounds"][int(len(run["rounds"]) * 0.8):]
    m = dict(
        rmse=last.get("rmse", np.nan), mae=last.get("mae", np.nan),
        crps=last.get("crps", np.nan), min_ade=last.get("min_ade", np.nan),
        picp=last.get("picp", np.nan),
        coverage=last.get("mode_coverage", np.nan),
        conv_rounds=conv, reached=float(bool(hit)), work_to_target=work,
        objective=run.get("objective_cost", np.nan),
        traffic=np.mean([r["network_traffic"] for r in tail]) / 1e6,
        latency=np.mean([r["latency"] for r in tail]),
        stability=float(np.std(rr[int(len(rr) * 0.8):])),
    )
    return m


def agg(d):
    out = {}
    for name, seeds in d.items():
        ms = [metrics(r) for r in seeds.values()]
        ms = [m for m in ms if m]
        if not ms:
            continue
        out[name] = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
        out[name]["n"] = len(ms)
        out[name]["rmse_sd"] = float(np.nanstd([m["rmse"] for m in ms], ddof=1)) if len(ms) > 1 else 0.0
        out[name]["_per_seed"] = {s: metrics(r) for s, r in seeds.items() if metrics(r)}
    return out


def table(rows, cols, keyfn, title, note=""):
    print(f"\n## {title}\n")
    hdr = f"{'arm':<26}" + "".join(f"{c:>11}" for c, _ in cols)
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(rows, key=keyfn):
        r = rows[name]
        line = f"{name:<26}"
        for c, k in cols:
            v = r.get(k, np.nan)
            line += f"{v:>11.4f}" if np.isfinite(v) else f"{'--':>11}"
        print(line)
    if note:
        print(f"\n{note}")


def main():
    d = load()
    if not d:
        print("no results yet")
        return
    A = agg(d)
    done = sum(len(v) for v in d.values())
    print("# MD-WGAN results\n")
    print(f"{done} runs, {len(A)} arms. Target for time-to-convergence: RMSE <= {TARGET}.")
    print("Lower is better for every metric except PICP (ideal 0.90) and coverage (ideal 1.0).")

    # --- best configuration by several criteria -------------------------
    print("\n## Best configuration by each criterion\n")
    crit = [("CRPS (distributional accuracy)", "crps", False),
            ("work to reach target", "work_to_target", False),
            ("rounds to reach target", "conv_rounds", False),
            ("minADE (multi-modal accuracy)", "min_ade", False),
            ("mode coverage", "coverage", True),
            ("RMSE (point accuracy)", "rmse", False),
            ("Eq.(8) objective", "objective", False)]
    for label, key, hi in crit:
        valid = {k: v[key] for k, v in A.items() if np.isfinite(v.get(key, np.nan))}
        if not valid:
            continue
        best = max(valid, key=valid.get) if hi else min(valid, key=valid.get)
        print(f"  {label:<34} -> {best:<26} ({valid[best]:.4f})")

    cols = [("CRPS", "crps"), ("minADE", "min_ade"), ("PICP", "picp"),
            ("cov", "coverage"), ("RMSE", "rmse"), ("conv", "conv_rounds"),
            ("work", "work_to_target")]
    table(A, cols, lambda k: A[k].get("crps", 9e9),
          "All arms, ranked by CRPS",
          "CRPS is strictly proper: unlike RMSE it cannot be improved by\n"
          "collapsing onto the conditional mean.")

    # --- significance vs the current method ------------------------------
    ref = "full" if "full" in d else ("cand-current" if "cand-current" in d else None)
    if ref:
        print(f"\n## Paired significance vs '{ref}' (Wilcoxon, Holm-corrected)\n")
        for metric in ("crps", "rmse", "work_to_target", "coverage"):
            names, ps, diffs = [], [], []
            for k in d:
                if k == ref or "_per_seed" not in A.get(k, {}):
                    continue
                a = {s: m[metric] for s, m in A[k]["_per_seed"].items()}
                b = {s: m[metric] for s, m in A[ref]["_per_seed"].items()}
                t = paired_test(a, b)
                if np.isfinite(t["p_wilcoxon"]):
                    names.append(k); ps.append(t["p_wilcoxon"]); diffs.append(t["mean_diff"])
            if not names:
                continue
            adj = holm_bonferroni(ps)
            sig = [(n, df, p) for n, df, p in zip(names, diffs, adj) if np.isfinite(p) and p < 0.05]
            print(f"  {metric}: " + (f"{len(sig)} of {len(names)} arms differ significantly" if sig
                                      else f"no arm differs significantly (n={len(names)})"))
            for n, df, p in sorted(sig, key=lambda x: x[1]):
                print(f"      {n:<26} {df:+.4f}  p={p:.4f}")

    print("\n## Caveats\n")
    print("  - The 0.86 target was chosen after inspecting data; fix it by a")
    print("    principled rule or report a threshold curve before submission.")
    print("  - 'work' counts gradient steps, so arms that simply train less")
    print("    score well; read it together with conv/CRPS, not alone.")
    print("  - Arms not reaching the target have conv/work from a truncated run.")


if __name__ == "__main__":
    main()
