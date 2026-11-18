#!/usr/bin/env python3
"""Summary tables for the full-NGSIM scalability study.

Reads `scalability results/runs/{scale_rsu,scale_pool}/*.json` (5 seeds per
arm) and writes summary.md and summary.csv into `scalability results/`.
Mean +- s.d. over seeds; resource terms are means over the final 20% of rounds,
traffic is the mean per round summed over all RSUs.
"""
import csv, glob, json, os
import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scalability results")
LAB = {"dqn": "Ours (dynamic) + DQN", "greedy": "Ours (dynamic) + greedy",
       "random": "Ours (dynamic) + random", "early-update-mdgan": "EU-MDGAN",
       "hvs-gan": "HVS-GAN", "fedavg-gan": "FedAvg-GAN"}
TH = 0.53   # RMSE target (about 6% below repeat-last-frame, 0.566)
GROUPS = [("scale_rsu", "RSU count (pool 5 per RSU)", [("rsu2", 2), ("rsu4", 4), ("rsu6", 6), ("rsu8", 8)], "RSUs"),
          ("scale_pool", "Pool size per RSU (2 RSUs)", [("pool3", 3), ("pool5", 5), ("pool8", 8), ("pool10", 10)], "Pool")]
MET = [("rmse", "RMSE", 3), ("crps", "CRPS", 3), ("min_ade", "minADE", 3), ("picp", "PICP", 2),
       ("mode_coverage", "Coverage", 3), ("latency", "Latency (s)", 2),
       ("traffic", "Traffic (MB/round, all RSUs)", 1), ("compute", "Compute/round", 1),
       ("memory", "Memory (MB)", 0), ("rounds", "Rounds to RMSE<=0.53", 1)]


def summarise(files):
    R = [json.load(open(f)) for f in files]
    tail = lambda r, k: np.mean([x[k] for x in r["rounds"][int(len(r["rounds"]) * .8):]])
    v = {k: np.array([r["evals"][-1][k] for r in R]) for k in ("rmse", "crps", "min_ade", "picp", "mode_coverage")}
    v["latency"] = np.array([tail(r, "latency") for r in R])
    v["traffic"] = np.array([sum(x["network_traffic"] for x in r["rounds"]) / r["config"]["rounds"] for r in R]) / 1e6
    v["compute"] = np.array([tail(r, "cpu_usage") for r in R])
    v["memory"] = np.array([tail(r, "memory_usage") for r in R])
    hits = [min([e["round"] for e in r["evals"] if e["rmse"] <= TH], default=None) for r in R]
    v["rounds"] = np.array([h for h in hits if h is not None], float)
    v["_reached"] = sum(h is not None for h in hits)
    v["_n"] = len(R)
    return v


def main():
    md = ["# Scalability results (full NGSIM)", "",
          "Mean +- s.d. over 5 seeds, 60 rounds. See README.md for the setup.", ""]
    rows = []
    for suite, title, levels, col in GROUPS:
        data = {}
        for tag, lv in levels:
            for m in LAB:
                fs = sorted(glob.glob(os.path.join(ROOT, "runs", suite, f"{tag}-{m}__seed*.json")))
                if fs:
                    data[(lv, m)] = summarise(fs)
        md += [f"## {title}", ""]
        for key, name, dp in MET:
            md += [f"### {name}", "", "| Method | " + " | ".join(f"{col} {lv}" for _, lv in levels) + " |",
                   "|---|" + "---|" * len(levels)]
            for m, lab in LAB.items():
                cells = []
                for _, lv in levels:
                    v = data.get((lv, m))
                    if v is None:
                        cells.append("-"); continue
                    x = v[key]
                    if key == "rounds":
                        cells.append("never" if not len(x) else f"{x.mean():.{dp}f} ({v['_reached']}/{v['_n']})")
                    else:
                        cells.append(f"{x.mean():.{dp}f} +- {x.std(ddof=1):.{dp}f}")
                    rows.append({"study": suite, col.lower(): lv, "method": lab, "metric": name,
                                 "mean": float(x.mean()) if len(x) else "", "sd": float(x.std(ddof=1)) if len(x) > 1 else "",
                                 "n": v["_n"]})
                md.append(f"| {lab} | " + " | ".join(cells) + " |")
            md.append("")
        # Selector comparison at each level.
        md += [f"### DQN vs greedy / random (seeds where DQN is better, of 5; paired Wilcoxon p)", "",
               "| " + col + " | Metric | DQN vs greedy | DQN vs random |", "|---|---|---|---|"]
        for _, lv in levels:
            for k in ("rmse", "crps", "min_ade", "picp"):
                a = data[(lv, "dqn")][k]; out = []
                for o in ("greedy", "random"):
                    b = data[(lv, o)][k]
                    w = int(np.sum(abs(a - .9) < abs(b - .9))) if k == "picp" else int(np.sum(a < b))
                    out.append(f"{w}/5, p={stats.wilcoxon(a, b).pvalue:.3f}")
                md.append(f"| {lv} | {k} | {out[0]} | {out[1]} |")
        md.append("")
    open(os.path.join(ROOT, "summary.md"), "w").write("\n".join(md) + "\n")
    with open(os.path.join(ROOT, "summary.csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=["study", "rsus", "pool", "method", "metric", "mean", "sd", "n"].index)
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote summary.md and summary.csv")


if __name__ == "__main__":
    main()
