#!/usr/bin/env python3
"""Generate LaTeX tables and figures for the paper from the sweep results."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

from analysis.stats import summarise, load_suite, arm_metrics   # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
    "savefig.bbox": "tight",
})

# Readable labels for the ablation arms.
LABELS = {
    "full": "MD-WGAN (full)",
    "no-rl": "w/o RL selection",
    "no-game": "w/o game coordination",
    "no-ensemble": "w/o ensemble feedback",
    "no-rl-game": "w/o RL + game",
    "no-rl-ensemble": "w/o RL + ensemble",
    "no-game-ensemble": "w/o game + ensemble",
    "none": "none (plain fed. WGAN)",
    "static-game": "static (non-adaptive) game",
    "unconditional": "unconditional $G(z)$",
    "no-supervised": "pure adversarial",
    "supervised-only": "supervised only",
    "ensemble-weighted": "performance-weighted ens.",
    "sel-dqn": "DQN (ours)", "sel-random": "Random-$k$",
    "sel-greedy": "Greedy-by-loss", "sel-roundrobin": "Round-robin",
    "sel-all": "All discriminators",
}


def stars(p):
    if not np.isfinite(p):
        return ""
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def fmt(mean, std, prec=3):
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{prec}f}\\,$\\pm$\\,{std:.{prec}f}"


def table_ablation(run_root, suite, reference, out_path, caption, label):
    rows, tests, _ = summarise(run_root, suite, reference=reference)
    order = [k for k in LABELS if k in rows] + [k for k in sorted(rows) if k not in LABELS]

    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{" + caption + r"}", r"\label{tab:" + label + r"}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Variant & RMSE & MAE & Mode cov. & CPU (\%) & Latency (s) & Traffic (MB) \\",
        r"\midrule",
    ]
    for name in order:
        r = rows[name]
        p = tests.get((name, "rmse"), {}).get("p_holm", np.nan)
        disp = LABELS.get(name, name.replace("_", r"\_"))
        if name == reference:
            disp = r"\textbf{" + disp + r"}"
        lines.append(" & ".join([
            disp,
            fmt(r.get("rmse", np.nan), r.get("rmse_std", 0)) + stars(p),
            fmt(r.get("mae", np.nan), r.get("mae_std", 0)),
            fmt(r.get("mode_coverage", np.nan), r.get("mode_coverage_std", 0)),
            fmt(r.get("cpu_usage", np.nan), r.get("cpu_usage_std", 0), 1),
            fmt(r.get("latency", np.nan), r.get("latency_std", 0), 2),
            fmt(r.get("network_traffic", np.nan) / 1e6,
                r.get("network_traffic_std", 0) / 1e6, 2),
        ]) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{flushleft}\footnotesize",
        r"Mean\,$\pm$\,s.d. over 5 seeds. Significance is a paired Wilcoxon "
        r"signed-rank test against the full framework, Holm-corrected across "
        r"variants: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.",
        r"\end{flushleft}", r"\end{table*}",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows, tests


def fig_convergence(run_root, suite, arms, out_path, title):
    """Mean RMSE vs round with a 95% CI band across seeds."""
    data = load_suite(run_root, suite)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name in arms:
        if name not in data:
            continue
        runs = list(data[name].values())
        rounds = [e["round"] for e in runs[0]["evals"]]
        mat = np.array([[e["rmse"] for e in r["evals"]] for r in runs])
        mean = mat.mean(0)
        se = mat.std(0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean)
        ax.plot(rounds, mean, label=LABELS.get(name, name), lw=1.6)
        ax.fill_between(rounds, mean - 1.96 * se, mean + 1.96 * se, alpha=0.18)
    ax.set_xlabel("Training round")
    ax.set_ylabel("RMSE (held-out)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.savefig(out_path)
    plt.close(fig)


def fig_tradeoff(run_root, suite, out_path):
    """Accuracy vs resource cost: shows the ablations are a real Pareto trade-off."""
    rows, _, _ = summarise(run_root, suite, reference="full")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, r in rows.items():
        x, y = r.get("cpu_usage", np.nan), r.get("rmse", np.nan)
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        marker = "*" if name == "full" else "o"
        size = 170 if name == "full" else 45
        ax.scatter(x, y, s=size, marker=marker, zorder=3)
        ax.annotate(LABELS.get(name, name), (x, y), fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean CPU utilisation (\\%)" if matplotlib.rcParams["text.usetex"]
                  else "Mean CPU utilisation (%)")
    ax.set_ylabel("RMSE (held-out)")
    ax.set_title("Accuracy vs resource cost")
    fig.savefig(out_path)
    plt.close(fig)


def fig_sensitivity(run_root, out_path):
    """Grouped sensitivity panels: subset size, pool size, reward weighting."""
    data = load_suite(run_root, "sensitivity")
    rows = {n: [arm_metrics(r) for r in s.values()] for n, s in data.items()}

    groups = [
        ("Subset size $k$", [f"k{i}" for i in (1, 2, 3, 4, 5)], [1, 2, 3, 4, 5]),
        ("Pool size per RSU", [f"pool{i}" for i in (3, 5, 8)], [3, 5, 8]),
        ("Generators $N_G$", [f"gens{i}" for i in (2, 4, 6)], [2, 4, 6]),
        ("$\\lambda_{sup}$", [f"lsup{v:g}" for v in (0, 1, 5, 10, 25)], [0, 1, 5, 10, 25]),
        ("$\\gamma_{nash}$", [f"gnash{v:g}" for v in (0, .3, .5, .7, .9)], [0, .3, .5, .7, .9]),
    ]
    groups = [g for g in groups if any(n in rows for n in g[1])]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 2.9))
    axes = np.atleast_1d(axes)
    for ax, (title, names, xs) in zip(axes, groups):
        mu, se, xv = [], [], []
        for n, x in zip(names, xs):
            if n not in rows:
                continue
            v = np.array([m["rmse"] for m in rows[n]], float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            mu.append(v.mean())
            se.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
            xv.append(x)
        ax.errorbar(xv, mu, yerr=np.array(se) * 1.96, marker="o", capsize=3, lw=1.4)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("RMSE", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(HERE), "runs"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "paper_assets"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if os.path.isdir(os.path.join(a.runs, "ablation")):
        table_ablation(a.runs, "ablation", "full",
                       os.path.join(a.out, "table_ablation.tex"),
                       "Component ablation of the MD-WGAN framework on NGSIM.",
                       "ablation")
        fig_convergence(a.runs, "ablation",
                        ["full", "no-rl", "no-game", "no-ensemble", "none"],
                        os.path.join(a.out, "fig_ablation_convergence.pdf"),
                        "Effect of removing each component")
        fig_tradeoff(a.runs, "ablation", os.path.join(a.out, "fig_tradeoff.pdf"))

    if os.path.isdir(os.path.join(a.runs, "baseline")):
        table_ablation(a.runs, "baseline", "sel-dqn",
                       os.path.join(a.out, "table_selection.tex"),
                       "Discriminator-selection strategies under an otherwise "
                       "identical framework.", "selection")
        fig_convergence(a.runs, "baseline",
                        ["sel-dqn", "sel-greedy", "sel-random", "sel-roundrobin", "sel-all"],
                        os.path.join(a.out, "fig_selection_convergence.pdf"),
                        "Discriminator selection strategies")

    if os.path.isdir(os.path.join(a.runs, "sensitivity")):
        fig_sensitivity(a.runs, os.path.join(a.out, "fig_sensitivity.pdf"))

    print("wrote assets to", a.out)
    for f in sorted(os.listdir(a.out)):
        print("  ", f)


if __name__ == "__main__":
    main()
