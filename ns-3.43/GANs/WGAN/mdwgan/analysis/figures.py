#!/usr/bin/env python3
"""Paper figures, styled for IEEE Internet of Things Journal.

Vector PDF at the journal's column widths, Times-metric serif, embedded fonts.
Figures are drawn at final size so LaTeX never has to scale them (scaling a
figure rescales its type and breaks the match with the body text).
"""
import csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ieee_style import apply, panel_label, COL1, COL2, C, ACCENT, MARKERS, INK, MUTED  # noqa

apply()
import matplotlib.pyplot as plt   # noqa: E402

ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "paper_assets")
os.makedirs(OUT, exist_ok=True)
TH = 0.86

LBL = {
    "ours": "Proposed", "ours-dqn": "Proposed (DQN)", "md-gan": "MD-GAN",
    "hvs-gan": "HVS-GAN", "fedavg-gan": "FedAvg-GAN", "full": "Full",
    "no-rl": "w/o RL selection", "no-game": "w/o game coordination",
    "no-ensemble": "w/o ensemble", "none": "All components removed",
    "unconditional": "w/o context", "static-game": "Static game",
    "supervised-only": "Supervised only", "no-supervised": "Adversarial only",
}


def load(suite):
    d = {}
    for f in glob.glob(os.path.join(RUNS, suite, "*.json")):
        n, s = os.path.basename(f)[:-5].split("__seed")
        d.setdefault(n, {})[int(s)] = json.load(open(f))
    return d


def curve(runs, key):
    ev = [r["evals"] for r in runs.values()]
    x = np.array([e["round"] for e in ev[0]])
    m = np.array([[e.get(key, np.nan) for e in s] for s in ev])
    mu = np.nanmean(m, 0)
    se = np.nanstd(m, 0, ddof=1) / np.sqrt(len(ev)) if len(ev) > 1 else np.zeros_like(mu)
    return x, mu, 1.96 * se


def finals(runs, key):
    return np.array([r["evals"][-1].get(key, np.nan) for r in runs.values()], float)


def conv(runs):
    r_, w_ = [], []
    for r in runs.values():
        hit = [e["round"] for e in r["evals"] if e["rmse"] <= TH]
        if hit:
            r_.append(hit[0])
            w_.append(sum(x["cpu_usage"] for x in r["rounds"] if x["round"] <= hit[0]))
    return r_, w_


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


# --- Fig 1: convergence vs baselines ---------------------------------------
def fig_convergence():
    d = load("sota")
    arms = [a for a in ("ours", "ours-dqn", "md-gan", "hvs-gan") if a in d]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35))
    for ax, key, ylab, pl in ((axes[0], "rmse", "RMSE", "(a)"),
                              (axes[1], "crps", "CRPS", "(b)")):
        for i, a in enumerate(arms):
            x, mu, ci = curve(d[a], key)
            ax.plot(x, mu, color=C[i], marker=MARKERS[i],
                    markevery=(i * 2, 10), label=LBL.get(a, a))
            ax.fill_between(x, mu - ci, mu + ci, color=C[i], alpha=0.13, lw=0)
        ax.set_xlabel("Training round")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 61)
        panel_label(ax, pl, dy=-0.21)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig1_convergence")


# --- Fig 2: cost of reaching a target --------------------------------------
def fig_cost():
    d = load("sota")
    arms = [a for a in ("ours", "ours-dqn", "md-gan", "hvs-gan", "fedavg-gan") if a in d]
    names, rr, ww, frac = [], [], [], []
    for a in arms:
        c, w = conv(d[a])
        names.append(LBL.get(a, a))
        rr.append(np.mean(c) if c else 0.0)
        ww.append(np.mean(w) if w else 0.0)
        frac.append(len(c) / len(d[a]))
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2))
    for ax, v, xl, pl in ((axes[0], rr, "Rounds to RMSE $\\leq$ %.2f" % TH, "(a)"),
                          (axes[1], ww, "Compute to target (grad-step units)", "(b)")):
        bars = ax.barh(y, v, height=0.62, color=C[0], edgecolor="white", linewidth=0.8)
        for b, val, f in zip(bars, v, frac):
            if val == 0:
                ax.text(max(v) * 0.015, b.get_y() + b.get_height() / 2,
                        "not reached", va="center", fontsize=7, color=ACCENT, style="italic")
            else:
                t = f"{val:.0f}" + ("" if f == 1 else f" ({f*100:.0f}% of seeds)")
                ax.text(val + max(v) * 0.02, b.get_y() + b.get_height() / 2,
                        t, va="center", fontsize=7, color=INK)
        ax.set_yticks(y); ax.set_yticklabels(names)
        ax.set_xlabel(xl); ax.set_xlim(0, max(v) * 1.38)
        ax.grid(axis="y", visible=False)
        panel_label(ax, pl, dy=-0.20)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig2_cost_to_target")


# --- Fig 3: ablation deltas -------------------------------------------------
def fig_ablation():
    d = load("ablation")
    if "full" not in d:
        return
    arms = [a for a in ("unconditional", "none", "no-supervised", "no-game",
                        "no-ensemble", "static-game", "no-rl", "supervised-only")
            if a in d]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for ax, key, xl, pl in ((axes[0], "rmse", "$\\Delta$RMSE", "(a)"),
                            (axes[1], "crps", "$\\Delta$CRPS", "(b)")):
        vals, errs = [], []
        for a in arms:
            ss = sorted(set(d[a]) & set(d["full"]))
            diff = np.array([d[a][s]["evals"][-1][key] - d["full"][s]["evals"][-1][key]
                             for s in ss])
            vals.append(diff.mean())
            errs.append(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))
        order = np.argsort(vals)
        y = np.arange(len(arms))
        ax.barh(y, [vals[i] for i in order], xerr=[errs[i] for i in order],
                height=0.62, color=C[2], edgecolor="white", linewidth=0.8,
                error_kw=dict(ecolor=MUTED, lw=0.7, capsize=1.8))
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([LBL.get(arms[i], arms[i]) for i in order])
        ax.set_xlabel(xl + " relative to full framework")
        ax.grid(axis="y", visible=False)
        panel_label(ax, pl, dy=-0.19)
    fig.subplots_adjust(wspace=0.75)
    save(fig, "fig3_ablation")


# --- Fig 4: calibration ------------------------------------------------------
def fig_calibration():
    d = load("diversity")
    order = [("div-none", "none"), ("div-ms0.1", "MS 0.1"), ("div-ms0.5", "MS 0.5"),
             ("div-ms1", "MS 1.0"), ("div-bok-k4", "BoK 4"),
             ("div-bok", "BoK 8"), ("div-bok-k16", "BoK 16")]
    order = [(a, l) for a, l in order if a in d]
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.1))
    spec = [("picp", "PICP", "(a)", True), ("crps", "CRPS", "(b)", False),
            ("min_ade", "minADE", "(c)", False)]
    for ax, (key, ylab, pl, zero) in zip(axes, spec):
        v = np.array([finals(d[a], key).mean() for a, _ in order])
        e = np.array([1.96 * finals(d[a], key).std(ddof=1) / np.sqrt(len(d[a]))
                      for a, _ in order])
        ax.bar(x, v, yerr=e, width=0.66, color=C[0], edgecolor="white", linewidth=0.7,
               error_kw=dict(ecolor=MUTED, lw=0.7, capsize=1.6))
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in order], rotation=50, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        if zero:
            ax.axhline(0.90, color=ACCENT, ls=(0, (3, 2)), lw=0.9)
            ax.text(len(order) - 0.45, 0.915, "ideal", ha="right", fontsize=6.5,
                    color=ACCENT)
            ax.set_ylim(0, 1.0)
        else:
            lo, hi = (v - e).min(), (v + e).max()
            pad = (hi - lo) * 0.30
            ax.set_ylim(lo - pad, hi + pad)
        panel_label(ax, pl, dy=-0.40)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig4_calibration")


# --- Fig 5: accuracy / cost trade-off ---------------------------------------
def fig_pareto():
    pts = {}
    for suite in ("sota", "candidates", "ablation", "diversity", "baseline"):
        for a, runs in load(suite).items():
            _, w = conv(runs)
            if not w:
                continue
            pts.setdefault(a, (np.mean(w), finals(runs, "crps").mean()))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    xs = np.array([v[0] for v in pts.values()])
    ys = np.array([v[1] for v in pts.values()])
    ax.scatter(xs, ys, s=10, color=MUTED, alpha=0.45, lw=0, zorder=2,
               label="Other configurations")
    o = np.argsort(xs)
    fx, fy, best = [], [], np.inf
    for i in o:
        if ys[i] < best:
            best = ys[i]; fx.append(xs[i]); fy.append(ys[i])
    ax.step(fx, fy, where="post", color=C[2], lw=1.0, zorder=3,
            label="Pareto frontier")
    hi = [("ours", C[0], "Proposed", (7, -2)),
          ("ours-dqn", C[3], "Proposed (DQN)", (7, 3)),
          ("md-gan", C[1], "MD-GAN", (-6, 6))]
    for a, col, lab, off in hi:
        if a not in pts:
            continue
        w, cr = pts[a]
        ax.scatter([w], [cr], s=46, marker="*", color=col, zorder=5,
                   edgecolor="white", linewidth=0.5)
        ax.annotate(lab, (w, cr), fontsize=7, color=INK, zorder=6,
                    xytext=off, textcoords="offset points",
                    ha="right" if off[0] < 0 else "left")
    ax.set_xlabel("Compute to reach target (grad-step units)")
    ax.set_ylabel("CRPS")
    ax.margins(x=0.18, y=0.14)
    ax.legend(loc="upper right", fontsize=6.5)
    save(fig, "fig5_pareto")


# --- Fig 6: sensitivity ------------------------------------------------------
def fig_sensitivity():
    d = load("sensitivity")
    groups = [("Subset size $k$", [f"k{i}" for i in (1, 2, 3, 4, 5)], [1, 2, 3, 4, 5]),
              ("$\\lambda_{\\mathrm{sup}}$", [f"lsup{v:g}" for v in (0, 1, 5, 10, 25)],
               [0, 1, 5, 10, 25]),
              ("$\\gamma_{\\mathrm{nash}}$", [f"gnash{v:g}" for v in (0, .3, .5, .7, .9)],
               [0, .3, .5, .7, .9])]
    groups = [g for g in groups if sum(a in d for a in g[1]) >= 2]
    if not groups:
        return
    fig, axes = plt.subplots(2, len(groups), figsize=(COL2, 3.3), sharex="col")
    axes = np.atleast_2d(axes)
    for j, (title, names, xv) in enumerate(groups):
        for r, (key, ylab) in enumerate((("crps", "CRPS"), ("picp", "PICP"))):
            ax = axes[r, j]
            X, M, E = [], [], []
            for a, x in zip(names, xv):
                if a not in d:
                    continue
                v = finals(d[a], key)
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                X.append(x); M.append(v.mean())
                E.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
            ax.errorbar(X, M, yerr=E, marker=MARKERS[j], color=C[j],
                        capsize=1.8, elinewidth=0.7, markersize=3.0)
            ax.set_ylabel(ylab)
            if r == 1:
                ax.set_xlabel(title)
                if key == "picp":
                    ax.axhline(0.90, color=ACCENT, ls=(0, (3, 2)), lw=0.9)
                    ax.set_ylim(0, 1.0)
        panel_label(axes[1, j], "(%s)" % "abc"[j], dy=-0.42)
    fig.subplots_adjust(wspace=0.34, hspace=0.20, bottom=0.17)
    save(fig, "fig6_sensitivity")


# --- Fig 7: ns-3 communication scalability ----------------------------------
def fig_ns3():
    base = os.path.join(ROOT, "ns3_runs")
    if not os.path.isdir(base):
        return

    def agg(tag):
        lat, dr = [], []
        for f in glob.glob(os.path.join(base, f"{tag}_seed*.csv")):
            rows = list(csv.DictReader(open(f)))
            if not rows:
                continue
            l = [float(r["latency_s"]) * 1000 for r in rows if float(r["latency_s"]) > 0]
            e = sum(int(r["replies_expected"]) for r in rows)
            g = sum(int(r["replies_received"]) for r in rows)
            if l:
                lat.append(np.median(l))
            if e:
                dr.append(100 * g / e)
        return lat, dr

    panels = [("Vehicles $K$", [f"nveh{n}" for n in (8, 16, 24, 32, 40)], [8, 16, 24, 32, 40]),
              ("Selected subset $k$", [f"k{n}" for n in (1, 2, 3, 4, 5)], [1, 2, 3, 4, 5]),
              ("RSUs $N_G$", [f"nrsu{n}" for n in (2, 4, 6, 8)], [2, 4, 6, 8])]
    fig, axes = plt.subplots(2, len(panels), figsize=(COL2, 3.3), sharex="col")
    for j, (title, tags, xv) in enumerate(panels):
        X, L, LE, D, DE = [], [], [], [], []
        for t, x in zip(tags, xv):
            lat, dr = agg(t)
            if not lat:
                continue
            X.append(x)
            L.append(np.mean(lat))
            LE.append(1.96 * np.std(lat, ddof=1) / np.sqrt(len(lat)) if len(lat) > 1 else 0)
            D.append(np.mean(dr) if dr else np.nan)
            DE.append(1.96 * np.std(dr, ddof=1) / np.sqrt(len(dr)) if len(dr) > 1 else 0)
        axes[0, j].errorbar(X, L, yerr=LE, marker=MARKERS[j], color=C[j],
                            capsize=1.8, elinewidth=0.7, markersize=3.0)
        axes[0, j].set_ylabel("Round latency (ms)")
        axes[1, j].errorbar(X, D, yerr=DE, marker=MARKERS[j], color=C[2],
                            capsize=1.8, elinewidth=0.7, markersize=3.0)
        axes[1, j].set_ylabel("Delivery ratio (%)")
        axes[1, j].set_xlabel(title)
        axes[1, j].set_ylim(0, 100)
        panel_label(axes[1, j], "(%s)" % "abc"[j], dy=-0.42)
    fig.subplots_adjust(wspace=0.36, hspace=0.20, bottom=0.17)
    save(fig, "fig7_ns3_scalability")


if __name__ == "__main__":
    for fn in (fig_convergence, fig_cost, fig_ablation, fig_calibration,
               fig_pareto, fig_sensitivity, fig_ns3):
        try:
            fn(); print("ok  ", fn.__name__)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("FAIL", fn.__name__, e)
