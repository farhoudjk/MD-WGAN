#!/usr/bin/env python3
"""Figures and tables for the objective-adaptive results.

Everything here is computed from the per-seed run records in mdwgan/runs; the
numbers in docs/SCENARIO.txt, ADAPT.txt and KNOBS.txt are the same quantities
printed as text.  Normalisation of the Eq. (8) terms is min-max over the pooled
set of runs being compared, exactly as in analysis/scenario_table.py.
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from ieee_style import apply, panel_label, COL1, COL2, C, ACCENT, MARKERS, INK, MUTED  # noqa
apply()
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "paper_assets")
os.makedirs(OUT, exist_ok=True)

ALPHA = {
    "balanced": (0.20, 0.20, 0.20, 0.20, 0.20),
    "accuracy": (0.40, 0.15, 0.15, 0.15, 0.15),
    "delay":    (0.15, 0.15, 0.15, 0.40, 0.15),
    "network":  (0.15, 0.15, 0.15, 0.15, 0.40),
    "resource": (0.14, 0.29, 0.29, 0.14, 0.14),
}
SCEN = list(ALPHA)
SCEN_LBL = {"balanced": "Balanced", "accuracy": "Accuracy", "delay": "Delay",
            "network": "Network", "resource": "Resource"}
FIXED = [("MD-GAN", "sota", "md-gan"), ("Early-update MD-GAN", "sota", "early-update-mdgan"),
         ("HVS-GAN", "sota", "hvs-gan"),
         ("FedAvg-GAN", "sota", "fedavg-gan"), ("Proposed (fixed, DQN)", "sota", "ours-dqn"),
         ("Proposed (fixed)", "sota", "ours")]
TERM_LBL = ["Training (CRPS)", "CPU", "Memory", "Latency", "Traffic"]


def terms(run):
    rd = run["rounds"]; tail = rd[int(len(rd) * 0.8):] or rd
    m = lambda k: float(np.mean([r[k] for r in tail]))
    return np.array([run["evals"][-1].get("crps", np.nan), m("cpu_usage"),
                     m("memory_usage"), m("latency"), m("network_traffic")])


def load_terms(suite, stem):
    out = {}
    for f in glob.glob(os.path.join(RUNS, suite, f"{stem}__seed*.json")):
        s = int(os.path.basename(f)[:-5].split("__seed")[1])
        out[s] = terms(json.load(open(f)))
    return out


def load_runs(suite, stem):
    out = {}
    for f in glob.glob(os.path.join(RUNS, suite, f"{stem}__seed*.json")):
        s = int(os.path.basename(f)[:-5].split("__seed")[1])
        out[s] = json.load(open(f))
    return out


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); plt.close(fig)


def ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


# --------------------------------------------------------------------------
# Scenario scoring (shared by fig8, fig12, tab5/6)
# --------------------------------------------------------------------------
def scenario_data():
    fixed = {lbl: load_terms(su, st) for lbl, su, st in FIXED}
    adapt = {(s, m): load_terms("adaptive", f"adapt-{s}-{m}")
             for s in SCEN for m in ("none", "static", "online")}
    adapt = {k: v for k, v in adapt.items() if v}
    pool = [t for d in fixed.values() for t in d.values()] + \
           [t for (s, m), d in adapt.items() if m != "none" for t in d.values()]
    P = np.array(pool, float)
    lo, hi = np.nanmin(P, 0), np.nanmax(P, 0)
    rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    nz = lambda t: (t - lo) / rng
    return fixed, adapt, nz


def pick_variant(adapt, nz, s):
    cands = {}
    for m in ("static", "online"):
        if (s, m) in adapt:
            cands[m] = {sd: float(np.dot(ALPHA[s], nz(t))) for sd, t in adapt[(s, m)].items()}
    best = min(cands, key=lambda m: np.mean(list(cands[m].values())))
    return best, cands[best]


# --- Fig 8: Eq.(8) cost per method per scenario ----------------------------
def fig_scenario_cost():
    fixed, adapt, nz = scenario_data()
    labels = [l for l, _, _ in FIXED] + ["Proposed (adaptive)"]
    cols = [C[1], "#e8a86b", C[3], MUTED, "#9bb4de", C[0], C[2]]
    nS, nM = len(SCEN), len(labels)
    fig, ax = plt.subplots(figsize=(COL2, 2.5))
    w = 0.8 / nM
    for j, lbl in enumerate(labels):
        means, errs = [], []
        for s in SCEN:
            if lbl == "Proposed (adaptive)":
                _, c = pick_variant(adapt, nz, s)
            else:
                c = {sd: float(np.dot(ALPHA[s], nz(t))) for sd, t in fixed[lbl].items()}
            v = np.array(list(c.values()))
            means.append(v.mean()); errs.append(ci(v))
        x = np.arange(nS) + (j - (nM - 1) / 2) * w
        ax.bar(x, means, w * 0.95, yerr=errs, color=cols[j], label=lbl,
               edgecolor="white", linewidth=0.5,
               error_kw=dict(ecolor=MUTED, lw=0.6, capsize=1.4))
    ax.set_xticks(np.arange(nS)); ax.set_xticklabels([SCEN_LBL[s] + "-weighted" for s in SCEN])
    ax.set_ylabel("Normalised objective cost $J(\\boldsymbol{\\alpha})$")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
    save(fig, "fig8_scenario_cost")


# --- Fig 9: none / static / online per scenario ----------------------------
def fig_adapt_modes():
    fixed, adapt, nz = scenario_data()
    modes = ["none", "static", "online"]
    lbl = {"none": "Fixed configuration", "static": "Static policy", "online": "Online controller"}
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    w = 0.26
    for j, m in enumerate(modes):
        means, errs = [], []
        for s in SCEN:
            c = [float(np.dot(ALPHA[s], nz(t))) for t in adapt[(s, m)].values()]
            means.append(np.mean(c)); errs.append(ci(c))
        x = np.arange(len(SCEN)) + (j - 1) * w
        ax.bar(x, means, w * 0.95, yerr=errs, color=[MUTED, C[0], C[2]][j], label=lbl[m],
               edgecolor="white", linewidth=0.5,
               error_kw=dict(ecolor=MUTED, lw=0.6, capsize=1.4))
    ax.set_xticks(np.arange(len(SCEN))); ax.set_xticklabels([SCEN_LBL[s] for s in SCEN])
    ax.set_ylabel("Objective cost $J(\\boldsymbol{\\alpha})$")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.2), frameon=False,
              fontsize=6.5, columnspacing=0.9, handlelength=1.2)
    save(fig, "fig9_adapt_modes")


# --- Fig 10: knob ablation ---------------------------------------------------
def fig_knobs():
    data = {}
    for f in glob.glob(os.path.join(RUNS, "knobs", "knob-*.json")):
        stem, seed = os.path.basename(f)[:-5].split("__seed")
        _, scen, knob = stem.split("-", 2)
        data.setdefault((scen, knob), {})[int(seed)] = terms(json.load(open(f)))
    P = np.array([v for d in data.values() for v in d.values()], float)
    lo, hi = np.nanmin(P, 0), np.nanmax(P, 0)
    rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    knobs = ["none", "top_k", "E_G", "sync_every", "lambda_sup", "all"]
    kl = {"none": "None (fixed)", "top_k": "$k$", "E_G": "$E_G$", "sync_every": "$S$",
          "lambda_sup": "$\\lambda_{\\rm sup}$", "all": "All four"}
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2))
    for ax, scen, pl in ((axes[0], "network", "(a)"), (axes[1], "resource", "(b)")):
        a = ALPHA[scen]; means, errs = [], []
        for k in knobs:
            c = [float(np.dot(a, (t - lo) / rng)) for t in data[(scen, k)].values()]
            means.append(np.mean(c)); errs.append(ci(c))
        x = np.arange(len(knobs))
        colors = [MUTED] + [C[0]] * 4 + [C[2]]
        ax.bar(x, means, 0.66, yerr=errs, color=colors, edgecolor="white", linewidth=0.6,
               error_kw=dict(ecolor=MUTED, lw=0.6, capsize=1.6))
        ax.set_xticks(x); ax.set_xticklabels([kl[k] for k in knobs], rotation=35, ha="right")
        ax.set_ylabel("Objective cost $J(\\boldsymbol{\\alpha})$")
        ax.set_title(f"{SCEN_LBL[scen]}-weighted $\\boldsymbol{{\\alpha}}$", fontsize=8)
        ax.grid(axis="x", visible=False)
        panel_label(ax, pl, dy=-0.46)
    fig.subplots_adjust(wspace=0.35, bottom=0.3)
    save(fig, "fig10_knobs")


# --- Fig 11: controller trace, network scenario ----------------------------
def fig_trace(scen="network"):
    runs = {m: load_runs("adaptive", f"adapt-{scen}-{m}") for m in ("none", "static", "online")}
    lbl = {"none": "Fixed", "static": "Static", "online": "Online"}
    cols = {"none": MUTED, "static": C[0], "online": C[2]}
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2))
    for m, d in runs.items():
        # per-round quantities, generator 0, averaged over seeds
        def series(key, fn=None):
            fn = fn or (lambda r: r[key])
            arr = []
            for r in d.values():
                rd = [x for x in r["rounds"] if x["generator"] == 0]
                arr.append([fn(x) for x in rd])
            return np.arange(1, len(arr[0]) + 1), np.mean(arr, 0)
        x, eg = series("E_G"); _, kk = series("subset", lambda r: len(r["subset"]))
        axes[0].step(x, eg, where="mid", color=cols[m], label=lbl[m], lw=1.0)
        axes[0].step(x, kk, where="mid", color=cols[m], ls=(0, (2, 1.5)), lw=1.0)
        _, tr = series("network_traffic", lambda r: r["network_traffic"] / 1e6)
        axes[1].plot(x, np.cumsum(tr), color=cols[m], label=lbl[m], lw=1.0)
        ev = np.array([[e["crps"] for e in r["evals"]] for r in d.values()])
        xe = np.array([e["round"] for e in next(iter(d.values()))["evals"]])
        mu = ev.mean(0); e = 1.96 * ev.std(0, ddof=1) / np.sqrt(len(ev))
        axes[2].plot(xe, mu, color=cols[m], label=lbl[m], lw=1.0)
        axes[2].fill_between(xe, mu - e, mu + e, color=cols[m], alpha=0.13, lw=0)
    axes[0].set_ylabel("$E_G$ (solid), $k$ (dashed)"); axes[0].set_xlabel("Round")
    axes[1].set_ylabel("Cumulative traffic (MB)"); axes[1].set_xlabel("Round")
    axes[2].set_ylabel("CRPS"); axes[2].set_xlabel("Round")
    for ax, pl in zip(axes, ("(a)", "(b)", "(c)")):
        panel_label(ax, pl, dy=-0.30)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.subplots_adjust(wspace=0.42, bottom=0.25)
    save(fig, "fig11_controller_trace")


# --- Fig 12: weighted cost decomposition -----------------------------------
def fig_breakdown():
    fixed, adapt, nz = scenario_data()
    labels = [l for l, _, _ in FIXED] + ["Proposed (adaptive)"]
    short = ["MD-GAN", "Early-upd.", "HVS-GAN", "FedAvg", "Ours (DQN)", "Ours (fixed)", "Ours (adapt.)"]
    tcols = [C[0], C[1], C[2], C[3], "#7f7f7f"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))
    for ax, s, pl in ((axes[0], "balanced", "(a)"), (axes[1], "network", "(b)")):
        a = np.array(ALPHA[s]); bottoms = np.zeros(len(labels))
        for ti in range(5):
            vals = []
            for lbl in labels:
                if lbl == "Proposed (adaptive)":
                    m, _ = pick_variant(adapt, nz, s); d = adapt[(s, m)]
                else:
                    d = fixed[lbl]
                vals.append(np.mean([a[ti] * nz(t)[ti] for t in d.values()]))
            ax.bar(np.arange(len(labels)), vals, 0.66, bottom=bottoms, color=tcols[ti],
                   label=TERM_LBL[ti], edgecolor="white", linewidth=0.4)
            bottoms += np.array(vals)
        ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(short, rotation=35, ha="right")
        ax.set_ylabel("Weighted contribution to $J$")
        ax.set_title(f"{SCEN_LBL[s]}-weighted $\\boldsymbol{{\\alpha}}$", fontsize=8)
        ax.grid(axis="x", visible=False)
        panel_label(ax, pl, dy=-0.50)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.subplots_adjust(wspace=0.35, bottom=0.32)
    save(fig, "fig12_cost_breakdown")


# --- Fig 2: workflow block diagram -----------------------------------------
def fig_workflow():
    fig, ax = plt.subplots(figsize=(COL1, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    def box(x, y, w, h, title, body, fc="#f3f5fa", ec=C[0], num=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center", fontsize=7.2,
                weight="bold", color=INK)
        ax.text(x + w / 2, y + 0.42, body, ha="center", va="center", fontsize=6.2, color=MUTED)
        if num is not None:
            ax.add_patch(plt.Circle((x + 0.3, y + h - 0.3), 0.24, fc=ec, ec="none"))
            ax.text(x + 0.3, y + h - 0.3, str(num), ha="center", va="center", fontsize=6.2,
                    color="white", weight="bold")
    def arrow(p, q, label=None, col=INK, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                     color=col, ls=ls, connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, label, fontsize=5.8, color=col,
                    ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.6))
    # Step 0 controller (top, full width)
    box(0.5, 8.0, 9.0, 1.6, "Objective-adaptive configuration",
        "$\\boldsymbol{\\alpha}\;\\to\;(k,\\,E_G,\\,S,\\,\\lambda_{\\rm sup})$ by cost-direction projection; online dual update",
        fc="#eaf0fb", ec=C[0], num=0)
    # Steps 1-4
    box(0.5, 5.3, 4.2, 1.7, "Subset selection", "DQN / heuristic rule\nactivates $k$ critics", num=1)
    box(5.3, 5.3, 4.2, 1.7, "Epoch allocation", "best-response $(E_D^\\star, E_G^\\star)$\n$E_G$ overridden by step 0", num=2)
    box(5.3, 2.6, 4.2, 1.7, "Distributed training", "critics: WGAN-GP; generator:\nensemble + best-of-$k_s$ supervision", num=3)
    box(0.5, 2.6, 4.2, 1.7, "Monitoring", "realised $\\tilde{\\mathcal{L}},\\tilde{C},\\tilde{M},\\tilde{L},\\tilde{T}$\nconvergence test", num=4)
    arrow((5.0, 8.0), (5.0, 7.0), col=C[0])
    arrow((4.7, 6.15), (5.3, 6.15))
    arrow((7.4, 5.3), (7.4, 4.3))
    arrow((5.3, 3.45), (4.7, 3.45))
    arrow((2.6, 4.3), (2.6, 5.3), label="next round")
    arrow((0.5, 3.45), (0.5, 8.0), label="realised terms\n(online only)", col=C[2], ls=(0, (3, 2)), rad=0.0)
    ax.text(5.0, 1.75, "repeat until $|\\mathcal{L}^{t}-\\mathcal{L}^{t-1}|<\\varepsilon$ for generator and critics",
            ha="center", fontsize=6.2, color=MUTED, style="italic")
    ax.text(5.0, 0.9, "Input: deployment weights $\\boldsymbol{\\alpha}$ of the objective", ha="center",
            fontsize=6.4, color=INK)
    save(fig, "fig2_workflow")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def tab_config():
    """Configuration the static policy deploys for each scenario (base = the
    trainer's defaults: k=3, E_G=5, S=1, lambda_sup=10), and what the online
    variant realised on average."""
    from core.adaptive import StaticPolicy
    base = dict(top_k=3, E_G=5, sync_every=1, lambda_sup=10.0)
    rows = []
    for s in SCEN:
        c = StaticPolicy(ALPHA[s], base).configure()
        on = load_runs("adaptive", f"adapt-{s}-online")
        kk = np.mean([len(x["subset"]) for r in on.values() for x in r["rounds"]])
        eg = np.mean([x["E_G"] for r in on.values() for x in r["rounds"]])
        rows.append((s, c, kk, eg))
    L = []
    L.append("% Generated by analysis/figures_adaptive.py")
    L.append("\\begin{table}[!t]\\centering")
    L.append("\\caption{Configuration selected by the static policy of Eq.~(\\ref{eq:policy}) under each deployment weighting, from the fixed-configuration baseline $(k,E_G,S,\\lambda_{\\rm sup})=(3,5,1,10)$, and the mean $(k,E_G)$ realised by the online controller over 60 rounds.}")
    L.append("\\label{tab:config}")
    L.append("\\begin{tabular}{lcccccc}\\toprule")
    L.append("Weighting & $k$ & $E_G$ & $S$ & $\\lambda_{\\rm sup}$ & Supervision & Online $\\bar{k}/\\bar{E}_G$ \\\\\\midrule")
    for s, c, kk, eg in rows:
        sup = "best-of-$k_s$" if c["sup_mode"] == "best_of_k" else "mean"
        L.append(f"{SCEN_LBL[s]} & {c['top_k']} & {c['E_G']} & {c['sync_every']} & {c['lambda_sup']:.1f} & {sup} & {kk:.1f}\\,/\\,{eg:.1f} \\\\")
    L.append("\\bottomrule\\end{tabular}\\end{table}")
    open(os.path.join(OUT, "tab5_config.tex"), "w").write("\n".join(L) + "\n")


def tab_modes():
    fixed, adapt, nz = scenario_data()
    L = ["% Generated by analysis/figures_adaptive.py",
         "\\begin{table}[!t]\\centering",
         "\\caption{Objective cost $J(\\boldsymbol{\\alpha})$ of the proposed framework with no controller, the static policy and the online controller; each row is scored under its own weighting. Mean over five seeds; $p$ is a paired Wilcoxon test of the better adaptive variant against the fixed configuration ($p=0.062$ is the smallest value attainable at $n=5$).}",
         "\\label{tab:modes}", "\\begin{tabular}{lcccrc}\\toprule",
         "Weighting & Fixed & Static & Online & Gain & $p$ \\\\\\midrule"]
    for s in SCEN:
        c = {m: np.array([float(np.dot(ALPHA[s], nz(t))) for _, t in sorted(adapt[(s, m)].items())])
             for m in ("none", "static", "online")}
        best = min(("static", "online"), key=lambda m: c[m].mean())
        gain = 100 * (c["none"].mean() - c[best].mean()) / c["none"].mean()
        p = stats.wilcoxon(c[best], c["none"]).pvalue
        b = lambda m, v: (f"\\textbf{{{v:.3f}}}" if m == best else f"{v:.3f}")
        L.append(f"{SCEN_LBL[s]} & {c['none'].mean():.3f} & {b('static', c['static'].mean())} & {b('online', c['online'].mean())} & {gain:.1f}\\% & {p:.3f} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(OUT, "tab7_modes.tex"), "w").write("\n".join(L) + "\n")


def tab_terms():
    """Raw realised terms (before normalisation) for every row of the scenario table."""
    fixed, adapt, nz = scenario_data()
    def extra(suite, stem):
        runs = load_runs(suite, stem)
        r = np.array([x["evals"][-1]["rmse"] for x in runs.values()])
        p = np.array([x["evals"][-1]["picp"] for x in runs.values()])
        return f"{r.mean():.3f} & {p.mean():.2f}"
    def fmt(d):
        T = np.array(list(d.values()), float)
        m, sd = T.mean(0), T.std(0, ddof=1)
        return (f"{m[0]:.3f}\\,$\\pm$\\,{sd[0]:.3f} & {m[1]:.1f} & {m[2]:.0f} & "
                f"{m[3]:.2f} & {m[4]/1e6:.2f}")
    L = ["% Generated by analysis/figures_adaptive.py",
         "\\begin{table}[!t]\\centering",
         "\\caption{Realised terms behind Table~\\ref{tab:scenario}, before normalisation. CRPS, RMSE and PICP are terminal values; compute (batch-gradient-step units per round), resident memory, round latency and traffic are means over the final 20\\% of rounds. Five seeds throughout.}",
         "\\label{tab:terms}", "\\begin{tabular}{lccccccc}\\toprule",
         "Configuration & CRPS & RMSE & PICP & Compute & Mem.\\ (MB) & Lat.\\ (s) & Traffic (MB) \\\\\\midrule"]
    for lbl, su, st in FIXED:
        a = fmt(fixed[lbl]).split(" & ")
        L.append(f"{lbl} & {a[0]} & {extra(su, st)} & {' & '.join(a[1:])} \\\\")
    L.append("\\midrule")
    for s in SCEN:
        m, _ = pick_variant(adapt, nz, s)
        a = fmt(adapt[(s, m)]).split(" & ")
        L.append(f"Adaptive, {SCEN_LBL[s].lower()} $\\boldsymbol{{\\alpha}}$ ({m}) & {a[0]} & {extra('adaptive', f'adapt-{s}-{m}')} & {' & '.join(a[1:])} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(OUT, "tab6_terms.tex"), "w").write("\n".join(L) + "\n")


def _suite_stats(suite, stems, TH=0.86):
    out = {}
    for st in stems:
        runs = load_runs(suite, st)
        if not runs:
            continue
        fin = lambda k: np.array([r["evals"][-1][k] for r in runs.values()])
        conv, work = [], []
        for r in runs.values():
            hit = [e["round"] for e in r["evals"] if e["rmse"] <= TH]
            if hit:
                conv.append(hit[0]); work.append(sum(x["cpu_usage"] for x in r["rounds"] if x["round"] <= hit[0]))
        tr = np.mean([np.mean([x["network_traffic"] for x in r["rounds"]]) for r in runs.values()]) / 1e6
        out[st] = dict(rmse=fin("rmse"), crps=fin("crps"), picp=fin("picp"), cov=fin("mode_coverage"),
                       conv=conv, work=work, n=len(runs), traffic=tr)
    return out


def tab_partition():
    st = _suite_stats("noniid", ["noniid-segment", "noniid-lane", "noniid-natural", "noniid-shuffled", "noniid-iid-split"])
    lbl = {"noniid-segment": "Road segment (used)", "noniid-lane": "Lane", "noniid-natural": "Natural (per-track)",
           "noniid-shuffled": "Shuffled", "noniid-iid-split": "IID split"}
    L = ["% Generated by analysis/figures_adaptive.py", "\\begin{table}[!t]\\centering",
         "\\caption{Effect of the data partitioning across discriminators. ``Rounds'' and ``Compute'' are the cost of first reaching RMSE\\,$\\leq$\\,0.86, with the fraction of seeds reaching it.}",
         "\\label{tab:partition}", "\\begin{tabular}{lcccc}\\toprule",
         "Partitioning & RMSE $\\downarrow$ & CRPS $\\downarrow$ & Rounds $\\downarrow$ & Compute $\\downarrow$ \\\\\\midrule"]
    for k in ["noniid-segment", "noniid-lane", "noniid-natural", "noniid-shuffled", "noniid-iid-split"]:
        d = st[k]; fr = f" ({len(d['conv'])}/{d['n']})" if len(d["conv"]) < d["n"] else ""
        L.append(f"{lbl[k]} & {d['rmse'].mean():.3f}\\,$\\pm$\\,{d['rmse'].std(ddof=1):.3f} & {d['crps'].mean():.3f}\\,$\\pm$\\,{d['crps'].std(ddof=1):.3f} & {np.mean(d['conv']):.1f}{fr} & {np.mean(d['work']):.0f} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(OUT, "tab8_partition.tex"), "w").write("\n".join(L) + "\n")


def tab_scale():
    st = _suite_stats("sensitivity", ["gens2", "gens4", "gens6", "pool3", "pool5", "pool8", "k1", "k2", "k3", "k4", "k5"])
    rows = [("$N_G=2$ (10 vehicles)", "gens2"), ("$N_G=4$ (20 vehicles)", "gens4"), ("$N_G=6$ (30 vehicles)", "gens6"),
            ("pool 3 per RSU", "pool3"), ("pool 5 per RSU", "pool5"), ("pool 8 per RSU", "pool8"),
            ("$k=1$", "k1"), ("$k=2$", "k2"), ("$k=3$", "k3"), ("$k=4$", "k4"), ("$k=5$", "k5")]
    L = ["% Generated by analysis/figures_adaptive.py", "\\begin{table}[!t]\\centering",
         "\\caption{Scaling the deployment: RSU count $N_G$, local pool size, and active subset $k$ (all else as in Table~\\ref{tab:sim}). Traffic is the mean per-round exchange; ``Compute'' is work to first reach RMSE\\,$\\leq$\\,0.86 with the fraction of seeds reaching it.}",
         "\\label{tab:scale}", "\\begin{tabular}{lcccc}\\toprule",
         "Setting & RMSE $\\downarrow$ & CRPS $\\downarrow$ & Compute $\\downarrow$ & Traffic (MB) \\\\\\midrule"]
    for lbl, k in rows:
        if k not in st:
            continue
        d = st[k]; fr = f" ({len(d['conv'])}/{d['n']})" if len(d["conv"]) < d["n"] else ""
        L.append(f"{lbl} & {d['rmse'].mean():.3f}\\,$\\pm$\\,{d['rmse'].std(ddof=1):.3f} & {d['crps'].mean():.3f}\\,$\\pm$\\,{d['crps'].std(ddof=1):.3f} & {np.mean(d['work']) if d['work'] else float('nan'):.0f}{fr} & {d['traffic']:.1f} \\\\")
        if k in ("gens6", "pool8"):
            L.append("\\midrule")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(OUT, "tab9_scale.tex"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    for fn in (fig_workflow, fig_scenario_cost, fig_adapt_modes, fig_knobs, fig_trace,
               fig_breakdown, tab_config, tab_modes, tab_terms, tab_partition, tab_scale):
        try:
            fn(); print("ok  ", fn.__name__)
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAIL", fn.__name__, e)
