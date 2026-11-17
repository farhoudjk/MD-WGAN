#!/usr/bin/env python3
"""Emit the paper's result tables as LaTeX, straight from the run records."""
import glob, json, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE))
from analysis.stats import paired_test, holm_bonferroni  # noqa
ROOT=os.path.dirname(HERE); RUNS=os.path.join(ROOT,"runs"); OUT=os.path.join(ROOT,"paper_assets")
TH=0.86

def load(s):
    d={}
    for f in glob.glob(os.path.join(RUNS,s,"*.json")):
        n,sd=os.path.basename(f)[:-5].split("__seed"); d.setdefault(n,{})[int(sd)]=json.load(open(f))
    return d

def fin(runs,k): return np.array([r["evals"][-1].get(k,np.nan) for r in runs.values()],float)

def conv(runs):
    r_,w_=[],[]
    for r in runs.values():
        hit=[e["round"] for e in r["evals"] if e["rmse"]<=TH]
        if hit:
            r_.append(hit[0]); w_.append(sum(x["cpu_usage"] for x in r["rounds"] if x["round"]<=hit[0]))
    return r_,w_

def pm(v,p=3):
    v=v[np.isfinite(v)]
    if not len(v): return "--"
    return f"{v.mean():.{p}f}\\,$\\pm$\\,{v.std(ddof=1):.{p}f}"

def stars(p):
    if not np.isfinite(p): return ""
    return "$^{\\ast\\ast\\ast}$" if p<.001 else "$^{\\ast\\ast}$" if p<.01 else "$^{\\ast}$" if p<.05 else ""

# ---- Table I: comparison with baselines -----------------------------------
def tab_sota():
    d=load("sota")
    order=[("ours","\\textbf{Proposed}"),("ours-dqn","Proposed (DQN variant)"),
           ("md-gan","MD-GAN \\cite{hardy2019mdgan}"),("hvs-gan","HVS-GAN \\cite{wu2024hvsgan}"),
           ("fedavg-gan","FedAvg-GAN")]
    L=[r"\begin{table*}[!t]",r"\centering",
       r"\caption{Comparison with federated GAN baselines on NGSIM. Mean\,$\pm$\,s.d. over five seeds. "
       r"Arrows give the preferred direction; PICP is best near its nominal 0.90. "
       r"``Rounds'' and ``Compute'' are the cost of first reaching RMSE\,$\leq$\,0.86, "
       r"with the fraction of seeds that reach it at all.}",
       r"\label{tab:sota}",r"\begin{tabular}{lcccccc}",r"\toprule",
       r"Method & RMSE $\downarrow$ & CRPS $\downarrow$ & minADE $\downarrow$ & PICP $\rightarrow$0.90 "
       r"& Rounds $\downarrow$ & Compute $\downarrow$ \\",r"\midrule"]
    for k,lab in order:
        if k not in d: continue
        c,w=conv(d[k]); n=len(d[k])
        rr="--" if not c else f"{np.mean(c):.1f}"
        ww="--" if not w else f"{np.mean(w):.0f}"
        frac="" if not c else ("" if len(c)==n else f" ({len(c)}/{n})")
        if not c: frac=" (0/%d)"%n
        L.append(" & ".join([lab,pm(fin(d[k],"rmse")),pm(fin(d[k],"crps")),
                             pm(fin(d[k],"min_ade")),pm(fin(d[k],"picp")),
                             rr+frac,ww])+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    open(os.path.join(OUT,"tab_sota.tex"),"w").write("\n".join(L)+"\n")

# ---- Table II: ablation ----------------------------------------------------
def tab_ablation():
    d=load("ablation"); ref="full"
    order=[("full","\\textbf{Full framework}"),("no-rl","w/o RL-based selection"),
           ("no-ensemble","w/o ensemble feedback"),("no-game","w/o game coordination"),
           ("static-game","w/o online adaptation (static game)"),
           ("no-supervised","w/o supervised term"),("supervised-only","w/o adversarial term"),
           ("unconditional","w/o context conditioning"),("none","all components removed")]
    names=[k for k,_ in order if k in d and k!=ref]
    ps=[paired_test({s:d[k][s]["evals"][-1]["crps"] for s in d[k]},
                    {s:d[ref][s]["evals"][-1]["crps"] for s in d[ref]})["p_wilcoxon"] for k in names]
    holm=dict(zip(names,holm_bonferroni(ps)))
    L=[r"\begin{table*}[!t]",r"\centering",
       r"\caption{Component ablation. Mean\,$\pm$\,s.d. over five seeds. Significance is a paired "
       r"Wilcoxon signed-rank test of CRPS against the full framework, Holm-corrected across variants "
       r"($^{\ast}p<0.05$, $^{\ast\ast}p<0.01$, $^{\ast\ast\ast}p<0.001$). Compute is "
       r"gradient-step units spent reaching RMSE\,$\leq$\,0.86.}",
       r"\label{tab:ablation}",r"\begin{tabular}{lccccc}",r"\toprule",
       r"Variant & RMSE $\downarrow$ & CRPS $\downarrow$ & minADE $\downarrow$ & PICP $\rightarrow$0.90 & Compute $\downarrow$ \\",
       r"\midrule"]
    for k,lab in order:
        if k not in d: continue
        _,w=conv(d[k])
        L.append(" & ".join([lab,pm(fin(d[k],"rmse")),pm(fin(d[k],"crps"))+stars(holm.get(k,np.nan)),
                             pm(fin(d[k],"min_ade")),pm(fin(d[k],"picp")),
                             "--" if not w else f"{np.mean(w):.0f}"])+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    open(os.path.join(OUT,"tab_ablation.tex"),"w").write("\n".join(L)+"\n")

# ---- Table III: selection strategies --------------------------------------
def tab_selection():
    d=load("baseline")
    order=[("sel-dqn","DQN (deep RL)"),("sel-greedy","\\textbf{Greedy-by-loss}"),
           ("sel-random","Random-$k$"),("sel-roundrobin","Round-robin"),("sel-all","All discriminators")]
    L=[r"\begin{table}[!t]",r"\centering",
       r"\caption{Discriminator-selection strategies under an otherwise identical framework. "
       r"Mean\,$\pm$\,s.d. over five seeds.}",r"\label{tab:selection}",
       r"\begin{tabular}{lccc}",r"\toprule",
       r"Strategy & RMSE $\downarrow$ & CRPS $\downarrow$ & Compute $\downarrow$ \\",r"\midrule"]
    for k,lab in order:
        if k not in d: continue
        _,w=conv(d[k])
        L.append(" & ".join([lab,pm(fin(d[k],"rmse")),pm(fin(d[k],"crps")),
                             "--" if not w else f"{np.mean(w):.0f}"])+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    open(os.path.join(OUT,"tab_selection.tex"),"w").write("\n".join(L)+"\n")

# ---- Table IV: diversity / calibration -------------------------------------
def tab_diversity():
    d=load("diversity")
    order=[("div-none","None (conditional-mean supervision)"),("div-ms0.1","Mode-seeking, $\\lambda_{\\rm div}=0.1$"),
           ("div-ms0.5","Mode-seeking, $\\lambda_{\\rm div}=0.5$"),("div-ms1","Mode-seeking, $\\lambda_{\\rm div}=1.0$"),
           ("div-bok-k4","Best-of-$k$, $k=4$"),("div-bok","Best-of-$k$, $k=8$"),("div-bok-k16","Best-of-$k$, $k=16$")]
    L=[r"\begin{table}[!t]",r"\centering",
       r"\caption{Effect of the diversity mechanism on predictive calibration. "
       r"Without it the generator ignores its latent input and its nominal 90\% interval "
       r"covers only 26\% of targets.}",r"\label{tab:diversity}",
       r"\begin{tabular}{lccc}",r"\toprule",
       r"Mechanism & PICP $\rightarrow$0.90 & CRPS $\downarrow$ & minADE $\downarrow$ \\",r"\midrule"]
    for k,lab in order:
        if k not in d: continue
        L.append(" & ".join([lab,pm(fin(d[k],"picp")),pm(fin(d[k],"crps")),pm(fin(d[k],"min_ade"))])+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    open(os.path.join(OUT,"tab_diversity.tex"),"w").write("\n".join(L)+"\n")

if __name__=="__main__":
    for f in (tab_sota,tab_ablation,tab_selection,tab_diversity):
        f(); print("ok",f.__name__)
