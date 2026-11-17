#!/usr/bin/env python3
"""Score arms on the full metric set, not point error alone.

Accuracy here is four complementary things -- point error (RMSE), distributional
fidelity (CRPS), multi-modal error (minADE) and calibration (PICP) -- plus the
resource terms of Eq. (8).  Any single one of them can be improved at the
expense of the others, so arms are compared on all of them and ranked by
average rank, with the per-metric table shown so the trade-off stays visible.
"""
import glob, json, os, sys
import numpy as np

TH = 0.90
ACC = [("rmse", False), ("crps", False), ("min_ade", False), ("picp", None),
       ("mode_coverage", True)]
RES = [("conv", False), ("work", False), ("latency", False), ("traffic", False)]


def summarise(run_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        name, seed = os.path.basename(f)[:-5].split("__seed")
        d = json.load(open(f))
        ev, rd = d["evals"], d["rounds"]
        last = ev[-1]
        hit = [e["round"] for e in ev if e["rmse"] <= TH]
        conv = hit[0] if hit else np.nan
        work = sum(x["cpu_usage"] for x in rd if not hit or x["round"] <= hit[0])
        tail = rd[int(len(rd) * 0.8):]
        m = {k: last.get(k, np.nan) for k, _ in ACC}
        m.update(conv=conv, work=work,
                 latency=np.mean([x["latency"] for x in tail]),
                 traffic=np.mean([x["network_traffic"] for x in tail]) / 1e6)
        out.setdefault(name, []).append(m)
    return {k: {kk: float(np.nanmean([r[kk] for r in v]))
                for kk in v[0]} | {"n": len(v)} for k, v in out.items()}


def rank_table(S):
    names = list(S)
    cols = ACC + RES
    ranks = {n: [] for n in names}
    for key, hi in cols:
        vals = {n: S[n][key] for n in names if np.isfinite(S[n].get(key, np.nan))}
        if len(vals) < 2:
            continue
        if hi is None:                      # PICP: closeness to nominal 0.90
            order = sorted(vals, key=lambda n: abs(vals[n] - 0.90))
        else:
            order = sorted(vals, key=vals.get, reverse=bool(hi))
        for i, n in enumerate(order):
            ranks[n].append(i)
    return {n: float(np.mean(r)) if r else np.nan for n, r in ranks.items()}


def main(run_dir):
    S = summarise(run_dir)
    if not S:
        print("no runs in", run_dir); return
    R = rank_table(S)
    hdr = f"{'arm':<13}{'RANK':>6}" + "".join(f"{k:>10}" for k, _ in ACC + RES)
    print(hdr); print("-" * len(hdr))
    for n in sorted(R, key=lambda x: R[x]):
        row = f"{n:<13}{R[n]:>6.2f}"
        for k, _ in ACC + RES:
            v = S[n].get(k, np.nan)
            row += f"{v:>10.4g}" if np.isfinite(v) else f"{'--':>10}"
        print(row)
    print(f"\nn seeds: {S[list(S)[0]]['n']}. RANK = mean rank over all "
          f"{len(ACC)+len(RES)} metrics (lower better).")
    print("PICP ranked by distance from nominal 0.90, not by magnitude.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs_setsel"))
