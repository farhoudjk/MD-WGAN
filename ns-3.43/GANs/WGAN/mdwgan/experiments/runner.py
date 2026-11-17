#!/usr/bin/env python3
"""Parallel, resumable runner for the MD-WGAN experiment matrix.

Each (arm, seed) pair is one independent job writing a single JSON result, so a
crashed or interrupted sweep resumes without redoing completed work and every
reported number is traceable to one file.
"""
import argparse
import json
import time
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace, asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from core.trainer import Config, run_experiment          # noqa: E402
from experiments.suite import SUITES                      # noqa: E402


def job_path(out_root, suite, name, seed):
    return os.path.join(out_root, suite, f"{name}__seed{seed}.json")


def run_one(args):
    suite, cfg_dict, seed, out_root, overwrite = args
    cfg = Config(**cfg_dict)
    cfg = replace(cfg, seed=seed)
    path = job_path(out_root, suite, cfg.name, seed)
    if os.path.exists(path) and not overwrite:
        return ("skip", cfg.name, seed, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        # A heartbeat file makes a long sweep observable: it records which job a
        # worker picked up and when, so a slow arm is distinguishable from a hang.
        hb = os.path.join(out_root, "_progress")
        os.makedirs(hb, exist_ok=True)
        with open(os.path.join(hb, f"{cfg.name}__seed{seed}.start"), "w") as f:
            f.write(f"{time.time()} pid={os.getpid()}\n")
        os.environ["MDWGAN_HEARTBEAT"] = os.path.join(
            hb, f"{cfg.name}__seed{seed}.progress")
        t0 = time.time()
        res = run_experiment(cfg)
        res["job_seconds"] = time.time() - t0
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, path)   # atomic: a reader never sees a partial file
        return ("ok", cfg.name, seed, res["final"].get("rmse"))
    except Exception:
        return ("fail", cfg.name, seed, traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=["ablation", "baseline"],
                    choices=sorted(SUITES) + ["all"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--generators", type=int, default=4)
    ap.add_argument("--disc-per-gen", type=int, default=5)
    ap.add_argument("--max-tracks", type=int, default=400)
    ap.add_argument("--noniid", default="segment",
                    help="shard scheme: segment|lane|iid-split|natural|shuffled")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "runs"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.environ["MDWGAN_THREADS"] = str(a.threads)
    suites = sorted(SUITES) if "all" in a.suites else a.suites

    base = Config(rounds=a.rounds, n_generators=a.generators,
                  disc_per_gen=a.disc_per_gen, max_tracks=a.max_tracks,
                  noniid=a.noniid)

    jobs = []
    for suite in suites:
        for cfg in SUITES[suite](base):
            for seed in range(a.seeds):
                jobs.append((suite, asdict(cfg), seed, a.out, a.overwrite))

    todo = [j for j in jobs
            if a.overwrite or not os.path.exists(job_path(a.out, j[0], j[1]["name"], j[2]))]
    print(f"{len(jobs)} jobs ({len(suites)} suites x {a.seeds} seeds); "
          f"{len(todo)} to run, {len(jobs)-len(todo)} already done")
    if a.dry_run:
        for j in todo[:20]:
            print("  ", j[0], j[1]["name"], "seed", j[2])
        return

    done = fail = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in todo}
        for fut in as_completed(futs):
            status, name, seed, info = fut.result()
            done += 1
            if status == "fail":
                fail += 1
                print(f"[{done}/{len(todo)}] FAIL {name} seed{seed}\n{info}", flush=True)
            elif status == "ok":
                print(f"[{done}/{len(todo)}] ok   {name} seed{seed} rmse={info:.4f}", flush=True)
    print(f"finished: {done} run, {fail} failed")


if __name__ == "__main__":
    main()
