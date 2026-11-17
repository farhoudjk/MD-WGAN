#!/usr/bin/env python3
"""Pre-build the dataset cache for every (node, seed) the sweep will touch."""
import os, sys, itertools
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import build_node_dataset_cached

ROOT = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/datasets"

def build(args):
    nid, seed, mt = args
    p = os.path.join(ROOT, f"discriminator_{nid}", "data.csv")
    if not os.path.exists(p):
        return f"missing {nid}"
    d = build_node_dataset_cached(p, window=8, test_frac=0.2, seed=seed, max_tracks=mt)
    return f"node{nid} seed{seed} tracks={mt} train={len(d['ctx_train'])}"

if __name__ == "__main__":
    nodes = range(1, 21)
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    tracks = [400]
    jobs = list(itertools.product(nodes, seeds, tracks))
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, r in enumerate(ex.map(build, jobs), 1):
            if i % 20 == 0:
                print(f"[{i}/{len(jobs)}] {r}", flush=True)
    print("cache ready")
