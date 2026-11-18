"""Full NGSIM release: site-aware partitions with equal data per vehicle.

The shipped `data.csv` is a random sample of rows from the whole NGSIM release:
consecutive records of a vehicle are 1-5+ frames apart and all four sites are
mixed together.  This module works from the full public release instead
(data.transportation.gov dataset 8ect-6jqj) and builds genuine trajectories:

  * records are de-duplicated and split into contiguous 10 Hz runs, so a
    context window never spans missing frames;
  * runs are resampled to 5 Hz, so a window of W=8 frames covers 1.6 s;
  * the same five features and leader filter as core/data.py are used.

Partitioning (one shard per discriminator):

  * RSU g is placed at site SITE_ORDER[g % 4]; with two RSUs that is one
    freeway and one urban arterial, with four every site is covered, with
    eight each site hosts two RSUs.
  * Within a site, shards are cells of (time third x road segment), ordered
    round-robin over time thirds.  RSU g takes the next `pool` cells of its
    site, so shards are disjoint and nested across scales: the vehicles of a
    4-RSU deployment are a subset of those of an 8-RSU one.
  * Every shard holds the same number of trajectories (tracks_per_node), split
    train/test by vehicle, so adding RSUs adds vehicles without thinning the
    data any one vehicle holds.
  * One global scaler is fitted on the training targets of all shards.
"""
import hashlib
import os

import numpy as np
import pandas as pd

from .data import FEATURES, FeatureScaler, _windows_from_track, _CACHE_DIR

NGSIM_FULL = "/home/ubuntu/ns-allinone-3.43/ns-3.43/GANs/WGAN/ngsim_full/ngsim_full.csv"
SITE_ORDER = ("i-80", "lankershim", "us-101", "peachtree")
CELLS_PER_SITE = 12
_RAW_COLS = {"Vehicle_ID": "vehicle_id", "Frame_ID": "frame_id",
             "Global_Time": "time", "Local_X": "position_x", "Local_Y": "position_y",
             "v_Vel": "speed", "v_Acc": "acceleration", "Lane_ID": "lane_id",
             "Space_Headway": "spacing", "Location": "location"}


def _site_key(name):
    s = str(name).strip().lower()
    for k in SITE_ORDER:
        if k in s:
            return k
    return s


def load_tracks(csv_path=NGSIM_FULL, hz=5, cache_dir=None):
    """Contiguous trajectories of the full release, cached as one .npz.

    Returns a dict with:
      feats  : float32 array of all resampled records, FEATURES order
      offs   : int64 start offset of each track into feats (len = n_tracks + 1)
      site   : site key per track
      period : recording-period index per track (within its site)
      veh    : vehicle-instance id per track (site, id, continuous time)
      ymean  : mean longitudinal position per track
    """
    cache_dir = cache_dir or _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    st = os.stat(csv_path)
    key = f"ngsimtracks-v3|{os.path.abspath(csv_path)}|{st.st_size}|{hz}"
    path = os.path.join(cache_dir, f"ng_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}

    df = pd.read_csv(csv_path, usecols=list(_RAW_COLS))
    df = df.rename(columns=_RAW_COLS)
    df["location"] = df["location"].map(_site_key)
    df = df[df["location"].isin(SITE_ORDER)]
    # Vehicle and frame ids restart with every 15-minute collection file, so
    # the same id names different cars.  A car is (site, id, continuous time):
    # a new vehicle instance starts whenever the id reappears after > 1 s.
    df = df.drop_duplicates(subset=["location", "vehicle_id", "time"])
    df = df.sort_values(["location", "vehicle_id", "time"]).reset_index(drop=True)
    new = ((df["location"] != df["location"].shift())
           | (df["vehicle_id"] != df["vehicle_id"].shift())
           | (df["time"].diff() > 1000))
    df["uid"] = new.cumsum().astype(np.int64)

    # Time bins: the collection periods run back to back, so there is no gap
    # to detect.  Each site's vehicle instances are split into equal-count
    # thirds by when they passed (early / middle / late), which captures
    # congestion building over the recording.
    df["period"] = 0
    for site, g in df.groupby("location"):
        vt = g.groupby("uid")["time"].mean()
        q = np.quantile(vt.to_numpy(), [1 / 3, 2 / 3])
        bins = pd.Series(np.searchsorted(q, vt.to_numpy(), side="right"), index=vt.index)
        df.loc[g.index, "period"] = g["uid"].map(bins).to_numpy()

    # Same leader filter as core/data.py: spacing is blank/0 without a leader.
    df = df[df["spacing"] > 0]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    feats, offs, site, period, veh, ymean = [], [0], [], [], [], []
    step = max(1, int(round(10 / hz)))
    for (s, p, v), g in df.groupby(["location", "period", "uid"], sort=False):
        fr = g["time"].to_numpy()
        arr = g[FEATURES].to_numpy(dtype=np.float32)
        y = g["position_y"].to_numpy()
        # Split where records are missing (10 Hz = 100 ms), then resample
        # each contiguous run.
        breaks = np.flatnonzero(np.diff(fr) != 100) + 1
        for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(fr)]):
            run = arr[a:b:step]
            if len(run) < 2 * 8 + 1:          # too short to give train windows
                continue
            feats.append(run); offs.append(offs[-1] + len(run))
            site.append(s); period.append(int(p)); veh.append(int(v))
            ymean.append(float(y[a:b].mean()))

    out = {"feats": np.concatenate(feats).astype(np.float32),
           "offs": np.asarray(offs, np.int64),
           "site": np.asarray(site), "period": np.asarray(period, np.int64),
           "veh": np.asarray(veh, np.int64), "ymean": np.asarray(ymean, np.float64)}
    tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return out


def site_cells(T, site, n_cells=CELLS_PER_SITE):
    """Track indices of each (period x segment) cell of one site, round-robin
    over periods.  Assignment is per vehicle, so a vehicle's runs share a cell."""
    idx = np.flatnonzero(T["site"] == site)
    periods = np.unique(T["period"][idx])
    per_p = int(np.ceil(n_cells / len(periods)))
    by_period = []
    for p in periods:
        ip = idx[T["period"][idx] == p]
        # One position per vehicle (mean over its runs), then quantile segments.
        vk = T["veh"][ip]
        uv, inv = np.unique(vk, return_inverse=True)
        vy = np.bincount(inv, weights=T["ymean"][ip]) / np.bincount(inv)
        q = np.quantile(vy, np.linspace(0, 1, per_p + 1)); q[-1] += 1e-6
        seg = np.clip(np.searchsorted(q, vy, side="right") - 1, 0, per_p - 1)
        by_period.append([ip[seg[inv] == s] for s in range(per_p)])
    cells = []
    for s in range(per_p):                     # round-robin over periods
        for cp in by_period:
            cells.append(cp[s])
    return cells[:n_cells]


def build_ngsim_partitions(n_generators, pool, window=8, test_frac=0.2, seed=0,
                           tracks_per_node=40, test_windows=1000, shuffled=False,
                           csv_path=NGSIM_FULL, cache_dir=None):
    """One dataset per discriminator, ordered by node id (RSU-major)."""
    cache_dir = cache_dir or _CACHE_DIR
    key = (f"ngsimparts-v3|{os.path.abspath(csv_path)}|{n_generators}|{pool}|{window}|"
           f"{test_frac}|{seed}|{tracks_per_node}|{test_windows}|{shuffled}")
    path = os.path.join(cache_dir, f"np_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return [{"ctx_train": z[f"ctx_train_{i}"], "tgt_train": z[f"tgt_train_{i}"],
                 "ctx_test": z[f"ctx_test_{i}"], "tgt_test": z[f"tgt_test_{i}"],
                 "scaler": FeatureScaler(z["mean"], z["std"]), "window": window,
                 "n_features": len(FEATURES), "site": str(z[f"site_{i}"]),
                 "n_tracks": int(z[f"ntr_{i}"])}
                for i in range(int(z["n_nodes"]))]

    T = load_tracks(csv_path)
    rng = np.random.default_rng(seed)
    cells = {s: site_cells(T, s) for s in SITE_ORDER}
    track = lambda i: T["feats"][T["offs"][i]:T["offs"][i + 1]]

    node_tracks, node_site = [], []
    for g in range(n_generators):
        s = SITE_ORDER[g % len(SITE_ORDER)]
        rank = g // len(SITE_ORDER)
        for j in range(pool):
            c = rank * pool + j
            if c >= len(cells[s]):
                raise ValueError(f"site {s} has {len(cells[s])} cells; "
                                 f"RSU {g} needs cell {c}")
            ids = cells[s][c]
            # Equal data per vehicle: sample whole vehicles until the shard
            # holds tracks_per_node runs.
            vehs = rng.permutation(np.unique(T["veh"][ids]))
            chosen = []
            for v in vehs:
                chosen.extend(ids[T["veh"][ids] == v].tolist())
                if len(chosen) >= tracks_per_node:
                    break
            node_tracks.append(chosen[:tracks_per_node]); node_site.append(s)

    if shuffled:
        # Control: pool each RSU's tracks and redeal them uniformly, giving its
        # discriminators statistically identical shards.
        for g in range(n_generators):
            sl = slice(g * pool, (g + 1) * pool)
            allt = rng.permutation([t for nt in node_tracks[sl] for t in nt])
            for j, chunk in enumerate(np.array_split(allt, pool)):
                node_tracks[g * pool + j] = chunk.tolist()

    def stack(ids):
        cs, ts = [], []
        for i in ids:
            c, y = _windows_from_track(track(i), window)
            if c is not None:
                cs.append(c); ts.append(y)
        f = len(FEATURES)
        if not cs:
            return np.zeros((0, window, f), np.float32), np.zeros((0, f), np.float32)
        return np.concatenate(cs), np.concatenate(ts)

    built = []
    for ids in node_tracks:
        # Train/test split by vehicle, so no vehicle appears in both.
        vk = T["veh"][ids]
        uv = rng.permutation(np.unique(vk))
        n_te = max(1, int(round(test_frac * len(uv))))
        te_v = set(uv[:n_te].tolist())
        tr = [i for i in ids if T["veh"][i] not in te_v]
        te = [i for i in ids if T["veh"][i] in te_v]
        (ctr, ttr), (cte, tte) = stack(tr), stack(te)
        if len(cte) > test_windows:            # fixed-size evaluation sample
            k = rng.choice(len(cte), test_windows, replace=False)
            cte, tte = cte[k], tte[k]
        built.append((ctr, ttr, cte, tte))

    scaler = FeatureScaler().fit(np.concatenate([b[1] for b in built if len(b[1])]))
    f = len(FEATURES)
    sc = lambda a: scaler.transform(a.reshape(-1, f)).reshape(a.shape).astype(np.float32)
    out, blob = [], {"n_nodes": len(built), "mean": scaler.mean, "std": scaler.std}
    for i, (ctr, ttr, cte, tte) in enumerate(built):
        d = {"ctx_train": sc(ctr), "tgt_train": sc(ttr), "ctx_test": sc(cte),
             "tgt_test": sc(tte), "scaler": scaler, "window": window, "n_features": f,
             "site": node_site[i], "n_tracks": len(node_tracks[i])}
        out.append(d)
        for k in ("ctx_train", "tgt_train", "ctx_test", "tgt_test"):
            blob[f"{k}_{i}"] = d[k]
        blob[f"site_{i}"] = node_site[i]; blob[f"ntr_{i}"] = len(node_tracks[i])
    tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, **blob)
    os.replace(tmp, path)
    return out
