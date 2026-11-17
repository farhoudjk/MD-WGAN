"""NGSIM trajectory data: per-vehicle (context, target) sequence pairs.

Implements the dataset construction described in Sec. V-B of the paper: each
sample is a window of W consecutive past feature vectors (the context c) paired
with the subsequent vector (the target y).  Splits are made per *vehicle track*
so that no sequence overlaps between train and test, which is what lets RMSE/MAE
be read as held-out next-step prediction error.
"""
import os
import numpy as np
import pandas as pd

# Sec. V-B feature set: longitudinal speed, longitudinal acceleration,
# lane identifier, headway distance to lead vehicle, lateral position.
FEATURES = ["speed", "acceleration", "lane_id", "spacing", "position_x"]

# Columns required to order and group raw NGSIM records into tracks.
_ID_COL = "vehicle_id"
_TIME_COL = "frame_id"


class FeatureScaler:
    """Z-score scaler fitted on training data only (no test-set leakage)."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        # Guard constant features; a zero std would blow up the transform.
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_state(state):
        return FeatureScaler(np.asarray(state["mean"]), np.asarray(state["std"]))


def _clean(df):
    """Select the modelled features and drop records NGSIM marks as unusable."""
    missing = [c for c in FEATURES + [_ID_COL, _TIME_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected NGSIM columns: {missing}")
    df = df[[_ID_COL, _TIME_COL] + FEATURES].copy()
    # NGSIM leaves spacing blank when the vehicle has no leader, which is the
    # majority of records (74% here).  Dropping them discards 95% of the usable
    # tracks, so "no leader" is encoded explicitly as a large headway instead --
    # it is a meaningful state (free-flow), not missing data.
    # Keep only records with an observed leader.  NGSIM leaves spacing blank in
    # free flow, and imputing a constant there makes the feature degenerate
    # (94% a single value) and the model worse -- measured, not assumed.
    df = df[df["spacing"] > 0]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _windows_from_track(track, window):
    """All (context, target) pairs from one vehicle's ordered trajectory."""
    n = len(track)
    if n <= window:
        return None, None
    # Strided view: row i is track[i : i+window], target is track[i+window].
    idx = np.arange(n - window)[:, None] + np.arange(window)[None, :]
    ctx = track[idx]
    tgt = track[np.arange(window, n)]
    return ctx, tgt


def build_node_dataset(csv_path, window=8, test_frac=0.2, seed=0, max_tracks=None):
    """Build train/test (context, target) tensors for one vehicular node.

    Splitting is done over whole vehicle tracks rather than over windows, so a
    context in the test set never overlaps a context seen during training.
    """
    df = _clean(pd.read_csv(csv_path))
    rng = np.random.default_rng(seed)

    tracks = []
    for _, g in df.groupby(_ID_COL, sort=True):
        g = g.sort_values(_TIME_COL)
        arr = g[FEATURES].to_numpy(dtype=np.float32)
        if len(arr) > window:
            tracks.append(arr)

    if not tracks:
        raise ValueError(f"no usable vehicle tracks in {csv_path}")

    order = rng.permutation(len(tracks))
    if max_tracks is not None:
        order = order[:max_tracks]
    n_test = max(1, int(round(test_frac * len(order))))
    test_ids, train_ids = order[:n_test], order[n_test:]
    if len(train_ids) == 0:  # degenerate tiny node: keep at least one track
        train_ids, test_ids = order[:1], order[1:2]

    def stack(ids):
        cs, ts = [], []
        for i in ids:
            c, t = _windows_from_track(tracks[i], window)
            if c is not None:
                cs.append(c)
                ts.append(t)
        if not cs:
            return (np.zeros((0, window, len(FEATURES)), np.float32),
                    np.zeros((0, len(FEATURES)), np.float32))
        return np.concatenate(cs), np.concatenate(ts)

    ctx_tr, tgt_tr = stack(train_ids)
    ctx_te, tgt_te = stack(test_ids)

    # Fit the scaler on training targets only, then apply everywhere.
    scaler = FeatureScaler().fit(tgt_tr if len(tgt_tr) else np.zeros((1, len(FEATURES)), np.float32))
    shp_tr, shp_te = ctx_tr.shape, ctx_te.shape
    ctx_tr = scaler.transform(ctx_tr.reshape(-1, len(FEATURES))).reshape(shp_tr).astype(np.float32)
    ctx_te = scaler.transform(ctx_te.reshape(-1, len(FEATURES))).reshape(shp_te).astype(np.float32)
    tgt_tr = scaler.transform(tgt_tr).astype(np.float32)
    tgt_te = scaler.transform(tgt_te).astype(np.float32)

    return {
        "ctx_train": ctx_tr, "tgt_train": tgt_tr,
        "ctx_test": ctx_te, "tgt_test": tgt_te,
        "scaler": scaler, "window": window, "n_features": len(FEATURES),
    }


# ---------------------------------------------------------------------------
# Disk cache.  Parsing a 197k-row NGSIM CSV costs several seconds and is
# identical across every arm and seed sharing (window, test_frac, seed,
# max_tracks); without a cache the sweep spends more time in pandas than in
# training.
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")


def build_node_dataset_cached(csv_path, window=8, test_frac=0.2, seed=0,
                              max_tracks=None, cache_dir=None):
    import hashlib
    cache_dir = cache_dir or _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    key = f"{os.path.abspath(csv_path)}|{window}|{test_frac}|{seed}|{max_tracks}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{h}.npz")

    if os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            return {
                "ctx_train": z["ctx_train"], "tgt_train": z["tgt_train"],
                "ctx_test": z["ctx_test"], "tgt_test": z["tgt_test"],
                "scaler": FeatureScaler(z["mean"], z["std"]),
                "window": int(z["window"]), "n_features": int(z["n_features"]),
            }
        except Exception:
            pass  # corrupt or partial entry: fall through and rebuild

    d = build_node_dataset(csv_path, window=window, test_frac=test_frac,
                           seed=seed, max_tracks=max_tracks)
    # np.savez appends ".npz" to whatever stem it is given, so build the temp
    # stem without the suffix and replace from the file it actually wrote.
    stem = path[:-4] + f".{os.getpid()}.tmp"
    np.savez(stem, ctx_train=d["ctx_train"], tgt_train=d["tgt_train"],
             ctx_test=d["ctx_test"], tgt_test=d["tgt_test"],
             mean=d["scaler"].mean, std=d["scaler"].std,
             window=d["window"], n_features=d["n_features"])
    os.replace(stem + ".npz", path)   # atomic: readers never see a partial file
    return d


# ---------------------------------------------------------------------------
# Genuine non-IID partitioning.
#
# The shipped per-node CSVs are random splits of one pool: their raw per-feature
# means agree to within 0.5%, so every partition is statistically the same and
# *no* selection rule can outperform any other.  Worse, each node was scaled by
# its own mean/std, which drives every partition to exactly N(0,1) and erases
# whatever heterogeneity did exist.
#
# Here partitions are cut by road segment (the spatial partitioning the system
# model actually describes) and all nodes share one global scaler, so a
# discriminator's shard genuinely differs from its neighbour's.
# ---------------------------------------------------------------------------

def build_spatial_partitions(csv_path, n_nodes=10, window=8, test_frac=0.2,
                             seed=0, max_tracks=None, scheme="segment",
                             cache_dir=None):
    """Return a list of n_nodes datasets cut by road segment or by lane.

    scheme='segment' slices by longitudinal position (distinct traffic regimes
    along the corridor); scheme='lane' slices by lane group (distinct speed and
    headway distributions); scheme='random' reproduces the original IID split as
    a control.
    """
    import hashlib
    cache_dir = cache_dir or _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    key = f"spatial|{os.path.abspath(csv_path)}|{n_nodes}|{window}|{test_frac}|{seed}|{max_tracks}|{scheme}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"sp_{h}.npz")
    if os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            n = int(z["n_nodes"])
            out = []
            for i in range(n):
                out.append({
                    "ctx_train": z[f"ctx_train_{i}"], "tgt_train": z[f"tgt_train_{i}"],
                    "ctx_test": z[f"ctx_test_{i}"], "tgt_test": z[f"tgt_test_{i}"],
                    "scaler": FeatureScaler(z["mean"], z["std"]),
                    "window": int(z["window"]), "n_features": int(z["n_features"]),
                })
            return out
        except Exception:
            pass

    df = _clean(pd.read_csv(csv_path))
    raw = pd.read_csv(csv_path, usecols=[_ID_COL, "position_y", "lane_id"])
    # One assignment per vehicle track, so a track is never split across nodes.
    key_by_veh = raw.groupby(_ID_COL).agg({"position_y": "mean", "lane_id": "median"})
    rng = np.random.default_rng(seed)

    if scheme == "segment":
        q = np.quantile(key_by_veh["position_y"], np.linspace(0, 1, n_nodes + 1))
        q[-1] += 1e-6
        assign = np.clip(np.searchsorted(q, key_by_veh["position_y"], side="right") - 1,
                         0, n_nodes - 1)
    elif scheme == "lane":
        lanes = key_by_veh["lane_id"].to_numpy()
        uniq = np.unique(lanes[np.isfinite(lanes)])
        groups = np.array_split(uniq, n_nodes)
        lut = {l: g for g, ls in enumerate(groups) for l in ls}
        assign = np.array([lut.get(l, rng.integers(n_nodes)) for l in lanes])
    else:
        assign = rng.integers(0, n_nodes, size=len(key_by_veh))

    node_of = dict(zip(key_by_veh.index, assign))

    # Build the windows per node first, unscaled.
    per_node = [[] for _ in range(n_nodes)]
    for vid, g in df.groupby(_ID_COL, sort=True):
        n = node_of.get(vid)
        if n is None:
            continue
        arr = g.sort_values(_TIME_COL)[FEATURES].to_numpy(dtype=np.float32)
        if len(arr) > window:
            per_node[n].append(arr)

    # One global scaler fitted on training targets across all nodes: this is
    # what preserves cross-node differences instead of normalising them away.
    splits = []
    for tracks in per_node:
        if max_tracks is not None and len(tracks) > max_tracks:
            idx = rng.permutation(len(tracks))[:max_tracks]
            tracks = [tracks[i] for i in idx]
        order = rng.permutation(len(tracks))
        n_test = max(1, int(round(test_frac * len(order)))) if len(order) else 0
        splits.append(([tracks[i] for i in order[n_test:]],
                       [tracks[i] for i in order[:n_test]]))

    def stack(tracks):
        cs, ts = [], []
        for t in tracks:
            c, y = _windows_from_track(t, window)
            if c is not None:
                cs.append(c); ts.append(y)
        if not cs:
            return (np.zeros((0, window, len(FEATURES)), np.float32),
                    np.zeros((0, len(FEATURES)), np.float32))
        return np.concatenate(cs), np.concatenate(ts)

    built = [(stack(tr), stack(te)) for tr, te in splits]
    all_tr = np.concatenate([b[0][1] for b in built if len(b[0][1])]) \
        if any(len(b[0][1]) for b in built) else np.zeros((1, len(FEATURES)), np.float32)
    scaler = FeatureScaler().fit(all_tr)

    out = []
    for (ctr, ttr), (cte, tte) in built:
        f = len(FEATURES)
        sc = lambda a: scaler.transform(a.reshape(-1, f)).reshape(a.shape).astype(np.float32)
        out.append({"ctx_train": sc(ctr), "tgt_train": scaler.transform(ttr).astype(np.float32),
                    "ctx_test": sc(cte), "tgt_test": scaler.transform(tte).astype(np.float32),
                    "scaler": scaler, "window": window, "n_features": f})

    blob = {"n_nodes": n_nodes, "window": window, "n_features": len(FEATURES),
            "mean": scaler.mean, "std": scaler.std}
    for i, d in enumerate(out):
        for k in ("ctx_train", "tgt_train", "ctx_test", "tgt_test"):
            blob[f"{k}_{i}"] = d[k]
    stem = path[:-4] + f".{os.getpid()}.tmp"
    np.savez(stem, **blob)
    os.replace(stem + ".npz", path)
    return out
