"""Experiment matrix: ablations, baselines, and sensitivity studies.

Each entry returns a list of Config overrides.  `runner.py` expands these over
seeds and executes them.  Keeping the matrix declarative means every number in
the paper is traceable to one named arm plus a seed.
"""
from dataclasses import replace
from core.trainer import Config


def _c(base, name, **kw):
    return replace(base, name=name, **kw)


def adaptive(base):
    """Objective-adaptive control: does reading the Eq. (8) weights help?

    For each deployment scenario, the same framework is run with no controller,
    with a static alpha->configuration map, and with the online dual-ascent
    controller.  Each arm must be scored under *its own* alpha, so the question
    asked is: for this weighting, does adapting beat a fixed configuration?
    """
    from core.adaptive import SCENARIOS
    out = []
    for name, a in SCENARIOS.items():
        for mode in ("none", "static", "online"):
            out.append(_c(base, f"adapt-{name}-{mode}",
                          objective_weights=a, adaptive_cfg=mode))
    return out


# ---------------------------------------------------------------------------
# 1. Component ablations: leave-one-out and leave-two-out over the three
#    claimed contributions (RL selection, game-theoretic allocation, ensemble).
# ---------------------------------------------------------------------------
def ablations(base):
    return [
        _c(base, "full"),

        # -- leave one out --
        _c(base, "no-rl",        selector="random"),
        _c(base, "no-game",      use_game=False),
        _c(base, "no-ensemble",  use_ensemble=False),

        # -- leave two out --
        _c(base, "no-rl-game",      selector="random", use_game=False),
        _c(base, "no-rl-ensemble",  selector="random", use_ensemble=False),
        _c(base, "no-game-ensemble", use_game=False, use_ensemble=False),

        # -- all three off: the plain federated WGAN skeleton --
        _c(base, "none", selector="random", use_game=False, use_ensemble=False),

        # -- finer-grained variants of each mechanism --
        # Is the *adaptivity* of the coordinator doing the work, or just the
        # particular (E_D, E_G) it happens to pick?
        _c(base, "static-game", adaptive_game=False),
        # Does conditioning on the trajectory context matter? (old code did not)
        _c(base, "unconditional", conditional=False),
        # Is the supervised term carrying all the accuracy?
        _c(base, "no-supervised", lambda_sup=0.0),
        # Is adversarial training contributing anything beyond regression?
        _c(base, "supervised-only", use_adversarial=False),
        # Does performance-weighting the ensemble beat uniform averaging?
        _c(base, "ensemble-weighted", ensemble_weighting="performance"),
    ]


# ---------------------------------------------------------------------------
# 2. Selection baselines: what the DQN must beat for the RL claim to hold.
# ---------------------------------------------------------------------------
def selection_baselines(base):
    return [
        _c(base, "sel-dqn",        selector="dqn"),
        _c(base, "sel-random",     selector="random"),
        _c(base, "sel-greedy",     selector="greedy"),
        _c(base, "sel-roundrobin", selector="roundrobin"),
        _c(base, "sel-all",        selector="all"),
        _c(base, "sel-genetic",    selector="genetic"),
        _c(base, "sel-lazygreedy", selector="lazygreedy"),
        _c(base, "sel-residence",  selector="residence"),
    ]


# ---------------------------------------------------------------------------
# 3. Sensitivity studies.
# ---------------------------------------------------------------------------
def sensitivity(base):
    out = []

    # (a) subset size k: the accuracy/resource knob.
    for k in (1, 2, 3, 4, 5):
        out.append(_c(base, f"k{k}", top_k=k))

    # (b) discriminator pool size per generator (scalability in K).
    for dpg in (3, 5, 8):
        out.append(_c(base, f"pool{dpg}", disc_per_gen=dpg, top_k=min(3, dpg)))

    # (c) generator count (scalability in N_G).
    for ng in (2, 4, 6):
        out.append(_c(base, f"gens{ng}", n_generators=ng))

    # (d) reward weighting: the four application scenarios of Sec. V-F7.
    scenarios = {
        "rw-balanced":   (0.2, 0.2, 0.2, 0.2, 0.2),
        "rw-accuracy":   (0.4, 0.15, 0.15, 0.15, 0.15),
        "rw-delay":      (0.15, 0.4, 0.15, 0.15, 0.15),
        "rw-network":    (0.15, 0.15, 0.4, 0.15, 0.15),
        "rw-resource":   (0.14, 0.14, 0.14, 0.29, 0.29),
    }
    for nm, w in scenarios.items():
        out.append(_c(base, nm, reward_weights=w))

    # (e) supervised/adversarial balance.
    for ls in (0.0, 1.0, 5.0, 10.0, 25.0):
        out.append(_c(base, f"lsup{ls:g}", lambda_sup=ls))

    # (f) interference coefficient of the coordination game.
    for g in (0.0, 0.3, 0.5, 0.7, 0.9):
        out.append(_c(base, f"gnash{g:g}", gamma_nash=g))

    return out


# ---------------------------------------------------------------------------
# 4. Non-IID severity: how much does partition heterogeneity matter.
# ---------------------------------------------------------------------------
def noniid(base):
    """Shard heterogeneity: this is the variable that decides whether learned
    selection has anything to learn."""
    return [_c(base, f"noniid-{s}", noniid=s)
            for s in ("segment", "lane", "iid-split", "natural", "shuffled")]


def diversity(base):
    """Does the generator actually use its latent input?

    The baseline generator's sample spread is ~3% of the data's, i.e. it ignores
    z and is a deterministic predictor wearing a GAN's clothes.  These arms test
    the two standard remedies and their combination.
    """
    out = [_c(base, "div-none")]
    for lam in (0.1, 0.5, 1.0):
        out.append(_c(base, f"div-ms{lam:g}", lambda_div=lam))
    out.append(_c(base, "div-bok", sup_mode="best_of_k"))
    out.append(_c(base, "div-bok-ms0.5", sup_mode="best_of_k", lambda_div=0.5))
    for k in (4, 16):
        out.append(_c(base, f"div-bok-k{k}", sup_mode="best_of_k", sup_k=k))
    return out


def candidates(base):
    """End-to-end candidates for the final recommended configuration."""
    best = dict(selector="greedy", sup_mode="best_of_k")
    return [
        _c(base, "cand-current"),
        _c(base, "cand-greedy", selector="greedy"),
        _c(base, "cand-bok", sup_mode="best_of_k"),
        _c(base, "cand-greedy-bok", **best),
        _c(base, "cand-greedy-bok-k2", top_k=2, **best),
        _c(base, "cand-greedy-bok-ms", lambda_div=0.5, **best),
        _c(base, "cand-greedy-bok-static", adaptive_game=False, **best),
        _c(base, "cand-greedy-bok-wens", ensemble_weighting="performance", **best),
    ]


def sota(base):
    """External comparison methods, all under one evaluation protocol."""
    best = dict(selector="greedy", sup_mode="best_of_k")
    return [
        _c(base, "ours", **best),
        _c(base, "ours-dqn"),
        _c(base, "hvs-gan", method="hvs-gan"),
        _c(base, "fedavg-gan", method="fedavg-gan"),
        _c(base, "md-gan", method="md-gan"),
        _c(base, "early-update-mdgan", method="early-update-mdgan"),
    ]


def knobs(base):
    """Which knob carries the adaptive result?

    The controller moves four knobs at once.  Each arm here enables exactly one
    (plus an all-knobs reference and a no-control control), so the contribution
    of each is separable.  Run under the network- and resource-weighted
    scenarios, where the fixed configuration is furthest from optimal.
    """
    from core.adaptive import SCENARIOS
    out = []
    for scen in ("network", "resource"):
        a = SCENARIOS[scen]
        out.append(_c(base, f"knob-{scen}-none", objective_weights=a,
                      adaptive_cfg="none"))
        out.append(_c(base, f"knob-{scen}-all", objective_weights=a,
                      adaptive_cfg="static"))
        for k in ("top_k", "E_G", "sync_every", "lambda_sup"):
            out.append(_c(base, f"knob-{scen}-{k}", objective_weights=a,
                          adaptive_cfg="static", adaptive_knobs=(k,)))
    return out


# ---------------------------------------------------------------------------
# Scalability.  Selection strategies of the proposed framework (DQN, greedy,
# random), each with the objective-adaptive controller (static policy,
# balanced weighting) so only the selector differs, plus the related-work
# baselines at their fixed configurations.  k = 3 throughout
# (all critics active when the pool has 3).
#   scale_pool: N_G = 2, local pool 3 / 5 / 8 / 10
#   scale_rsu : pool 5, N_G = 2 / 4 / 6 / 8
# The baselines load one shipped per-node dataset per vehicle and only 20
# exist, so they are limited to N_G * pool <= 20 here.
# ---------------------------------------------------------------------------
_SCALE_SEL = ("dqn", "greedy", "random")
_SCALE_BL = ("hvs-gan", "fedavg-gan", "early-update-mdgan")
_MAX_BASELINE_NODES = 20
_SCALE_ALPHA = (0.2, 0.2, 0.2, 0.2, 0.2)


def _scale_arms(base, tag, n_gen, pool):
    out = [_c(base, f"{tag}-{sel}", n_generators=n_gen, disc_per_gen=pool,
              top_k=min(3, pool), selector=sel, adaptive_cfg="static",
              objective_weights=_SCALE_ALPHA)
           for sel in _SCALE_SEL]
    # Shipped per-node datasets cap the baselines at 20 vehicles; the full
    # NGSIM shards (noniid="ngsim") have no such limit.
    if base.noniid.startswith("ngsim") or n_gen * pool <= _MAX_BASELINE_NODES:
        out += [_c(base, f"{tag}-{m}", n_generators=n_gen, disc_per_gen=pool,
                   top_k=min(3, pool), method=m) for m in _SCALE_BL]
    return out


def scale_pool(base):
    return [a for pool in (3, 5, 8, 10) for a in _scale_arms(base, f"pool{pool}", 2, pool)]


def scale_rsu(base):
    return [a for g in (2, 4, 6, 8) for a in _scale_arms(base, f"rsu{g}", g, 5)]


def partition(base):
    """Full-NGSIM partitioning control: site-aware shards vs the same tracks
    redealt uniformly within each RSU (statistically identical shards)."""
    best = dict(selector="greedy", sup_mode="best_of_k")
    return [_c(base, "part-site", noniid="ngsim", **best),
            _c(base, "part-shuffled", noniid="ngsim-shuffled", **best)]


SUITES = {
    "knobs": knobs,
    "adaptive": adaptive,
    "sota": sota,
    "diversity": diversity,
    "candidates": candidates,
    "ablation": ablations,
    "baseline": selection_baselines,
    "sensitivity": sensitivity,
    "noniid": noniid,
    "scale_pool": scale_pool,
    "scale_rsu": scale_rsu,
    "partition": partition,
}
