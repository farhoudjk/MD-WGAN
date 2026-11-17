"""Objective-adaptive configuration control.

Every method evaluated so far -- ours included -- occupies a single fixed
operating point, so each wins only the Eq. (8) scenario whose weights happen to
suit it.  Measured on NGSIM: MD-GAN wins the accuracy- and network-weighted
scenarios, FedAvg-GAN the delay-weighted one, ours the resource-weighted one.
No configuration wins more than one.

That is a solvable problem rather than a fact of life, because the framework
already exposes knobs that map onto the individual terms of Eq. (8):

    top_k        -> CPU term and traffic term  (fewer active critics)
    E_G, E_D     -> training term vs CPU term  (more generator epochs converge
                    faster per round but cost compute; measured interior
                    optimum at E_G = 15)
    sync_every   -> traffic term vs convergence (fewer synchronisations)
    lambda_sup   -> RMSE against CRPS/PICP      (supervision vs calibration)
    sup_mode     -> point error vs calibration  (mean vs best-of-k)

A controller that reads the deployment's own alpha vector and moves the
configuration toward the knobs that matter can therefore track the objective
instead of being fixed to one corner of it.  The baselines cannot do this: MD-GAN
activates every worker by construction, and HVS-GAN's scheduler optimises
participation, not the training configuration.

Two controllers are provided.

`StaticPolicy` maps alpha to a configuration once, before training, from the
analytic direction of each knob.  It is cheap and has no learning dynamics, so
it isolates how much of the benefit comes from merely *knowing* the objective.

`OnlineController` additionally adjusts during training from measured per-term
costs, using a dual-ascent step on the normalised Eq. (8) terms: a term that is
running above its weighted share pulls its own knob down.  This is the part
that can respond to conditions the static map cannot anticipate -- congestion,
unexpectedly slow nodes, a loss curve that plateaus early.

Both are deliberately tiny.  Selection experiments showed that elaborate search
(DQN, GA, lazy-greedy) buys nothing over trivial rules here, so the controller
is a handful of bounded updates rather than another learned policy.
"""
import numpy as np


# Knob ranges. Each is (lo, hi); the controller only moves inside these.
BOUNDS = {
    "top_k": (1, 5),
    "E_G": (2, 25),
    "sync_every": (1, 4),
    "lambda_sup": (0.0, 25.0),
}

# Direction each knob pushes each Eq. (8) term: +1 means raising the knob
# *increases* that term's cost. Derived from the measured sweeps, not assumed.
#                  train   cpu   mem   lat  traffic
DIRECTION = {
    "top_k":      (-1.0, +1.0, +1.0, +1.0, +1.0),
    "E_G":        (-1.0, +1.0,  0.0, +1.0, +1.0),
    "sync_every": (+1.0, -0.3,  0.0, -1.0, -1.0),
    "lambda_sup": (-0.5,  0.0,  0.0,  0.0,  0.0),
}


def _clip(name, v):
    lo, hi = BOUNDS[name]
    return float(np.clip(v, lo, hi))


class StaticPolicy:
    """Map an alpha vector to a configuration once, before training.

    alpha = (alpha_G, alpha_1 cpu, alpha_2 mem, alpha_3 latency, alpha_4 traffic),
    summing to one, exactly as in Eq. (8).
    """

    name = "static-policy"

    def __init__(self, alpha, base=None):
        self.alpha = np.asarray(alpha, float)
        self.alpha = self.alpha / max(self.alpha.sum(), 1e-9)
        self.base = dict(top_k=3, E_G=15, sync_every=1, lambda_sup=10.0)
        if base:
            self.base.update(base)

    def configure(self):
        a = self.alpha
        cfg = dict(self.base)
        # Score each knob by how much raising it would cost under this alpha:
        # positive score means it is expensive here and should come down.
        for knob, d in DIRECTION.items():
            cost = float(np.dot(a, np.asarray(d, float)))
            lo, hi = BOUNDS[knob]
            span = hi - lo
            # Move proportionally against the cost direction, at most half the
            # range, so a single weighting can never drive a knob to a corner.
            cfg[knob] = _clip(knob, cfg[knob] - 0.5 * span * cost)
        cfg["top_k"] = int(round(cfg["top_k"]))
        cfg["E_G"] = int(round(cfg["E_G"]))
        cfg["sync_every"] = int(round(cfg["sync_every"]))
        # Calibration-sensitive deployments keep best-of-k; a purely
        # accuracy-weighted one does not need it and plain supervision is
        # cheaper and gives slightly better RMSE.
        cfg["sup_mode"] = "mean" if a[0] > 0.35 else "best_of_k"
        return cfg


class OnlineController(StaticPolicy):
    """Static map plus dual ascent on the realised Eq. (8) terms.

    Each round the realised normalised terms are compared against the share
    their weight entitles them to. A term consuming more than its share raises
    its multiplier, which pushes the knobs that drive it downward; a term
    running under budget relaxes. This is the standard dual update for a
    weighted-sum objective, and it keeps the configuration tracking the
    objective rather than a prediction of it.
    """

    name = "online-controller"

    def __init__(self, alpha, base=None, lr=0.25, ema=0.7):
        super().__init__(alpha, base)
        self.lr = lr
        self.ema = ema
        self.mult = np.ones(5)        # dual variables, one per Eq. (8) term
        self.avg = None               # smoothed realised terms
        self.cfg = self.configure()
        self.history = []

    def update(self, terms):
        """terms: realised normalised (train, cpu, mem, latency, traffic) in [0,1]."""
        t = np.asarray(terms, float)
        t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
        self.avg = t if self.avg is None else self.ema * self.avg + (1 - self.ema) * t

        # A term's fair share of the realised total, given its weight.
        total = float(self.avg.sum()) + 1e-9
        share = self.avg / total
        excess = share - self.alpha            # >0 means over-consuming
        self.mult = np.clip(self.mult * np.exp(self.lr * excess), 0.25, 4.0)

        eff = self.alpha * self.mult
        eff = eff / max(eff.sum(), 1e-9)

        cfg = dict(self.base)
        for knob, d in DIRECTION.items():
            cost = float(np.dot(eff, np.asarray(d, float)))
            lo, hi = BOUNDS[knob]
            cfg[knob] = _clip(knob, cfg[knob] - 0.5 * (hi - lo) * cost)
        cfg["top_k"] = int(round(cfg["top_k"]))
        cfg["E_G"] = int(round(cfg["E_G"]))
        cfg["sync_every"] = int(round(cfg["sync_every"]))
        cfg["sup_mode"] = "mean" if eff[0] > 0.35 else "best_of_k"
        self.cfg = cfg
        self.history.append({"terms": t.tolist(), "mult": self.mult.tolist(),
                             **{k: cfg[k] for k in BOUNDS}})
        return cfg


SCENARIOS = {
    # (alpha_G, cpu, mem, latency, traffic) -- the four of Sec. V-F7 plus balanced
    "balanced":  (0.20, 0.20, 0.20, 0.20, 0.20),
    "accuracy":  (0.40, 0.15, 0.15, 0.15, 0.15),
    "delay":     (0.15, 0.15, 0.15, 0.40, 0.15),
    "network":   (0.15, 0.15, 0.15, 0.15, 0.40),
    "resource":  (0.14, 0.29, 0.29, 0.14, 0.14),
}


# ---------------------------------------------------------------------------
# Knob ablation.  The controller moves four knobs at once, so "adaptation
# helps" does not say *which* knob carries the result.  `only` restricts the
# policy to a subset, leaving the rest at their defaults, which isolates the
# contribution of each.
# ---------------------------------------------------------------------------

class RestrictedPolicy(StaticPolicy):
    """StaticPolicy restricted to a named subset of knobs."""

    name = "restricted-policy"

    def __init__(self, alpha, base=None, only=()):
        super().__init__(alpha, base)
        self.only = tuple(only) if only else tuple(BOUNDS)

    def configure(self):
        a = self.alpha
        cfg = dict(self.base)
        for knob, d in DIRECTION.items():
            if knob not in self.only:
                continue          # left at its default: this is the ablation
            cost = float(np.dot(a, np.asarray(d, float)))
            lo, hi = BOUNDS[knob]
            cfg[knob] = _clip(knob, cfg[knob] - 0.5 * (hi - lo) * cost)
        cfg["top_k"] = int(round(cfg["top_k"]))
        cfg["E_G"] = int(round(cfg["E_G"]))
        cfg["sync_every"] = int(round(cfg["sync_every"]))
        cfg["sup_mode"] = ("mean" if ("sup_mode" in self.only and a[0] > 0.35)
                           else self.base.get("sup_mode", "best_of_k"))
        return cfg
