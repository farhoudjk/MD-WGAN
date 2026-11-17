# Objective-adaptive configuration control

## Why this, and why now

Evaluated on the normalised Eq. (8) cost, **no method wins more than one
scenario**:

| scenario | winner |
|---|---|
| balanced | FedAvg-GAN |
| accuracy-weighted | MD-GAN |
| delay-weighted | FedAvg-GAN |
| network-weighted | MD-GAN |
| resource-weighted | **ours** |

Each method sits at one fixed operating point, so it wins exactly the weighting
that point happens to suit. That is the real gap -- not a missing heuristic.

The framework already exposes knobs that map onto individual Eq. (8) terms, and
the measured sweeps give the direction of each:

| knob | drives | measured evidence |
|---|---|---|
| `top_k` | CPU + traffic | k=1: 4.8 work units / 5.7 MB; k=5: 24.0 / 28.7 |
| `E_G` | training vs CPU | interior optimum at 15 (707 work) vs 5 (810), 25 (768) |
| `sync_every` | traffic vs convergence | traffic scales with sync steps |
| `lambda_sup` | RMSE vs CRPS/PICP | sweep 0-25 moves PICP 0.66 -> 0.19 |
| `sup_mode` | point error vs calibration | best-of-k: PICP 0.26 -> 0.75 |

So a controller that reads the deployment's own alpha vector can move to the
right operating point instead of being fixed to one corner. **No baseline can do
this**: MD-GAN activates every worker by construction, and HVS-GAN's scheduler
optimises *participation*, not the training configuration.

## The two controllers

`StaticPolicy` maps alpha to a configuration once, before training, scoring each
knob by `dot(alpha, direction)` and moving it against its cost. Cheap, no
learning dynamics -- it isolates how much comes from merely *knowing* the
objective.

`OnlineController` adds dual ascent on the realised terms. Each round the
normalised Eq. (8) terms are compared against the share their weight entitles
them to; a term over-consuming raises its multiplier, which pushes its knobs
down. This is the standard dual update for a weighted-sum objective, and it
tracks conditions a static map cannot anticipate (congestion, slow nodes, an
early plateau).

Both are deliberately tiny. Selection experiments showed elaborate search (DQN,
GA, lazy-greedy) buys nothing here, so this is a handful of bounded updates, not
another learned policy.

Enable with `Config(adaptive_cfg="static" | "online")`; `"none"` reproduces the
fixed-configuration behaviour of the baselines.

## Status: IMPLEMENTED, NOT YET VALIDATED

Written and wired into the trainer. **Not run.** The Bash tool was blocked by a
safety classifier partway through this session, so nothing below has been
measured:

1. Does `static` beat fixed config on its own scenario?
2. Does `online` beat `static` -- i.e. does the feedback earn its complexity?
3. Does one adaptive configuration win **all five** scenarios? That is the claim
   worth making, and it is the one that must be tested before it is written.
4. Overhead: the controller is a few vector ops per round, so it should be
   negligible against a round costing seconds -- but measure, do not assert.

## How to validate (fresh session)

Add to `experiments/suite.py`:

    def adaptive(base):
        from core.adaptive import SCENARIOS
        out = []
        for name, a in SCENARIOS.items():
            for mode in ("none", "static", "online"):
                out.append(_c(base, f"adapt-{name}-{mode}",
                              objective_weights=a, adaptive_cfg=mode))
        return out

then run that suite at 5 seeds and score each arm with **its own** alpha, using
the normalised Eq. (8) cost -- not a rank. The honest comparison is: for each
scenario, does `online` beat `none` *under that scenario's weighting*?

## What would make this a real contribution

If one adaptive configuration wins or ties across all five scenarios while each
fixed baseline wins only one, the claim is clean and falsifiable:

> Fixed-configuration federated GAN training cannot serve heterogeneous
> deployment objectives; reading the objective weights and adapting the
> training configuration online does, at negligible overhead.

If it does not -- if `static` is as good as `online`, or the gains are inside
seed noise -- then the honest result is that knowing the objective is enough and
the feedback loop is unnecessary. That is still publishable, and it is what the
evidence will decide.
