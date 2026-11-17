# MD-WGAN: adaptive multi-discriminator WGAN for resource-constrained IoV

Reference implementation and experiment harness for the MD-WGAN framework:
RSU-hosted conditional WGAN generators, vehicle-hosted critics, DQN-based
discriminator selection, and game-theoretic epoch allocation.

Every mechanism described in the paper is implemented here and can be switched
off independently, so each claimed contribution is separately measurable.

## Layout

```
mdwgan/
  core/
    data.py       NGSIM -> per-vehicle (context, target) windows; per-track
                  train/test split (no sequence overlap); cached to .npz
    models.py     conditional G(c,z) and critic D(c,y); WGAN-GP penalty
    game.py       best-response epoch coordination with online re-estimation
    agent.py      Double-DQN subset selection + non-learning selectors
    network.py    802.11p link model (log-distance path loss + Shannon rate)
    trainer.py    federated round loop (Algorithm 1), evaluation, Eq. (8) cost
  experiments/
    suite.py      the experiment matrix (ablation/baseline/sensitivity/non-IID)
    runner.py     parallel, resumable executor; one JSON per (arm, seed)
    prewarm.py    pre-builds the dataset cache
    ns3_sweep.sh  ns-3 802.11p communication sweep
  analysis/
    stats.py      bootstrap CIs, paired Wilcoxon, Holm correction, Cohen's d_z
    report.py     LaTeX tables + figures for the paper
  docs/
    REVISION_PLAN.md  diagnosis of the prior rejections and what changed
```

## Reproducing the results

```bash
PY=/home/ubuntu/ns-allinone-3.43/ns-3.43/myenv/bin/python3

# 1. build the dataset cache (once; ~30 s)
$PY mdwgan/experiments/prewarm.py 5

# 2. run the full matrix: 46 arms x 5 seeds
$PY mdwgan/experiments/runner.py --suites all --seeds 5 --rounds 60 \
      --generators 4 --disc-per-gen 5 --max-tracks 400 --workers 6

# 3. tables and figures
$PY mdwgan/analysis/report.py

# 4. ns-3 communication measurements
cd /home/ubuntu/ns-allinone-3.43/ns-3.43 && ./ns3 build mdwgan-comm
bash mdwgan/experiments/ns3_sweep.sh
```

The runner is resumable: re-running skips any `(arm, seed)` whose JSON already
exists, so an interrupted sweep continues where it stopped. Pass `--overwrite`
to force recomputation.

## Ablation switches

All are fields of `core.trainer.Config`:

| Switch | Default | Turns off |
|---|---|---|
| `selector` | `dqn` | RL selection (`random`, `greedy`, `roundrobin`, `all`) |
| `use_game` | `True` | game-theoretic allocation -> fixed `(E_D, E_G)` |
| `adaptive_game` | `True` | online re-estimation -> static paper constants |
| `use_ensemble` | `True` | ensemble feedback -> a single critic |
| `ensemble_weighting` | `uniform` | `performance` weights critics by recent loss |
| `conditional` | `True` | context conditioning -> unconditional `G(z)` |
| `use_adversarial` | `True` | critic feedback -> supervised-only |
| `lambda_sup` | `10.0` | supervised term (`0.0` -> pure adversarial) |
| `noniid` | `natural` | `shuffled` re-deals partitions IID |

## Metrics

Accuracy is held-out next-step prediction error. The generator is a stochastic
map, so point predictions are the mean over `eval_samples` latent draws (the
conditional mean is the Bayes-optimal estimator under squared error).

Alongside RMSE/MAE the harness records mode coverage and a Jensen-Shannon
proxy, because RMSE alone cannot detect mode collapse — a generator that
predicts the conditional mean everywhere scores well on RMSE while producing no
diversity at all.

Resource metrics (CPU, memory) are measured from the training processes.
Latency and traffic come from the 802.11p model in `network.py`, not from the
size of files on local disk.

## Notes

The generator head is a **bounded** residual around the last observed frame.
An unbounded head extrapolates without limit on held-out contexts: in testing it
produced deltas of ~2300 standard deviations and a test RMSE of ~95 on data
bounded within 3 standard deviations. The bound is also physically motivated —
a vehicle's state cannot change arbitrarily between consecutive samples.

Training uses a supervised term alongside the adversarial one. A pure WGAN
objective rewards distributional realism, which does not constrain per-sample
prediction error; with `lambda_sup = 0` the held-out RMSE oscillates and does
not beat a persistence baseline. `lambda_sup` is swept in the sensitivity suite
so the contribution of each term is quantified rather than assumed.
