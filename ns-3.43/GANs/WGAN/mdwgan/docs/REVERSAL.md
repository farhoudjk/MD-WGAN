# Correction: the RL result reverses on correctly partitioned data

## What was wrong

Every earlier selection experiment ran on the shipped per-node CSVs. Those are
random splits of one pool, and each was scaled by **its own** mean and standard
deviation. The result: every shard is exactly N(0,1), and the measured spread of
per-node feature means across nodes is **0.000**.

In that setting no selection rule can beat any other, because the candidates are
statistically identical. The earlier conclusions -- "the DQN does nothing",
"greedy wins", "learned selection has nothing to offer" -- were measuring an
artefact of the preprocessing, not a property of the method.

## The fix

`build_spatial_partitions()` cuts the NGSIM source by road segment (the spatial
partitioning the system model actually describes) and fits **one global scaler**
across all nodes, so cross-node heterogeneity survives normalisation. Measured
across-node spread rises from 0.000 to 0.42 sigma.

## The reversed result (segment partitions, 6 seeds, all metrics)

| arm | RANK | RMSE | CRPS | minADE | PICP | conv | work |
|---|---|---|---|---|---|---|---|
| **dqn** | **1.22** | 0.8528 | 0.4030 | **1.097** | **0.717** | 10 | 459 |
| random | 1.44 | 0.8569 | 0.4093 | 1.103 | 0.690 | 8 | 402 |
| genetic | 1.67 | 0.8557 | 0.4088 | 1.109 | 0.690 | 8 | 412 |
| lazygreedy | 2.22 | 0.8577 | 0.4110 | 1.110 | 0.693 | 8 | 408 |
| greedy | 3.44 | 0.8633 | 0.4211 | 1.136 | 0.687 | 11 | 471 |

Paired Wilcoxon against greedy, six seeds:

| metric | DQN vs greedy | p |
|---|---|---|
| RMSE | -0.0105 | **0.031** |
| CRPS | -0.0180 | **0.031** |
| minADE | -0.0395 | **0.031** |
| PICP | -0.0300 | **0.031** |

**The DQN beats greedy significantly on all four accuracy metrics**, and is the
only arm to improve calibration significantly (genetic p=0.84, lazygreedy
p=0.69, random p=1.00). Greedy is now *last* of five.

## What this means

1. **The RL contribution is defensible after all.** It was not the method that
   failed; it was the data preparation. On genuinely heterogeneous shards there
   is something to learn, and the agent learns it.
2. **The condition under which it holds must be stated.** On IID shards the DQN
   is indistinguishable from random -- we measured that directly. The honest
   claim is conditional: *learned selection pays off in proportion to shard
   heterogeneity*, and the paper should present both regimes. That is a stronger
   and more interesting claim than an unconditional one.
3. **Set-level search (GA, lazy-greedy) beats greedy but loses to the DQN.**
   The submodular/max-term objective is a real improvement over per-item
   ranking, but the learned policy captures more.

## Caveats

* Six seeds; p=0.031 is the smallest attainable value for n=6 under Wilcoxon,
  so these are "significant at the resolution the test can deliver", not strong
  evidence. Repeat at 10+ seeds before the claim goes in the paper.
* These runs use `n_generators=2, disc_per_gen=10, top_k=3, rounds=40`, which is
  not the configuration of the main 325-run sweep. The main sweep must be
  re-run on segment partitions before any of its numbers are quoted.
* Differences in `work` are not significant for any arm.
