# Empirical findings as the sweep lands

Updated during the 230-run sweep. Numbers here are measured, not projected.

## F1. The accuracy claim is inside the noise (confirmed, n=5)

Five seeds of the full framework, identical configuration, varying only the RNG:

| seed | final RMSE |
|---|---|
| 0 | 0.8848 |
| 1 | 0.8114 |
| 2 | 0.8479 |
| 3 | 0.8209 |
| 4 | 0.8126 |

mean 0.8355, sd 0.0312, spread (max-min) **0.0734**.

Compare against the gaps the submitted paper reports *between methods*:

| comparison | claimed gap | vs. within-method spread |
|---|---|---|
| proposed 1.029 vs HVS-GAN 1.032 | 0.003 | **24x smaller than the spread** |
| proposed 1.029 vs Early Update 1.044 | 0.015 | ~5x smaller than the spread |
| proposed 1.029 vs random 1.078 | 0.049 | comparable to the spread |

**Consequence.** With five seeds, the manuscript cannot distinguish its method
from HVS-GAN or Early Update MD-GAN on RMSE. Reporting those numbers to three
decimals with no variance invites exactly the rejection the paper has been
getting: a reviewer who computes the spread sees that the ordering could flip
by rerunning with different seeds.

Note the absolute RMSE here (~0.84) is not comparable to the paper's ~1.03: the
evaluation is now genuine held-out next-step prediction on per-track splits,
against a conditional generator. The relevant quantity is the *ratio* of
between-method gap to within-method spread, and that is damning either way.

**Action.** Do not lead with accuracy. State comparable accuracy explicitly, with
CIs, and let the resource efficiency carry the contribution. If the ablations
show a component improves resource cost significantly while leaving accuracy
statistically unchanged, that is a publishable and honest claim; "lowest RMSE
among all evaluated methods" is not.

## F2. Runs are exactly reproducible
Same seed reproduces identical RMSE to 10 decimal places, identical
discriminator subsets, and identical epoch allocations. Several sensitivity arms
deliberately duplicate the `full` configuration as sweep centre points, which
doubles as a built-in consistency check on the harness.

## F3. (pending) Component attribution
Awaiting the ablation suite. The question each arm answers:
- `no-rl` vs `full`: does DQN selection beat random-k?
- `sel-greedy` vs `sel-dqn`: does it beat a nearly-free heuristic? **This is the
  one that decides whether the RL contribution is defensible.**
- `no-game` / `static-game`: does coordination beat a fixed schedule, and does
  its *adaptivity* matter beyond the particular (E_D, E_G) it lands on?
- `no-ensemble`: does Eq. (24) beat a single critic?
- `no-supervised` / `supervised-only`: how is credit split between the
  adversarial game and the reconstruction term?

## F4. Two resource metrics were measuring the harness, not the method (fixed)

The first 15 runs exposed a measurement bug that would have produced
meaningless efficiency numbers — the same class of error as P4 in the
revision plan, and worth recording because the original code had it too.

**Network traffic was a constant.** Every arm reported exactly 1,148,549 bytes
per round, because traffic was computed as (parameter count) x (subset size)
and the subset size is fixed by `top_k`. A metric that cannot vary cannot
support a claim about communication efficiency. It now scales with the number
of synchronisation steps actually taken, so an arm that trains longer locally
between syncs genuinely exchanges less data.

**CPU was measuring thread saturation.** Every arm reported ~199%, which is
"two torch threads running flat out" — a property of the `--threads 2` setting,
not of the algorithm. On a shared, oversubscribed host, wall-clock CPU
percentage is a measurement of the scheduler. The comparable quantity is now
algorithmic work in batch-gradient-step units, counted as the critic and
generator updates actually executed, with generator steps weighted by the
ensemble size they query. Measured process CPU is retained as
`cpu_usage_measured` for reference.

After the fix the metrics discriminate as they should:

| configuration | work units | traffic (per round) | RMSE |
|---|---|---|---|
| full (top-k=3) | 14.4 | 17.2 MB | 0.926 |
| all discriminators | 24.0 | 28.7 MB | 0.928 |
| top-k=1 | 4.8 | 5.7 MB | 0.927 |
| fixed epochs (no game) | 10.6 | 5.7 MB | 0.928 |

and the unified objective of Eq. (8) separates them (0.237 for top-k=1 vs
0.552 for all-discriminators) while RMSE stays flat — which is precisely the
accuracy/resource trade-off the paper should be arguing.

The 15 runs completed under the old metrics were discarded and the sweep
restarted; they are kept in `runs_oldmetrics/` only as a record.

## F5. Early seed-paired signal (preliminary, n=2-4, pre-fix metrics)

From the discarded run set, pairing by seed to remove the seed effect:

- `no-rl` (random instead of DQN): **-0.0010 RMSE**, i.e. random selection was
  marginally *better*, consistently across all four seeds. The DQN was not
  buying accuracy.
- `no-game` (fixed epochs): **+0.0120 RMSE**, consistently worse on both seeds.

If this survives the full sweep, the contribution ranking in the paper needs
revising: the game-theoretic coordination is the mechanism carrying accuracy,
and RL selection must justify itself on resource cost instead — which is now
measurable, since F4 is fixed. Treat as a hypothesis until the 5-seed
significance tests land.

## F6. The ns-3 communication model had a silent 98 dB error (fixed)

`mdwgan-comm.cc` initially reported zero successful transmissions: every round
logged `latency=-1`, `bytes_up=0`. UDP reported successful sends and a valid
route, but no ARP request was ever issued and the PHY showed no activity.

Root cause: `YansWifiChannelHelper::Default()` already installs a Friis
propagation loss model, and `AddPropagationLoss(...)` **chains** a second model
onto it rather than replacing it. The log-distance model was therefore applied
*on top of* Friis, double-counting attenuation — measured receive power was
-163.8 dBm at 50 m where the link budget predicts -65.8 dBm, roughly 98 dB of
spurious loss, far below the -101 dBm receiver sensitivity. Every link in the
simulation was dead, silently.

This is worth flagging because the same idiom appears in `dsrc-simulations*.cc`.
Any latency or overhead figure taken from a simulation with a chained loss model
describes a network where nothing is received.

Fixed by constructing the channel explicitly (no `Default()`) and setting the
delay model directly. Also corrected: `ReferenceLoss` now 47.86 dB, the
free-space loss at 1 m for the 5.9 GHz DSRC band rather than ns-3's 46.6777 dB
default, which is the 2.4 GHz figure.

A second bug: replies were attributed to "whichever round for this RSU still
wants replies", which mis-assigned late arrivals and left later rounds showing
-1. Replies now carry a `ReplyTag` naming their round, and outstanding requests
are queued per vehicle rather than overwritten.

**Verified behaviour after the fix** (12 vehicles, 2 RSUs, k=3, 10 rounds):

| deployment area | delivery ratio | median round latency |
|---|---|---|
| 200 m | 68% | 15.1 ms |
| 600 m | 73% | 16.8 ms |
| 1500 m | 27% | 6.0 ms |

Latencies of 12-25 ms are physically sensible for an 802.11p synchronisation
round, and delivery degrades as vehicles leave RSU coverage — the model now
responds to the parameters it is supposed to respond to. (The sub-100% delivery
at short range is contention between simultaneous broadcasts, which is realistic
for this access scheme and should be reported rather than tuned away.)

Note the contrast with the paper's reported latency of ~9.9 s: that figure was
Python wall-clock time including model construction and disk I/O, not a network
latency. The two quantities are not comparable, and the revision should say so
explicitly rather than appearing to improve the same number by three orders of
magnitude.

## F7. Judged on the paper's own objective, the full framework is not the best configuration

Ranking every arm by Eq. (8) (5 seeds each) rather than by RMSE alone:

| arm | Eq.(8) cost | RMSE | work | traffic MB | mode cov. |
|---|---|---|---|---|---|
| **static-game** | **0.333** | 0.8375 | 10.4 | 10.34 | 0.933 |
| no-ensemble | 0.435 | 0.8412 | 15.4 | 17.23 | 0.933 |
| no-game | 0.440 | 0.8473 | 10.6 | 5.74 | 0.944 |
| **full (proposed)** | **0.449** | 0.8355 | 17.3 | 17.23 | 0.923 |
| unconditional | 0.475 | 1.0036 | 17.3 | 17.23 | 0.101 |

`full` ranks 11th of 14. Re-normalising all metrics and re-weighting under four
scenarios (balanced, accuracy-first, resource-first, delay-first) it ranks
8th-11th in every one; `no-game` wins all four. The ordering is not an artefact
of one weighting choice.

**Why.** The adaptive coordinator drives the epoch allocation to the top of both
strategy sets (mean E_D 73.2, E_G 14.9, versus 45/9 static and 50/5 fixed). That
roughly triples compute and traffic relative to the cheapest arms.

**But it is buying something real.** Convergence trajectory:

| arm | round 10 | round 30 | round 60 |
|---|---|---|---|
| full | 0.9051 | 0.8432 | 0.8355 |
| static-game | 0.9135 | 0.8594 | 0.8375 |
| no-game | 0.9156 | 0.8998 | 0.8473 |

The coordinator converges markedly faster (0.843 vs 0.900 at round 30) and to a
slightly better final point. The gap has largely closed by round 60, so on a
long run the extra compute is poorly repaid; on a short or interrupted one -- a
vehicle in coverage for a limited window, which is the paper's own setting -- it
is repaid well.

**Honest framing for the paper.** The contribution is not "best accuracy at
lowest cost". It is: *the coordinator converts compute budget into convergence
speed, and the adaptive variant does so automatically from measured state.*
Report the Pareto front, state where each operating point wins, and let the
deployment pick. A paper that shows its own method losing under some weightings
and explains exactly when it wins is far more credible than one claiming a
uniform victory at 0.3% margins.

**Caveat on the metric.** `cpu_usage` counts gradient steps, so any arm that
simply trains less scores better on three of the five terms. Only RMSE and mode
coverage push back, and both are weak discriminators at 60 rounds. The ranking
says the current alpha weighting does not justify the coordinator's cost -- not
that the coordinator is useless. A deployment that values convergence speed (or
runs fewer rounds) would weight it differently.

## F8. RMSE and mode coverage disagree, and that matters

`supervised-only` has the best RMSE of any arm (0.8181) and the worst mode
coverage (0.906): it regresses toward the conditional mean, which is exactly the
failure RMSE cannot detect. `none` and `no-game-ensemble` have the best coverage
(~0.975). `unconditional` collapses outright (coverage 0.101).

This is the strongest available argument for keeping the adversarial objective:
it does not win on RMSE, it wins on distributional fidelity. The revision should
report both metrics side by side and make that argument explicitly, rather than
competing on RMSE where a plain regressor beats the GAN.

## F9. The generator was ignoring its latent input entirely (fixed)

Measured directly: the standard deviation of the generator's output across 40
draws of z, against the standard deviation of the real data, per feature:

| | speed | accel | lane | spacing | pos_x |
|---|---|---|---|---|---|
| generator sample spread | 0.035 | 0.035 | 0.025 | 0.025 | 0.025 |
| real data spread | 1.053 | 0.999 | 0.881 | 1.038 | 1.076 |
| **ratio** | **0.033** | **0.035** | **0.028** | **0.024** | **0.023** |

The generator's outputs varied by ~3% of the data's variation. It produced
essentially the same prediction for every latent draw: a deterministic
regressor, not a generative model. PICP confirmed it independently -- the
model's own 90% prediction interval contained 12% of the true values.

**Cause.** The supervised term penalises every draw against the single observed
future, so the optimal response is to put all draws on the conditional mean.
The adversarial term is too weak to counteract it. This is the standard
conditional-GAN collapse, and it means the WGAN machinery -- the paper's whole
subject -- was contributing almost nothing.

**Fix.** Two standard remedies, both now implemented and ablatable:
  * `sup_mode="best_of_k"`: winner-take-all supervision. Only the closest of k
    draws is penalised, so the remaining draws are free to cover alternative
    futures.
  * `lambda_div`: mode-seeking regulariser (MSGAN), which directly rewards
    output diversity per unit of latent distance.

Effect at 10 rounds (1 generator, 3 discriminators):

| config | RMSE | minADE | CRPS | PICP | coverage |
|---|---|---|---|---|---|
| baseline | 0.9582 | 1.3801 | 0.4322 | 0.283 | 0.988 |
| + mode-seeking (0.5) | 0.9582 | 1.3473 | 0.4240 | 0.444 | 1.000 |
| **+ best-of-k** | **0.9570** | **1.1546** | **0.4066** | **0.630** | 0.978 |

Best-of-k improves every probabilistic metric at unchanged RMSE -- PICP more
than doubles. This is the single most important fix found: it is what turns the
framework back into an actual generative model, and it is invisible to the
RMSE/MAE metrics the paper currently reports.

## F10. SOTA comparison under a common protocol (5 seeds)

All methods on the same NGSIM partitions, backbone, seeds and evaluation.

| method | RMSE ± sd | CRPS | minADE | PICP | cov |
|---|---|---|---|---|---|
| md-gan | 0.8389 ± .032 | **0.3718** | 1.1327 | 0.472 | 0.940 |
| **ours** | **0.8332 ± .030** | 0.3974 | **1.0687** | **0.750** | 0.940 |
| ours-dqn | 0.8355 ± .031 | 0.4003 | 1.2398 | 0.266 | 0.929 |
| hvs-gan | 0.9014 ± .034 | 0.4015 | 1.3043 | 0.399 | 0.995 |
| fedavg-gan | 0.9014 ± .034 | 0.4021 | 1.3050 | 0.390 | 0.990 |

| method | rounds to RMSE<=0.86 | seeds reaching | work spent | traffic MB | latency |
|---|---|---|---|---|---|
| **ours** | **16.5** | 80% | **1059** | 16.78 | 2.42 |
| ours-dqn | 17.2 | 80% | 1109 | 17.23 | 2.01 |
| md-gan | 36.5 | 80% | 2570 | 9.57 | 1.47 |
| hvs-gan | never | 0% | -- | 11.96 | 1.01 |
| fedavg-gan | never | 0% | -- | 11.53 | 0.99 |

**What can be claimed.**
* Beats HVS-GAN on RMSE by 0.068 -- about twice the seed s.d., so unlike the
  original 0.003 claim this one is real.
* Best minADE and by far the best calibration (PICP 0.750 against 0.27-0.47).
* Reaches the accuracy target 2.2x faster than MD-GAN and at 2.4x less compute;
  the two federated baselines never reach it within 60 rounds.

**What must also be reported.**
* **MD-GAN wins on CRPS** (0.3718 vs 0.3974). The framework's advantage is the
  cost of reaching accuracy, not distributional score at convergence. Do not
  lead on CRPS alone.
* Traffic is *higher* than MD-GAN (16.8 vs 9.6 MB) and latency is the worst of
  the five. These are genuine losses and should appear in the table.
* "80% reaching" means one seed in five never hit the target. Report the
  fraction; do not average it away.

**Incidental finding about HVS-GAN.** `hvs-gan` and `fedavg-gan` agree to four
decimals on every metric: the DDPG scheduler contributes nothing over uniform
random participant selection. That is a result about the baseline's own
mechanism, and it mirrors what we found for our DQN -- in this setting, learned
participant selection does not beat trivial selection.

## F11. Two sensitivity arms could not run

`pool8` (needs 32 nodes) and `gens6` (needs 30) exceed the 20 available NGSIM
partitions, so all 10 of their seeds failed with a missing-dataset error. This
is a data limit, not a defect: the scalability sweep is bounded at 20 nodes
(4 generators x 5 discriminators). Either partition the NGSIM source more
finely or state the bound explicitly in the scalability section.

## F12. The first ns-3 sweep measured a disconnected network (rerun)

The initial communication sweep ran at the default 1500 m x 1500 m deployment
(Table II's simulation area) with 4 RSUs. Delivery ratios came out at 1-8%:
at that spread most vehicles are simply outside 802.11p range of any RSU, so
the sweep characterised a network that barely connects rather than the
federated round itself. Latencies looked deceptively good (3-6 ms) precisely
because only the few near-range links ever completed.

Rerun at 600 m, where RSU coverage overlaps the vehicle population, which is
the regime the framework actually targets. Results kept in `ns3_runs/`; the
1500 m set is retained in `ns3_runs_1500m/`.

**Implication for the paper.** The 1500 m figure in Table II is a *mobility*
area, not a connectivity-feasible single-cell deployment. If the paper reports
communication results at that scale it must either model more RSUs (so coverage
is continuous) or state that vehicles spend much of the time disconnected and
report the delivery ratio alongside latency. Reporting latency without delivery
ratio is misleading: dropping 90%+ of exchanges lowers mean latency.
