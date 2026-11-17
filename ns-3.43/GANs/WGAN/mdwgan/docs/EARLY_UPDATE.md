# Early-update MD-GAN baseline (MASCOTS 2025)

Ported from `scratch/dsrc-simulations4.cc` into the common harness as
`method="early-update-mdgan"` (`core/baselines.py::EarlyUpdateMDGAN`).

**Rule.** Every discriminator trains every round at fixed epochs (E_D=50,
E_G=5). A node *transmits* only when its loss beats its own best so far:
a critic sends feedback only if its loss improved; the generator trains on
whatever feedback arrived (minimum one report) and broadcasts new samples
only if its own loss improved. No selection, learning, coordination or
discriminator swap. Same backbone, supervision, mobility dropout, seeds and
data loading as the other baselines in `runs/sota/`.

**Results** (5 seeds, `runs/sota/early-update-mdgan__seed*.json`):

| Method | RMSE | CRPS | PICP | Rounds to RMSE<=0.86 | Latency (s) | Traffic (MB/round) |
|---|---|---|---|---|---|---|
| Ours (fixed) | 0.776 | 0.396 | 0.72 | 6.4 (5/5) | 2.16 | 16.6 |
| MD-GAN | 0.836 | 0.371 | 0.47 | 36.0 (4/5) | 1.56 | 9.6 |
| Early-update MD-GAN | 0.882 | 0.392 | 0.44 | 52.5 (2/5) | 1.45 | 1.6 |
| HVS-GAN | 0.897 | 0.400 | 0.40 | never | 1.08 | 11.9 |
| FedAvg-GAN | 0.897 | 0.400 | 0.40 | never | 1.02 | 11.8 |

On average 1.5 of 5 critics transmit per round and the generator broadcasts
in 4% of rounds, giving the lowest traffic of any method.

**Objective cost under five weightings** (see `docs/SCENARIO.txt`): the
adaptive framework remains best in all five; early-update is the strongest
fixed method under the balanced, accuracy and network weightings. Gain of
adaptive over the best fixed method: 29-58% (accuracy column p=0.125, others
p=0.062 at n=5).
