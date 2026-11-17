# Comparison methods

All baselines are trained on the same NGSIM partitions, with the same
conditional generator/critic backbone, the same optimiser settings, the same
seeds, and the same evaluation protocol as the proposed framework. Only the
*coordination strategy* differs, so a difference in results is attributable to
coordination rather than to architecture, data handling or tuning.

## What was wrong with the previous comparison

The submitted paper compares against "HVS-GAN [19]". The code that produced
those numbers (`run_hybrid_node.py` + `ddpg_agent.py`) implements a
**peer-to-peer hybrid node** -- every node holds both a generator and a
discriminator and exchanges samples with neighbours -- driven by DDPG.

That is not the method of Wu, Lin & Wang. HVS-GAN is a *server-coordinated
federated* GAN: vehicles train full local GANs, a central server aggregates
them, and a DDPG agent schedules which heterogeneous vehicles participate.
The previous baseline shares only the use of DDPG.

Labelling an in-house hybrid-node method as a published baseline is the kind of
claim a reviewer familiar with [19] will check, and it is more damaging than
having no baseline at all. It has been reimplemented.

## HVS-GAN (Wu, Lin & Wang, IEEE IoT-J 2024) -- `method="hvs-gan"`

- Each selected ICV trains a **complete local GAN** (its own generator and
  discriminator) on its own shard.
- The server aggregates local generators by **FedAvg**, weighted by local
  sample count, and broadcasts the global model each round.
- **Heterogeneity**: each vehicle draws a fixed capability in [0.4, 1.0] and a
  participation cost in [0.5, 1.5]. Local epochs scale with capability, so a
  weaker vehicle genuinely completes less work.
- **Scheduling**: a DDPG actor-critic (OU exploration, target networks, soft
  updates) scores every vehicle from a state of per-vehicle
  (loss, capability, cost, staleness); the top-k are scheduled. Its reward
  trades training progress against the cost of the vehicles chosen.

Contrast with the proposed framework: one discriminator per *local* generator
rather than a multi-discriminator ensemble feeding a shared generator;
centralised aggregation rather than per-RSU control; continuous-action
scheduling rather than discrete subset selection.

## FedAvg-GAN -- `method="fedavg-gan"`

Identical to HVS-GAN except participants are sampled uniformly at random.
Isolates exactly what the DDPG scheduler contributes: any gap between
`hvs-gan` and `fedavg-gan` is the scheduling, nothing else.

## MD-GAN (Hardy, Le Merrer & Sericola, IPDPS 2019) -- `method="md-gan"`

The canonical multi-discriminator reference the proposed framework extends:
one central generator, one discriminator per worker, **every worker active
every round**, with discriminators periodically swapped between workers so none
overfits its local shard. Isolates the contribution of *selective*
participation, since this is the same architecture without selection.

## Early Update MD-GAN (ref [24])

The authors' own prior work. The existing implementation is legitimate -- it is
their own published method -- but was measured under the old pipeline. It
should be ported into this harness before the final table so every row is
produced under one protocol.

## Reporting note

Report these as *our reimplementations*, state the hyperparameters chosen where
the source papers leave them unspecified, and say plainly that absolute numbers
may differ from the original publications because the dataset, backbone and
evaluation protocol are held common here by design. That framing is both honest
and stronger: it makes the comparison about coordination strategy, which is
what the paper is actually claiming.
