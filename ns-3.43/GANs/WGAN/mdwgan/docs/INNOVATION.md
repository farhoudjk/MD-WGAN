# Rebuilding the contribution

## The problem with the current framing

Removing the DQN and reporting the game-theoretic step as near-inert leaves the
paper with no mechanism to call its own. "We removed two things and report
negative results" is an honest paper but a weak one, and a reviewer will say so.
We need a contribution that is (a) genuinely novel, (b) specific to the
vehicular setting, and (c) supported by the data we already have.

## Why the RL failed -- the diagnosis that points to the fix

The DQN's state vector (Eq. 14 of the submitted paper) is seven dimensions of
*loss and resource averages*. Nothing in it describes mobility. So the agent was
asked to choose among vehicles in a world where the only thing distinguishing
them was local loss -- which a one-line greedy rule already captures exactly.
The RL did not fail because deep RL is unsuited to the problem; it failed
because **the state contained no information that a greedy rule did not already
exploit.** Learning cannot beat a heuristic that is sufficient for the state it
is given.

The same explanation covers the HVS-GAN result: its DDPG scheduler is
statistically indistinguishable from uniform random selection in our
re-implementation, for the same reason.

That diagnosis identifies the missing variable.

## The proposed contribution: residence-time-aware participant selection

In a vehicular network the operative question is not "which discriminator is
most informative?" but **"which discriminator will still be in coverage when its
feedback is needed?"** A vehicle about to exit the serving RSU's range returns a
stale update or none at all, no matter how good its local loss is, and the round
proceeds with fewer critics than planned.

No existing federated-GAN scheduler models this. MD-GAN activates everyone;
HVS-GAN schedules on capability and cost but not on *availability*; loss-greedy
and random ignore it entirely. The submitted version of this paper ignored it
too, while describing an environment in which it is the dominant constraint.

We score each candidate by expected useful contribution over the round:

    score_j = informativeness_j  x  P(complete_j)  x  throughput_j

with the survival probability derived from the mobility state the channel model
already maintains. Writing tau_j for residence time and T for the round
duration, and taking residual dwell time as exponential (the standard
memoryless assumption in cell-residence analysis),

    P(complete_j) = exp(-T / tau_j).

The three factors are the three conditions a useful participant must satisfy:
carry information, survive to deliver it, and deliver it over a link fast enough
to matter. Loss-greedy uses only the first; random uses none.

## Why this is defensible rather than another arbitrary heuristic

1. **It is falsifiable and we test it.** If departures do not matter, the
   selector reduces to loss-greedy and the ablation will show it.
2. **It explains the existing negative results** rather than sitting beside
   them. The paper becomes: *learned selection fails here because the state
   omits the governing variable; supply that variable and a simple rule
   outperforms both.* That is a stronger and more interesting claim than "our
   DQN won".
3. **It is specific to IoV.** It would be pointless in datacentre federated
   learning, which is exactly what makes it appropriate for this venue.
4. **It is cheap.** No learning, no replay buffer, no exploration schedule --
   relevant for an RSU coordinating under a latency budget.

## What had to be added to test it honestly

The original harness had no notion of a vehicle failing to return an update, so
every selector faced a static world and any mobility-aware rule would have been
decoration. We added `mobility_dropout`: a selected vehicle whose residence time
expires during the round loses its update, with the loss realised stochastically
from the same channel model that produces the reported latency and traffic.
Measured survival probabilities span 0.06-0.93 across candidates, so the signal
is real and large.

**This must be stated plainly in the paper.** Introducing a failure mode and
then proposing the mechanism that handles it is only legitimate if the failure
mode is independently justified and the baselines are evaluated under exactly
the same conditions. Both hold here: vehicles leaving RSU coverage mid-round is
a documented property of IoV, not an artefact, and every baseline runs with
dropout enabled. The `mobility_dropout=False` ablation reports what happens when
it is switched off.

## Status

Implemented (`ResidenceAwareSelector` in `core/agent.py`, `residence_time` and
`link_quality` in `core/network.py`). Under evaluation against greedy, random
and DQN at a pool size where selection has room to discriminate. **If it does
not win, it does not go in the paper** -- the point of the harness is that this
question has an answer rather than a preference.

## A second, independent finding: the generator/critic step ratio

Tracing where the 2.2x speedup actually comes from produced an uncomfortable
result. The adaptive coordinator **saturates at the top of both strategy sets by
round 5** (E_D=75, E_G=15) and never moves again. After round 5 it is a
constant. What it does, in effect, is raise the generator's share of the round
from the conventional 5 updates to 15.

Holding epochs fixed and sweeping only E_G reproduces the entire effect without
any coordination mechanism:

| E_G | RMSE | CRPS | rounds to target | work to target |
|---|---|---|---|---|
| 5  | 0.8861 | **0.3703** | 35.0 | 810 |
| 10 | 0.8498 | 0.3771 | 30.0 | 768 |
| 15 | **0.8443** | 0.3853 | **22.5** | **707** |

More generator updates per round converge faster **and at lower total cost**
(707 vs 810 work units), while giving up distributional score. That is a real
accuracy/efficiency trade-off controlled by one interpretable parameter.

**The honest implication.** The "game-theoretic coordination" contribution does
not survive this. It is an expensive, hard-to-justify way of discovering
E_G = max, and a reviewer who runs this sweep will reach the same conclusion.
Two options:

1. **Drop the game formulation** and present the generator/critic ratio as what
   it is: a tunable trade-off with a measured Pareto curve. Simple, honest,
   defensible -- but not itself novel; WGAN's n_critic is the same knob read in
   the other direction, and we would be reporting that the conventional 5:1
   critic:generator ratio is wrong for resource-constrained federated settings.
   That is a useful finding but a modest one.

2. **Keep a coordination mechanism but make it do something a constant cannot.**
   Saturation happened because the utilities had no term that grows with E_G.
   A budget-aware formulation -- allocate epochs subject to a per-round
   deadline, which is exactly the vehicular constraint -- would have an interior
   optimum and would genuinely vary with network state.

Option 2 is the stronger paper if it works, and it composes naturally with the
residence-time contribution: both derive from the same measured mobility state.

## Status of the deadline-constrained coordinator

Implemented as `DeadlineCoordinator` in `core/game.py`, with the budget taken
from a quantile of the selected vehicles' residence times.

**It does what it was designed to do.** With converged cost estimates the
allocation tracks the budget rather than saturating:

| budget | allocation |
|---|---|
| unconstrained | (75, 15) |
| 1.0 s | (45, 13) |
| 0.5 s | (45, 2) |

versus the original formulation, which reaches (75, 15) by round 5 and never
moves again.

**But it is not yet carrying its weight.** In a live 25-round run it produced
5 distinct allocations, of which (75, 15) accounted for 19. The measured
residence times (median 5.1 s, p10 1.0 s) are mostly generous relative to the
converged per-round cost (~0.5 s minimum), so the constraint binds only
occasionally.

Three honest options:

1. **Report it as a mechanism that activates under pressure.** Show the
   allocation responding when dwell times are short (dense, fast traffic) and
   reverting to the unconstrained optimum otherwise. This is defensible and
   true, but it is a modest claim: most of the time the mechanism does nothing.
2. **Evaluate in the regime where it matters.** Higher vehicle speeds or smaller
   cells shorten dwell times and make the constraint bind in most rounds. This
   is legitimate *only if the regime is independently motivated* -- e.g. highway
   speeds or small-cell urban deployment -- and if baselines are evaluated in
   the same regime. It must not be presented as the general case.
3. **Drop it.** Keep residence-aware selection as the single contribution and
   present the generator/critic ratio as a measured trade-off rather than a
   mechanism.

Option 2 followed by honest reporting of both regimes is the strongest
defensible position. Option 1 alone is weak. Option 3 is safe.

**What must not happen** is presenting the deadline coordinator as generally
beneficial on the strength of runs where it was inactive in three quarters of
rounds. That is the same category of error -- claiming a mechanism the
implementation does not exercise -- that the original submission made.

## VERDICT: residence-aware selection does not work

Four selectors, pool of 10, k=3, 4 seeds, evaluated with and without mobility
dropout.

| selector | drop rate | RMSE | CRPS | PICP | rounds to target |
|---|---|---|---|---|---|
| residence | **0.158** | 0.8526 | 0.3852 | 0.679 | 18.3 |
| greedy | 0.219 | **0.8454** | 0.3875 | **0.762** | **16.7** |
| random | 0.256 | 0.8554 | 0.3903 | 0.657 | 20.0 |
| dqn | 0.257 | 0.8498 | **0.3775** | 0.720 | 18.3 |

With dropout disabled, every selector reaches the target in 16.7-18.3 rounds --
essentially unchanged.

**The mechanism works; the premise does not.** Residence-aware scoring cuts lost
updates by 28-38% relative to the alternatives, exactly as designed. But that
does not translate into better learning: greedy converges faster and to better
RMSE *while losing more updates*, and disabling mobility dropout entirely barely
moves any selector.

The reason is that the ensemble degrades gracefully. With k=3 active critics,
losing one leaves two informative gradients, and the generator update is
weighted anyway. The cost of a dropped participant is far smaller than the
benefit of choosing an informative one, so the informativeness term dominates --
and that is precisely what greedy already optimises.

**This is dropped from the paper.** It was a well-motivated hypothesis with a
clean mechanism and a decisive negative answer. Recording it here because the
same idea will occur to a reviewer or a future reader, and the answer is now
measured rather than assumed.

## Consolidated position on the contribution

What survived testing:

1. **Context conditioning** is the dominant factor in predictive quality
   (removing it: CRPS +0.21, mode coverage collapses to 0.10). Already in the
   submitted paper's Section III; never implemented in the released code.
2. **Best-of-k supervision** repairs a latent-collapse failure that RMSE cannot
   see (PICP 0.26 -> 0.75). Novel in this setting and well supported.
3. **The generator/critic step ratio** governs a real accuracy/efficiency
   trade-off with an interior optimum (E_G=15: 707 work units, versus 810 at
   E_G=5 and 768 at E_G=25). This is what actually produced the claimed
   speedup -- not the game-theoretic coordination.
4. **Two negative results with a shared explanation**: neither our DQN nor
   HVS-GAN's DDPG beats a trivial heuristic, because the state they observe
   contains nothing a heuristic does not already exploit. Supplying the one
   variable they omit -- availability -- does not rescue them either, because
   the ensemble is robust to losing participants.

What did not survive: game-theoretic coordination as a mechanism (it saturates
and is reproducible by a constant), and residence-aware selection.

**Recommended framing.** The paper's contribution is an *evaluation* result as
much as a mechanism: in resource-constrained federated GAN training for IoV,
coordination machinery -- learned or game-theoretic -- is not what determines
outcomes; conditioning, supervision design, and the generator/critic budget are.
That is a useful, falsifiable, and well-evidenced claim, and it is supported by
63 configurations at 5 seeds with paired significance testing. It is a weaker
*novelty* claim than the submitted version but a far stronger *evidence* claim,
and it does not collapse when a reviewer runs the code.
