# MD-WGAN: diagnosis of the rejections and the revision plan

Written for the author. This is an internal working document, not paper text.

## 1. What a reviewer sees today

The submitted manuscript makes three architectural claims — RL-based
discriminator selection, game-theoretic epoch allocation, and multi-discriminator
ensemble feedback — and supports them with a single end-to-end comparison against
three baselines. Reviewers of TNSM-class venues reject this pattern for reasons
that are largely independent of how good the idea is. Four specific problems,
ordered by how likely each is to be fatal.

### P1. No ablation separates the three contributions
Table/figures report only "proposed vs. baselines". Nothing shows what the DQN
contributes over a cheaper selection rule, what the game-theoretic step
contributes over a fixed epoch schedule, or what the ensemble contributes over a
single critic. A reviewer cannot attribute the reported gain to any of the three
claimed novelties, so the contribution is unverifiable as stated. This alone is
a standard reject-or-major-revision trigger.

### P2. The headline differences are inside the noise
The accuracy claim is RMSE 1.029 (ours) vs 1.032 (HVS-GAN) vs 1.044 (Early
Update MD-GAN) vs 1.078 (random). A 0.003 gap over the nearest baseline is
reported with no standard deviation, no confidence interval, and no significance
test, while Sec. V-A states results are means over five seeds — so the variance
was available and omitted. A reviewer reads "0.3% better, variance withheld" as
a negative result. The resource claims (~28% CPU, ~6% memory) are the ones that
are plausibly real and large; the paper leads with the weakest number it has.

### P3. The released code contradicts the paper
This is the most dangerous one, because TNSM reviewers do check released
artifacts, and the repository link is in Sec. V-E.

| Paper claims | What the released code did |
|---|---|
| Eq. (24): generator trains against a weighted ensemble `D_ens = (1/\|S\|) Σ w_i D_j` | `run_generator2.py` loaded `disc_ids[0]` only — one critic, no ensemble |
| Sec. IV-D: game-theoretic coordination adapting to network state | `find_nash_equilibrium()` took no arguments and used only hardcoded constants — it returns the *same* (E_D, E_G) every round, i.e. a constant, not a game |
| Eq. (16): reward includes relative GAN-loss improvement | `compute_reward(metrics, prev_metrics)` was called as `compute_reward(metrics, metrics)`, making the improvement term identically zero |
| Sec. III-A2: generator predicts `ŷ = G(c, z)` from trajectory context | Generator consumed pure noise `gen(noise)`; no context ever entered the model |
| DQN selects a subset, learns from outcomes | Replay used `action[0]` only (credit to one of three selected), no target network, and `next_state = state` |

Any reviewer who opens the repo finds that the central mechanisms are either
absent or inert. The revision must close this gap — which is what the new
`mdwgan/` implementation does.

### P4. Latency and overhead are not what the paper says they are
"Network Traffic Overhead (Bytes)" was computed as `os.path.getsize()` of CSV
files written to local disk. That measures CSV serialisation, not vehicular
channel usage, and it is invariant to every network parameter in Table II.
Likewise the reported latency was wall-clock Python runtime including model
construction and disk I/O. The paper's own Eqs. (2)-(3) (path loss, Shannon
capacity) are never used to produce any reported number.

## 2. What the revision provides

### Reimplementation (`mdwgan/core/`)
Every mechanism in Sections III-IV is now actually implemented, and each is
independently switchable so it can be ablated:

- `models.py` — conditional `G(c, z)` and conditional critic `D(c, y)`, with a
  *bounded* residual head. (The unbounded version diverged: deltas up to 2300σ
  on held-out contexts, test RMSE ~95 on data with |y| < 3.)
- `trainer.py` — the Eq. (24) ensemble with uniform or performance weighting;
  `use_ensemble`, `use_game`, `adaptive_game`, `conditional`, `use_adversarial`,
  `lambda_sup`, `selector` switches.
- `game.py` — the coordinator now re-estimates effectiveness rates κ and
  per-epoch costs online from realised loss improvement and measured epoch time,
  so the allocation genuinely responds to state. `adaptive=False` reproduces the
  old static behaviour as a control arm. Also reports how often the two-pass
  search lands on a mutual best response, which is the honest way to describe an
  approximate equilibrium.
- `agent.py` — Double-DQN with a target network, set-valued credit assignment,
  genuine successor states, and a reward whose improvement term is non-zero.
  Plus four non-learning selectors (random, greedy-by-loss, round-robin, all) as
  the comparison set.
- `network.py` — latency and traffic derived from Eqs. (2)-(3) with vehicles
  moving between rounds; round latency is the slowest link, not the mean.

### A note on what the fix revealed
Training with a purely adversarial objective does not minimise next-step
prediction error — a WGAN critic rewards distributional realism, which does not
constrain per-sample error. With `lambda_sup = 0` the RMSE oscillated between
1.2 and 10.9 across rounds and never beat a trivial persistence baseline. The
framework therefore uses a supervised term alongside the adversarial one
(standard conditional-GAN practice), and `lambda_sup` is itself swept so the
paper can state exactly how much each part contributes. **This must be disclosed
in the revision**: claiming pure-WGAN training while using a reconstruction loss
would be the same class of error as P3.

### Experiments (`mdwgan/experiments/`, 230 runs = 46 arms x 5 seeds)
1. **Ablations** — full; leave-one-out (no-RL / no-game / no-ensemble);
   leave-two-out; all-off; plus `static-game`, `unconditional`, `no-supervised`,
   `supervised-only`, `ensemble-weighted`.
2. **Selection baselines** — DQN vs greedy-by-loss, round-robin, random-k, all.
   Greedy-by-loss is the one that matters: it is nearly free, and if the DQN
   cannot beat it, the RL contribution is not defensible.
3. **Sensitivity** — subset size k, pool size, generator count, reward
   weightings (the four Sec. V-F7 scenarios), λ_sup, and γ_nash.
4. **Non-IID severity** — natural spatial partitions vs IID reshuffling.
5. **ns-3 communication sweep** (`mdwgan-comm.cc`) — measured 802.11p latency
   and byte overhead vs vehicle count, subset size, RSU count and payload.

### Statistics (`mdwgan/analysis/`)
Paired Wilcoxon signed-rank tests across common seeds (paired because the seed
effect is large relative to the arm effect — in pilot runs every seed-1 arm
landed within 0.002 RMSE of every other), Holm correction across the family of
comparisons, bootstrap CIs, and Cohen's d_z. Tables emit directly to LaTeX with
significance markers.

## 3. How to write the revision

**Lead with resources, not accuracy.** The honest story is: *comparable accuracy
at substantially lower resource cost*. The ~28% CPU reduction is a real,
defensible, large effect; the 0.003 RMSE gap is not. Reframing this way also
removes the reviewer's strongest objection, because you are no longer asking
them to believe a difference that your own variance would swallow.

**Report negative and null results.** If the DQN ties greedy-by-loss on accuracy
but wins on resource cost, say exactly that. If the game-theoretic step matters
only under high heterogeneity, say that and show the non-IID sweep. A paper that
reports where its mechanism does *not* help reads as credible; one that reports
uniform wins at 0.3% margins does not.

**State the approximation honestly.** The coordination step is a two-pass
best-response search over discretised strategy sets, not a converged Nash
equilibrium. Sec. IV-D3 already half-admits this; make it unambiguous and report
the measured agreement rate.

**Disclose the supervised term** and the bounded residual head, with the
ablations that quantify both.

## 4. Order of work
1. Run the sweep; read `table_ablation.tex` and `table_selection.tex`.
2. Decide the framing from what the ablations actually show (not before).
3. Rewrite Sec. V around ablations + significance; move the baseline comparison
   to a supporting role.
4. Update Secs. III-IV so every equation matches the released code.
5. Re-record the repository state referenced in Sec. V-E.
