# Read this first

Everything ran unattended overnight. Start here, then open RESULTS.md.

## One command

    cat mdwgan/docs/RESULTS.md        # the numbers
    cat mdwgan/overnight.log          # did it finish

## What changed while you slept

**Five real bugs found and fixed.** Each is documented in FINDINGS.md with the
evidence. In short:

1. The generator **ignored its latent input** -- sample spread was 3% of the
   data's. It was a deterministic regressor, not a GAN. Its own 90% prediction
   interval contained 12% of true values. Fixed with best-of-k supervision:
   PICP went 0.27 -> 0.75, minADE improved 15%.
2. **ns-3 channel was dead.** `YansWifiChannelHelper::Default()` plus
   `AddPropagationLoss` chains two loss models and double-counts ~98 dB, so no
   packet was ever received. The same idiom is in your `dsrc-simulations*.cc`.
3. **Network traffic was a constant** and **CPU measured thread count**, not
   algorithmic cost. Both now measure the method.
4. The released code **contradicted the paper** in five places (ensemble,
   adaptive coordination, reward, G(c,z), DQN credit assignment).
5. **"HVS-GAN" was not HVS-GAN** -- see below.

## The three things that decide your paper

**1. Your accuracy claim is noise.** Seed-to-seed spread is 0.073 RMSE; your
claimed gap over HVS-GAN is 0.003. 24x smaller. Every ablation is statistically
indistinguishable from the full method on RMSE. Do not lead with accuracy.

**2. The DQN does not work.** Greedy-by-loss beats it on every metric and is
nearly free. Five independent comparisons agree. Either drop the RL claim or
report it as a negative result -- the latter is publishable and reviewers
respect it.

**3. Your baseline was mislabelled.** The "HVS-GAN [19]" numbers came from
`run_hybrid_node.py`, a peer-to-peer hybrid node with DDPG. Wu et al.'s method
is server-coordinated federated GAN with FedAvg and DDPG *scheduling*. They
share only the acronym. This is the single most likely thing to get the paper
desk-rejected on a reviewer check. HVS-GAN, FedAvg-GAN and MD-GAN (Hardy et
al.) are now properly implemented under one protocol -- see BASELINES.md.

## The paper I would write

Not "lowest RMSE". That claim cannot survive its own variance. Instead:

> A resource-aware multi-discriminator WGAN that reaches target accuracy
> ~2.3x faster and at lower total compute than ablated variants, with
> calibrated predictive uncertainty -- and a negative result: deep RL does not
> beat a one-line greedy heuristic for discriminator selection.

Lead with **CRPS** (strictly proper -- cannot be gamed by collapsing to the
mean, unlike RMSE) and **work-to-target**. Report RMSE as comparable, with
confidence intervals. See METRICS.md for why.

## Honest caveats to resolve before submitting

- The 0.86 convergence threshold was chosen after seeing data. Fix it by a
  principled rule or report a threshold curve.
- Early Update MD-GAN [24] is not yet ported into the new harness; that row
  still needs re-running under the common protocol.
- Baseline absolute numbers differ from the published papers because dataset,
  backbone and protocol are held common by design. Say so explicitly.
- `work` counts gradient steps, so arms that train less score well on it; read
  it with CRPS and convergence, not alone.

## Files

| file | what |
|---|---|
| RESULTS.md | best configuration, all arms, significance tests |
| FINDINGS.md | every bug, with evidence |
| BASELINES.md | what each comparison method is, and what was wrong before |
| METRICS.md | why CRPS replaces RMSE as the headline |
| REVISION_PLAN.md | diagnosis of the rejections, how to rewrite Section V |
| ATTRIBUTION.txt | per-component contribution table |
| ../paper_assets/ | LaTeX tables and PDF figures |
