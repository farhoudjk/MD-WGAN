# Which accuracy metric should this paper report?

## The problem with RMSE as the headline

The submitted paper reports RMSE and MAE as its accuracy claim. For a
*conditional generative* model this is the wrong primary metric, for a reason
that is not a matter of taste:

**RMSE is minimised by predicting the conditional mean E[y|c].** A generator
that ignores its latent input z entirely — producing the same output for every
draw — achieves the best possible RMSE. That behaviour is mode collapse, the
exact failure WGAN exists to prevent. So the paper's headline metric actively
rewards the failure its method claims to fix.

This is not hypothetical here. In our runs `supervised-only` (no critic at all,
pure regression) achieves the **best RMSE of any arm** (0.818) while having the
**worst mode coverage** (0.906). Competing on RMSE means competing on ground
where a plain regressor beats the GAN — and a reviewer will point that out.

## The metrics now reported

| metric | what it measures | why it belongs |
|---|---|---|
| **RMSE / MAE** | point error of the conditional-mean estimate | retained for comparability with prior work; no longer the headline |
| **minADE_k** | error of the *closest* of k sampled futures | the standard multi-modal trajectory metric (Social-GAN and successors). Credits a model for covering the plausible set rather than averaging it |
| **CRPS** | quality of the whole predictive distribution | a *strictly proper* scoring rule: uniquely minimised by the true conditional law, so it cannot be gamed by collapsing to the mean. This is the principled replacement for RMSE |
| **PICP** | fraction of targets inside the model's own 90% interval | calibration. 0.90 is ideal; far below means overconfident |
| **mode coverage / JS** | marginal distributional fidelity | detects collapse directly |

CRPS is the one that should carry the accuracy claim. It is standard in
probabilistic forecasting, it is strictly proper, and unlike RMSE it rewards a
model for being right *about its own uncertainty*.

## Convergence metrics (where this framework is actually strong)

Accuracy at round 60 barely separates the arms — every ablation is within seed
noise on RMSE. What separates them is **how fast and how cheaply they get
there**:

- **rounds-to-target**: rounds to first reach a fixed absolute RMSE threshold,
  the same threshold for every arm. Reported with the fraction of seeds that
  reach it at all (some never do — that must not be averaged away).
- **work-to-target**: cumulative gradient-step work spent reaching it. This is
  the metric on which the framework wins outright, and it is the one a
  resource-constrained IoV deployment actually cares about.
- **late-training stability**: s.d. of RMSE over the final 20% of rounds.

A fixed threshold matters: a "rounds to within 2% of own best" definition
flatters an arm that converges quickly to a bad point. Our `unconditional` arm
scored 1.0 round on that definition purely because it collapsed immediately.

**Caveat to resolve before submission.** The 0.86 threshold used so far was
chosen after seeing the data. Fix it by a principled rule — e.g. the
persistence-baseline error, or a pre-registered fraction of it — or report a
curve across several thresholds, so the choice cannot be read as tuned.

## Recommended reporting structure

Primary: **CRPS** and **work-to-target**.
Secondary: minADE_k, mode coverage, PICP.
Retained for comparability: RMSE/MAE, explicitly described as comparable rather
than superior, with confidence intervals.

Report all of them, including the ones where the method loses. Selecting the
metric that happens to win is the single easiest thing for a reviewer to catch,
and it is what the current submission looks like it is doing.
