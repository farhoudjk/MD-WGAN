# Revised manuscript

`mdwgan_revised.tex` — IEEEtran journal class, targeting IEEE IoT-J.

## Building

    cd mdwgan/paper
    pdflatex mdwgan_revised && pdflatex mdwgan_revised

No LaTeX is installed on this machine, so the file has been validated
structurally (balanced environments, no dangling `\ref`/`\cite`, all figure
paths resolve) but not compiled. Compile once before trusting the layout.

Figures and tables are *generated*, not hand-written:

    python3 mdwgan/analysis/figures.py       # fig1..fig7 .pdf
    python3 mdwgan/analysis/paper_tables.py  # tab_sota, tab_ablation, ...

Re-run both after any new experiments; the `.tex` picks the files up by name.

## What changed from the submitted version

**Framing.** The claim is no longer "lowest RMSE". The submitted 0.003 RMSE gap
over HVS-GAN is roughly 24x smaller than the 0.073 seed-to-seed spread of the
method itself, so it cannot be defended. The revision claims cost-to-target and
calibration, both of which are large relative to their variance.

**Title and contributions.** "Reinforcement Learning and Game Theory" is gone
from the title: the DQN does not beat greedy-by-loss on any metric, across five
independent comparisons. It is now reported as a negative result, which is
defensible and which reviewers tend to value.

**Metrics.** CRPS, minADE and PICP alongside RMSE. RMSE is minimised by
predicting the conditional mean, which is the mode collapse adversarial training
exists to prevent, so it cannot be the sole accuracy metric for a generative
model.

**Baselines.** HVS-GAN, FedAvg-GAN and MD-GAN re-implemented under one protocol.
The previous "HVS-GAN" numbers came from a peer-to-peer hybrid-node
implementation, which is not the method of Wu et al.

**Section III-IV.** Every equation now matches the released code: the ensemble
of Eq. (3), the bounded residual head of Eq. (2), online re-estimation of the
coordination parameters, and the best-of-k supervision of Eq. (5).

**Negative results reported, not omitted.** MD-GAN beats us on terminal CRPS;
our traffic per round is higher; the interference coefficient has no measurable
effect; one seed in five never reaches the accuracy target. All are in the text.

## Before submitting — the `%% TODO` markers

Nine, all substantive:

1. Section II can be carried over from the submitted text, with HVS-GAN
   re-described to match what is actually re-implemented.
2. The 0.86 accuracy target was chosen after inspecting the data. Justify it by
   a principled rule, or report a curve over several thresholds.
3. Early Update MD-GAN [24] is not yet ported into this harness; its old numbers
   come from a different pipeline and are not comparable.
4. Reuse Fig. 1 (architecture) from the submitted manuscript.
5. State the delta against the authors' MASCOTS paper [24] explicitly.
6. Merge this bibliography with the submitted one (only new entries are listed).
7. State the RSU deployment area in the Fig. 7 caption (600 m).
8. Push the `mdwgan/` tree and `runs/` records so the repository matches the
   equations in the manuscript. This is the single highest-risk item: the
   current public repository contradicts the submitted paper in five places.
9. Author list, funding and acknowledgements are placeholders.
