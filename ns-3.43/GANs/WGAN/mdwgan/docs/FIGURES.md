# Figure set (IEEE IoT-J)

Vector PDF, drawn at final size for the journal's two-column layout, so LaTeX
never scales them -- scaling a figure rescales its type and breaks the match
with body text. Include with `\includegraphics[width=\columnwidth]{...}` for
single-column figures and `width=\textwidth` inside `figure*` for the rest.

**Typography.** Nimbus Roman (Times-metric) at 8 pt axis text / 7 pt legend, so
figure type matches the 8 pt caption size. Fonts are embedded and subset
(verified with `pdffonts`: `emb yes, sub yes`) -- IEEE rejects figures with
unembedded fonts, which matplotlib's default Type 3 output produces.

**Marks.** Hairline 0.6 pt axes, top and right spines removed, recessive grid,
1.1 pt lines with 3.2 pt markers, 95% CI bands or error bars throughout. Panel
letters (a)/(b)/(c) sit centred below each panel.

**Colour.** Four-hue categorical palette (blue #2f62b8, orange #cf5c0a, teal
#0e8f7a, purple #8c2f91) that passes an all-pairs colour-vision-deficiency
check; a reserved red (#b02030) marks reference lines only, never a data series.
Every series also carries a distinct marker, so identity never depends on colour
-- the figures stay readable in greyscale print.

| file | width | what it shows |
|---|---|---|
| `fig1_convergence.pdf` | full | RMSE and CRPS vs round, 4 methods, 95% CI |
| `fig2_cost_to_target.pdf` | full | rounds and compute to reach RMSE <= 0.86 |
| `fig3_ablation.pdf` | full | Delta RMSE / Delta CRPS per removed component |
| `fig4_calibration.pdf` | full | PICP, CRPS, minADE across diversity settings |
| `fig5_pareto.pdf` | single | CRPS vs compute, with Pareto frontier |
| `fig6_sensitivity.pdf` | full | CRPS and PICP vs k, lambda_sup, gamma_nash |
| `fig7_ns3_scalability.pdf` | full | 802.11p latency and delivery vs K, k, N_G |

## What the figures say that the text should not contradict

**`fig_sota_convergence` panel (b).** The proposed method's CRPS *rises* after
round 20 and MD-GAN overtakes it near round 45. The figure shows this plainly.
The accompanying text must acknowledge it: the framework's advantage is reaching
a target quickly and cheaply, not the best distributional score at convergence.

**`fig_ablation_delta`.** Only `w/o context G(z)` is unambiguously large on both
metrics. Every other component sits near zero with error bars crossing it. Do
not describe those as significant contributions -- the figure refutes it.

**`fig_sensitivity_full`, gamma_nash panel.** Flat, with error bars spanning the
whole range: the interference coefficient of the coordination game has no
measurable effect on CRPS or PICP in this setting. That is a null result about a
parameter the paper introduces, and it should be reported as one rather than
omitted.

**`fig_calibration` panels (b) and (c)** use a non-zero baseline, because the
differences are a few percent and a zero baseline hides them. State the axis
range in the caption so it cannot read as exaggeration.

**`fig_cost_to_target`.** Bars annotated "80% of seeds" mean one seed in five
never reached the target; "never reached" means zero of five did. Keep those
annotations -- removing them would overstate the result.
