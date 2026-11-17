# Gemini prompt — methodology figure (IEEE IoT-J)

Two variants. **Variant A** is the one to submit (clean IEEE vector style).
**Variant B** is the iconized network version for a wider audience / talk.
Generate A first, then ask for B as a restyle so the topology stays identical.

---

## VARIANT A — IEEE journal methodology diagram

Create a professional system-architecture diagram for an IEEE Internet of Things
Journal paper, titled "Objective-Adaptive Federated GAN Training for
Resource-Constrained Internet of Vehicles".

STYLE — this is a camera-ready IEEE journal figure, not an infographic:
- Flat 2D vector line art. No 3D, no drop shadows, no gradients, no glow, no
  gloss, no photographic or rendered elements, no clip-art.
- Pure white background.
- Strictly greyscale plus ONE accent colour (deep blue) used only for the
  adaptive control path. The figure must stay fully legible printed in black
  and white, so encode meaning by line style and fill shade, never by hue alone.
- Thin uniform strokes (~1 pt). Boxes are rectangles with square or very
  slightly rounded corners, 1 pt borders, white or 10% grey fills.
- All text horizontal, sans-serif (Helvetica/Arial), 7–9 pt equivalent, crisp
  and correctly spelled. No text rotated vertically. No handwriting font.
- Single-column proportions: about 3.5 inches wide by 4 inches tall, i.e. a
  portrait/square aspect ratio roughly 1:1.1. Dense but not crowded; leave
  clean white gutters between the three tiers.
- No title text inside the image (the caption goes in LaTeX). No legend box
  unless it is needed for line styles — if included, keep it a single thin
  row along the bottom.

LAYOUT — three stacked horizontal tiers, plus a control block on the left.

TIER 1 (top) — "Deployment objective":
- A small box on the upper left containing the weight vector
  "alpha = (alpha_G, alpha_1, alpha_2, alpha_3, alpha_4)" with the sub-label
  "accuracy, CPU, memory, latency, traffic". Mark it clearly as an INPUT, e.g.
  a thicker border or an inward arrow from outside the figure boundary.
- To its right, a box labelled "Objective-adaptive controller" containing two
  stacked sub-rows: "Static policy: v* = Pi(v_0 - 1/2 (v_max - v_min)
  <alpha, d_v>)" and "Online variant: dual ascent on realised terms".
- From the controller, a deep-blue arrow labelled
  "configuration (k, E_G, S, lambda_sup)" descends into TIER 2. This blue path
  is the paper's contribution — make it the visually dominant arrow, dashed or
  heavier than the data arrows.

TIER 2 (middle) — "Edge tier: generators at roadside units":
- Four identical boxes in a row, labelled "RSU 1 / G_1" through "RSU 4 / G_4".
- Inside one of them (RSU 1), expanded as a small inset, show the conditional
  generator: context window "c = (x_{t-7} ... x_t), W = 8" and noise "z, dim 16"
  both feeding a block "f_theta", whose output goes to a block labelled
  "bounded residual head: G(c,z) = c_W + delta_max tanh(f_theta(c,z))".
  Label this inset "conditional generator (one shown)".
- A thin horizontal double-headed arrow across the top of the four RSU boxes
  labelled "periodic synchronisation, every S rounds".

TIER 3 (bottom) — "Vehicle tier: critics over V2I":
- Under each RSU, a small cluster of 5 identical small boxes labelled "D_1 ...
  D_5", i.e. "5 vehicle critics per generator", 20 total. Draw all four clusters
  but you may draw the inner boxes of three clusters compactly.
- In the FIRST cluster, shade/fill exactly 3 of the 5 boxes solid dark grey and
  leave 2 white with a dashed border. Add the label "top-k active subset, k = 3"
  pointing at the shaded ones and "idle" at the dashed ones. This selection is
  what the controller changes — make it obvious.
- Each vehicle box carries a tiny sub-label "disjoint NGSIM road-segment shard".
- Between TIER 2 and TIER 3, two distinct arrow styles, both labelled:
  a solid downward arrow "G(c,z) synthetic pairs (c, y_hat)" and a different
  upward arrow (dotted) "critic gradients, Eq. (1) with gradient penalty".

LEFT SIDE — a vertical block spanning tiers 2 and 3, labelled
"Per-round measured cost terms", listing on separate lines: "training loss",
"CPU", "memory", "latency", "traffic (ns-3 802.11p)". A deep-blue feedback arrow
runs from the BOTTOM of this block, up the left margin, back into the
"Objective-adaptive controller" in TIER 1, labelled "realised terms (online
variant only)". Draw this as a clean right-angled routed path hugging the left
margin — it must not cross or overlap any other arrow or any box.

ALSO SHOW, as a small annotation near the vehicle tier: "vehicles leaving
coverage drop their update (mobility dropout)" with a thin arrow pointing to one
vehicle box drawn with a dashed/faded border.

CRITICAL CONSTRAINTS:
- Render every label exactly as written. Mathematical subscripts should read
  correctly (alpha_G, E_G, lambda_sup, delta_max, c_W).
- Do not invent extra components, servers, clouds, databases, satellites or
  base stations that are not listed above. There is no central aggregator in the
  adaptive method — do not draw one.
- Keep exactly 4 generators and 5 critics per generator.
- Arrows must not cross each other or pass through boxes. Route orthogonally.
- No watermark, no signature, no frame around the whole figure.

---

## VARIANT B — iconized network version

Restyle the diagram above for a figure that communicates the deployment at a
glance. Keep the EXACT same topology, counts, labels and three-tier structure.
Change only the visual vocabulary:

- Replace the RSU boxes with simple flat line-art roadside-unit icons: a pole
  with a small antenna and concentric broadcast arcs.
- Replace the vehicle critic boxes with simple flat top-view or side-view car
  icons, all identical, drawn in the same line weight.
- Show the 3 active critics as filled/solid cars connected to their RSU by solid
  V2I links with broadcast arcs; show the 2 idle ones as outline-only, greyed,
  with dashed links.
- Draw a thin two-lane road strip behind each vehicle cluster to convey the road
  segment shards, with the four strips visibly distinct (different shading) to
  show the partitions are disjoint.
- Keep the controller as a clean rectangular block — it is logic, not hardware —
  and keep its deep-blue configuration arrow and feedback path exactly as in
  Variant A.
- Still flat 2D vector, white background, greyscale plus the one deep-blue
  accent, all text horizontal sans-serif. Icons must be minimal and uniform:
  consistent stroke width, no colour fills beyond grey shades, no cartoon
  styling, no perspective, no shadows.
- Slightly wider is acceptable here: about 7 inches by 4 inches, a double-column
  landscape figure.

Keep every text label from Variant A. The two figures must be recognisably the
same system.
