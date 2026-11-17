# Gemini — REVISION prompt (paste with the previous image attached)

Keep this image's overall layout, composition and visual style exactly as it is.
I like the three-tier structure, the tier bands, the RSU towers, the road
segments and the car icons. Do NOT redesign it. Change ONLY the text labels and
the specific items listed below, which are currently wrong.

## 1. Controller box — both lines are currently meaningless. Replace exactly.

Delete the current two lines ("α = c in F_θG(α1,α2)α4" and the "y = (R81 − ...)"
line). They are not real equations. The box must read exactly:

    Objective-adaptive controller
    Static policy:    v* = Π(v₀ − ½(v_max − v_min)·⟨α, d_v⟩)
    Online variant:   dual ascent on realised cost terms

Render the Π as a capital Greek Pi, the ⟨ ⟩ as angle brackets, and keep
"v*", "v₀", "v_max", "v_min", "d_v" with correct subscripts. The online line is
PLAIN TEXT — it must contain no formula at all.

## 2. Alpha box sub-label — a term is missing.

Currently: "cost, terms, CPU, memory, traffic, ...". Replace with exactly:

    accuracy, CPU, memory, latency, traffic

Five terms, in that order, matching α_G, α₁, α₂, α₃, α₄. The word "latency"
must be present.

## 3. Bounded residual head — currently shows a fake exponential.

Replace "(e^{α,jt+1})" with exactly:

    G(c,z) = c_W + δ_max · tanh( f_θ(c,z) )

## 4. Generator inset — the arrows are backwards.

It currently shows "c → f_θ → z", which wrongly makes the noise z an output.
Both c AND z are INPUTS. Redraw as:

    c ─┐
       ├─→ [ f_θ ] ─→ ŷ
    z ─┘

So two arrows enter f_θ from the left (one labelled c, one labelled z) and one
arrow leaves to the right labelled ŷ (y-hat). Label c as "context window, W = 8"
and z as "noise, dim 16".

## 5. Add the missing critic-gradient arrow.

Between RSU 1 and its active cars there is currently only a downward path. Add a
SECOND arrow pointing UPWARD from the active cars back to RSU 1, drawn DOTTED to
distinguish it, labelled:

    critic gradients, Eq. (1) + gradient penalty

Label the existing downward arrow exactly "synthetic pairs (c, ŷ)". Without the
upward arrow the diagram shows no adversarial feedback, which is wrong.

## 6. Fix duplicated and truncated text.

- "synthetic synthetic pairs" → "synthetic pairs" (remove the duplicate word).
- "V2I broa" and "V2I brod" are cut off → both should read "V2I link".
- "periodic synchronization" → "periodic synchronisation" (British spelling,
  with an s, to match the manuscript).
- "road segment label" → "road segment shard".

## 7. Car counts — first cluster has too many.

The first (leftmost) cluster must contain EXACTLY 5 cars: 3 active and 2 idle.
It currently shows 6. Remove one so the count is 3 + 2 = 5. Every other cluster
must also show exactly 5 cars, 3 active and 2 idle.

## 8. Colour — remove the red/blue/green cars.

The three active cars are currently red, blue and green, which implies they
differ from each other; they do not. Make all three active cars the SAME single
solid mid-grey. Idle cars stay outline-only light grey with dashed borders. Keep
the deep blue/teal ONLY for the controller's configuration arrow and the
feedback path. The figure must remain readable when printed in black and white.

## Constraints
- Change nothing else. Same layout, same tier bands, same icons, same positions.
- Spell every label exactly as written above, with correct subscripts and
  superscripts.
- Do not add any component that is not already present. There is no central
  server or cloud in this architecture.
- No watermark, no title text inside the image.
