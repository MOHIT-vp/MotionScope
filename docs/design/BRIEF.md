# MotionScope UI refinement brief

Design brief for the October 2026 interface refinement. No imagery was generated for the interface: every visual on this product is real pipeline output.

## Direction (from the brief)
- Purpose-built CV / vehicle-tracking tool. Not a chatbot, not an AI dashboard.
- Keep: near-black ground, subtle grid, warm amber accent, off-white type, thin borders, compact mono labels, large video.
- One prominent analysis entry point in the nav; the homepage upload area is the primary action for first-time users.

## Journey (landing)
1. **Recognition.** Hero: real processed footage next to the upload area.
2. **Substance.** How it works: seven pipeline stages, with the original method marked.
3. **Honesty.** Calibration: why pixels are not km/h.
4. **Proof.** Demonstrations: three genuine runs.
5. **Commitment.** Back to the upload area (nav CTA), no repeated banner.

## Feeling curve
| Act | Feeling | Cause |
|---|---|---|
| Hero | curiosity | dragging the divider reveals the raw clip under the tracked one |
| How it works | confidence | a precise, ordered list with real equations, not marketing cards |
| Calibration | trust | the product says what it cannot do |
| Demos | conviction | the same pipeline on three very different clips |

**Peak:** the hero wipe. *"You drag a line across the video and the tracking appears."* It gets the largest media and sits above the fold.

**Tell-someone sentence:** It's the site where you drag across a traffic video and watch every car get an ID.

## Signature move
Hero before/after wipe: two synchronised genuine videos (original and processed demo clip) with a draggable, keyboard-accessible divider built on `clip-path`. The workspace's Compare mode uses the same mechanism on the user's own video, so the landing page previews a real product interaction.

## Taste rules applied
- At most one eyebrow per three sections; no section counters except the pipeline order, which is information.
- No identical card grids; demos use an asymmetric mosaic sized to each clip's real aspect ratio.
- No em dashes in visible text; no invented numbers (all stats are read from generated reports).
- Motion: `transform` / `opacity` / `clip-path` only, custom ease-out, UI transitions at most 240 ms, press feedback `scale(0.97)`, hover gated to fine pointers, reduced motion keeps opacity and drops movement.
