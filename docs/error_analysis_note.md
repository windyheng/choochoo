# Error Analysis Note

**Checkpoint:** `ckpt_step_00004200.pt` (step 4,200 of the planned 14,150 —
training stopped early by running out of free GPU quota, not by
convergence). Trained on Kaggle notebooks (see
[`notebooks/train_kaggle.ipynb`](../notebooks/train_kaggle.ipynb)).
**Evaluated against:** full 19,410-image held-out test set, 10 of 16
conditions (`clean`, JPEG ×4, blur ×3, resize ×2 — see
[`results/robustness_table.csv`](../results/robustness_table.csv)).
**Source data:** [`results/error_analysis/error_summary.csv`](../results/error_analysis/error_summary.csv)
(bucket counts) and the thumbnail grids alongside it (representative
FP/FN examples per condition).

## The dominant failure mode: false negatives under blur and resize

Across every evaluated condition, false negatives (an AI-generated image
scored as real) massively outnumber false positives (a real image scored as
AI-generated) — and that gap widens sharply under blur and resize:

| Condition | FP | FN | FPR | FNR |
|---|---|---|---|---|
| clean | 177 | 429 | 1.8% | 4.5% |
| blur_sigma_0.5 | 38 | 2,046 | 0.4% | 21.3% |
| blur_sigma_1.0 | 59 | 6,611 | 0.6% | 68.9% |
| blur_sigma_2.0 | 742 | 6,180 | 7.6% | 64.4% |
| resize_scale_0.5 | 116 | 6,550 | 1.2% | 68.3% |
| resize_scale_0.25 | 1,129 | 5,655 | 11.5% | 59.0% |
| jpeg_quality_30 | 250 | 1,425 | 2.5% | 14.9% |

At `blur_sigma_1.0`, the model misses **68.9%** of AI-generated images while
wrongly flagging only **0.6%** of real ones. JPEG degrades far more
gracefully — even at the worst tested quality (30), FNR only reaches 14.9%.

## Why this happens (and why it's expected at this training stage)

This matches the two-branch design's own failure hypothesis, but sharper
than expected. Heavy blur and downsampling are low-pass operations — they
destroy exactly the high-frequency pixel statistics the SRM artifact branch
reads. At this early checkpoint (29.7% through the planned training run),
the fusion head still appears to lean on that fragile artifact signal more
than the design intends; when blur/resize wipes it out, predictions get
pulled toward "real" rather than "AI-generated," rather than the CLIP
branch's more robust semantic signal picking up the slack.

This reads as a **training-progress gap, not an architectural one** — the
whole point of the frozen-CLIP + SRM fusion design is that the CLIP branch
should compensate exactly in this scenario. More training steps are the
most likely fix, more than any architecture or threshold change. This is
the single highest-priority thing to verify once training resumes.

## The one condition with errors on both sides

`resize_scale_0.25` (the most severe transform tested — downscale to a
quarter, then upscale back) is the only condition with *elevated FPR too*
(11.5%, vs. typically under 2.5% elsewhere), on top of a still-high FNR
(59.0%). Consistent with it having the lowest AUROC of any tested condition
(0.750) — this is simply the hardest condition in the current battery, on
both sides of the classification.

## False-positive cost trade-off

For a content-moderation use case, a false positive (flagging a real user's
photo as AI-generated) carries a real trust/reputation cost distinct from a
false negative's cost — wrongly accusing someone of posting fake content is
a worse user-facing failure than missing a fake. The current model already
reflects that asymmetry reasonably well: FPR stays low (under 2.5% in 7 of
10 evaluated conditions — the exceptions are `jpeg_quality_30` at 2.5%,
`blur_sigma_2.0` at 7.6%, and `resize_scale_0.25` at 11.5%) even where FNR
is high. That's consistent with
keeping the operating threshold at the default 0.5 rather than lowering it
to chase FNR — lowering the threshold would directly trade away the
low-FPR property that already matters most for this use case. See the
README's Robustness Evaluation Summary section for the full threshold
discussion.

## What to look at in the thumbnail grids

`results/error_analysis/{condition}_{fp,fn}.png` holds the top-8
most-confidently-wrong examples per bucket (highest-scoring FPs, lowest-
scoring FNs — see `error_analysis.py::select_representative`). Worth
specifically pulling into the demo video:
- `blur_sigma_1.0_fn.png` / `resize_scale_0.5_fn.png` — the two conditions
  with the most dramatic FNR spike, good visual evidence of the
  artifact-branch signal loss discussed above.
- `resize_scale_0.25_fp.png` — the one bucket with real elevated false
  positives, worth showing what specifically fools the model at that
  severity.

## Follow-up once training resumes

1. Re-run this analysis against a more-trained checkpoint (ideally the full
   14,150 steps) — expect the blur/resize FNR gap to close substantially if
   the training-progress hypothesis above is correct.
2. Run the remaining 6 conditions (noise ×3, color jitter, center crop,
   combined resize→JPEG) for the full 16-condition picture.
3. Train the `clip_only`/`artifact_only` ablation branches to directly test
   the hypothesis above — if it's right, `artifact_only` should show the
   same blur/resize FNR collapse even more severely, and `clip_only` should
   be comparatively robust to it.
