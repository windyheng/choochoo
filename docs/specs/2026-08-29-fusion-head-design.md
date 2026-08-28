# Design: Fusion head

**Status:** proposed — for Training/Infra Lead (Iniyan) to accept/amend before
the real `train.py` integration.
**Owner of the design:** CLIP Branch Lead. **Owner of `train.py` integration and
the training runs:** Training/Infra Lead.
**Supersedes:** the placeholder note in `models/fusion_head.py` ("swap the
architecture here if the Training/Infra Lead wants something different").
**Date:** 2026-08-29

Related: [`docs/interfaces.md`](../interfaces.md) §§1–3, main design doc
[`2026-08-26-aigc-detection-design.md`](2026-08-26-aigc-detection-design.md) §§3, 5.

---

## 1. What the fusion head is

The only substantial trainable component (alongside the SRM CNN). It takes the
CLIP embedding and the SRM embedding for one image and outputs a single
AIGC-vs-authentic score. CLIP backbone is frozen; the SRM filter bank is fixed;
everything the model *learns* about combining semantic and artifact signal lives
here.

## 2. Input contract (recap — authoritative version is `docs/interfaces.md`)

| Input | Shape | Notes |
|---|---|---|
| `clip_embed` | `[B, 512]` float32 | L2-normalized (unit vectors). From `data/clip_embedding_cache.py` for clean images, or live `CLIPBackbone.embed()` for augmented ones. |
| `srm_embed` | `[B, 32]` float32 | `ArtifactBranch.embed()`, live only (cheap; not cached). ReLU-`AdaptiveAvgPool`-ed → non-negative. Measured scale below. |

**Measured embedding scales** (64 real CIFAKE images, CLIP + untrained SRM):

| | per-component mean \|val\| | row L2-norm |
|---|---|---|
| `clip_embed` | 0.023 | 1.00 (constant — normalized) |
| `srm_embed` (untrained) | 0.016 | 0.16 |

In the concatenated `[B, 544]` vector, **CLIP contributes ~6× the L2 energy of
SRM**. And the SRM half's scale is not fixed — that branch trains from random
init, so its magnitude drifts across training, whereas CLIP's is frozen.

Concat order is fixed: **`[clip_embed, srm_embed]`** → `x[:, :512]` is CLIP,
`x[:, 512:]` is SRM. `explainability.py` and `error_analysis.py` slice on this.

## 3. Architecture

```
clip_embed [B,512] ─┐
                    ├─ concat ─> BatchNorm1d(544) ─> Linear(544, 256) ─> ReLU
srm_embed  [B,32] ──┘                                                     │
                                                                    Dropout(0.2)
                                                                         │
                                                              Linear(256, 1) ─> logit [B]
```

Changes from the current placeholder (`Linear → ReLU → Linear`) and why:

- **`BatchNorm1d` on the concatenated input.** Two reasons (see §2 measurements):
  (a) CLIP contributes ~6× the L2 energy of SRM in the raw concat, so the first
  `Linear` starts biased toward the CLIP half; (b) more importantly, the SRM
  branch trains from random init, so its scale *drifts* during training while
  CLIP's stays fixed — normalizing the input decouples the head from the SRM
  branch's training dynamics. BatchNorm standardizes each component per batch.
  (LayerNorm is a fine alternative — see §9 Q1 — and removes the batch-statistics
  dependency if batch size ends up small; BatchNorm chosen because batch size is
  64 and CLIP's half of the distribution is stationary.)
- **`Dropout(0.2)` before the output layer.** ~90k training images vs. a
  ~140k-param head is not a severe overfit risk, but the head *will* be run in
  three ablation configurations and on a small internal val set; light dropout
  buys robustness cheaply. Tune 0.1–0.3 if val/train gap is small.
- **Keep it 2-layer, `hidden_dim` from `configs/train.yaml` (`fusion_hidden_dim:
  256`).** No evidence a deeper head helps for a linear-probe-style setup
  (cf. Ojha et al. 2023 use a *single* linear layer). Depth is the first thing
  to cut if training is unstable.

Parameter count: ~140k. The `<2B` budget is a non-issue.

## 4. Output, loss, class handling

- `forward()` returns **raw logits `[B]`**. Caller applies sigmoid (or
  `predict_proba`, §5). Do not sigmoid inside `forward` — the loss needs logits.
- Loss: `BCEWithLogitsLoss`.
- **Class weighting.** `data/prepare_datasets.py` reports per-source class
  balance; CIFAKE is 50/50 but SID_Set (real + full-synthetic only) and the
  WildFake subset (celebahq=real, DDIM=fake) may not be. If the pooled train
  split is more than ~55/45, pass `pos_weight` to `BCEWithLogitsLoss`
  (`= n_neg / n_pos`) rather than resampling — simpler and doesn't change epoch
  length. Decision goes in `train.py`; the head doesn't need to know.
- Optimizer/schedule: `AdamW`, `lr 1e-3`, `weight_decay 1e-4` (already in
  `configs/train.yaml`). A cosine or step decay over the 10 epochs is a nice-to-
  have, not required.

## 5. Calibration

The main design doc §3 calls for a **calibrated** confidence score, and
`error_analysis.py` needs a meaningful operating threshold. The graded metric
(AUROC) is ranking-based and unaffected, but the `infer.py` JSON and the
FPR/FNR discussion both want real probabilities.

**Approach: temperature scaling** (single-parameter Platt scaling). After
training:

1. Run the trained head over the **held-out val split** to get logits + labels.
2. Fit one scalar `T > 0` minimizing `BCEWithLogitsLoss(logits / T, labels)`
   (LBFGS, ~50 iters, or a coarse grid).
3. Store `T` as a non-trainable buffer on `FusionHead` (`register_buffer
   ("temperature", torch.ones(1))`).

Add:

```python
def predict_proba(self, clip_embed, srm_embed) -> Tensor:  # [B] in (0,1)
    return torch.sigmoid(self.forward(clip_embed, srm_embed) / self.temperature)
```

`infer.py::predict` and `evaluate.py` call `predict_proba`. Training and the
temperature fit both use `forward` (raw logits). `T` defaults to 1.0, so an
uncalibrated checkpoint still produces sane output.

Threshold: once calibrated, `error_analysis.py` picks the operating point from
the FPR/FNR trade-off (design doc §7) and writes it to `configs/train.yaml`
`eval.threshold` (currently the `0.5` placeholder). Not the head's concern.

## 6. Ablation / branch selection

The design doc (§5, §9) requires **three separately trained models** —
full, CLIP-only, SRM-only — each run on its own Colab account, compared per
transform. This is the real ablation: a jointly-trained head with one branch
zeroed at eval is *not* equivalent (the head co-adapted to both inputs).

So `FusionHead` must build in any of the three shapes:

```python
FusionHead(clip_dim=512, srm_dim=32)   # full
FusionHead(clip_dim=512, srm_dim=0)    # CLIP-only  -> forward ignores srm_embed
FusionHead(clip_dim=0,   srm_dim=32)   # SRM-only   -> forward ignores clip_embed
```

- `__init__`: input width = `clip_dim + srm_dim`; a branch with dim `0` is
  dropped from the concat and its `forward` argument is accepted but ignored
  (may be `None`).
- Exactly one of the two may be `0`; both `0` is a `ValueError`.
- `from_config(config, branch="full"|"clip_only"|"artifact_only")` reads
  `model.clip_model`-implied `512` and `model.srm_out_channels` and zeroes the
  right one.

**Impact on `evaluate.py`:** `load_predict_fns` currently implies scoring a
single branch of *one* checkpoint ("extra hooks into the fusion head"). Under
this design it instead loads **three checkpoints** (`--checkpoint-full`,
`--checkpoint-clip`, `--checkpoint-srm` or a directory) and wraps each. Flag for
the Eval Lead.

## 7. Integration points (all currently stubbed or dummy)

| Site | Change |
|---|---|
| `train.py::build_model` | `return FusionHead.from_config(config, branch=args.branch)` instead of `DummyFusionHead`. Add a `--branch` arg (default `full`). |
| `train.py` dataloader | needs a real `EmbeddingDataset` yielding `(clip_embed, srm_embed, label)` — CLIP from cache/live, SRM live via `ArtifactBranch`. **Does not exist yet** — separate piece of work, Training/Infra seam. For a CLIP-only run, `srm_embed` can be a `[B,0]` tensor. |
| `train.py` checkpoint | save/load `temperature` buffer with the state dict (already covered by `model.state_dict()`). |
| `infer.py::load_model` | assemble `CLIPBackbone` (frozen) + `ArtifactBranch` + `FusionHead`, load checkpoint, `.eval()`. |
| `infer.py::predict` | `clip = backbone.embed(x); srm = artifact.embed(x); return head.predict_proba(clip, srm).item()`. |
| post-training | a `calibrate.py` step (or a `train.py --calibrate` flag) that fits `T` on val and rewrites the checkpoint. |

## 8. Gap list — current `models/fusion_head.py` vs. this design

- [ ] `BatchNorm1d` input normalization
- [ ] `Dropout`
- [ ] `temperature` buffer + `predict_proba`
- [ ] `clip_dim=0` / `srm_dim=0` branch-drop for the ablation
- [ ] `from_config(config, branch=...)`
- [ ] keep the existing `forward` contract (concat order, logits `[B]`, the
      shape/batch-size `ValueError`s) and its tests — those are correct as-is

## 9. Open questions for the Training/Infra Lead

1. BatchNorm vs LayerNorm on the concat input? (Spec says BatchNorm; LayerNorm
   removes the batch-statistics dependency if batch size ends up small.)
2. Is a `train.py --branch {full,clip_only,artifact_only}` flag the right way to
   drive the three ablation runs, or separate configs?
3. Who owns `calibrate.py` / the `--calibrate` step — Training/Infra or Eval?
4. `EmbeddingDataset` (the real dataloader) — Training/Infra seam, but the CLIP
   Branch Lead can draft it since it's mostly cache-plumbing. Want that?
