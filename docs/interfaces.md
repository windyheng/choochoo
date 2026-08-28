# Branch interface contract

Fill this in and agree on it as a team **before** the CLIP branch, SRM branch,
and fusion head are coded independently. Changing a shape after both sides
have written code is the single most likely Day-2 integration stall — avoid
it by settling this on Day 1.

## 1. CLIP branch output (owned by CLIP Branch Lead) → Fusion head input

**Settled** (implemented in `models/backbone_clip.py`).

- Function: `models/backbone_clip.py::CLIPBackbone.embed(images: Tensor[B, 3, 224, 224]) -> Tensor[B, 512]`
- `D` = **512** — output of `model.visual(...)` for `ViT-B-16` / `openai` via
  open_clip (the projected image embedding; the same feature Ojha et al. 2023
  "UniversalFakeDetect" probe). ViT-L/14 would be 768 and is not the default.
- Input images: float32 in **[0, 1]**, channel-first, **NOT normalized by the
  caller** — exactly what `data/dataset.py` already emits. CLIP's own mean/std
  (`open_clip` `OPENAI_DATASET_MEAN` / `OPENAI_DATASET_STD`) is applied inside
  `embed()`. Images must already be 224×224.
- Dtype: `embed()` returns **float32** regardless of backbone dtype. (Storing
  the cache as float16 is a decision for the embedding-cache pipeline, not part
  of this contract.)
- L2-normalization: **yes** — embeddings are L2-normalized along `dim=-1`
  before return (`CLIPBackbone(l2_normalize=True)`, the default). Any cached
  embeddings must be produced with the same setting.
- Construction: `CLIPBackbone.from_config(config)` reads `model.clip_model`,
  `model.clip_pretrained`, `model.freeze_clip` from `configs/train.yaml`.

## 2. SRM/artifact branch output (owned by SRM Branch Lead) → Fusion head input

- Function: `models/artifact_branch.py::embed(images: Tensor[B, 3, H, W]) -> Tensor[B, D2]`
- `D2` = `srm_out_channels` (default `32` in `configs/train.yaml`)
- Input expected: raw pixel images; the fixed SRM filter bank is applied inside
  `ArtifactBranch`, independently to each RGB channel
- Dtype: float32 input and float32 embedding
- The branch returns an embedding, not a probability. Its output shape is
  `[B, D2]` for any input height and width of at least 3 pixels.

## 3. Fusion head input (owned by Training/Infra Lead)

- Function: `models/fusion_head.py::forward(clip_embed: Tensor[B, D], srm_embed: Tensor[B, D2]) -> Tensor[B]` (logit or probability — specify which)
- Concatenation order: **`[clip_embed, srm_embed]`** — CLIP first (dims
  `x[:, :512]`), then SRM (`x[:, 512:]`). Matches `train.py`'s current
  `DummyFusionHead`. `explainability.py` and `error_analysis.py` must slice with
  the same convention.

## 4. Shared transform module (owned by Data Lead) → everyone

- `data/transforms.py` must expose the same transform functions/signatures
  used by BOTH training augmentation (`dataset.py`) and evaluation
  (`evaluate.py`). Do not let a second, slightly-different implementation of
  "gaussian blur" or "jpeg compress" exist anywhere else in the repo.
- Severities must match `configs/train.yaml`'s `augmentation` section exactly.

## 5. `infer.py` contract (owned by Eval & Deliverables Lead) — hard requirement

- CLI: `python infer.py --input_dir <dir> --out preds.json`
- Output: JSON array, one object per image:
  `{"image_path": "<path>", "pred": <float in [0, 1]>}`
- `pred` = probability/confidence the image is AI-generated (higher = more
  likely AIGC). This exact schema is graded — do not change field names.
