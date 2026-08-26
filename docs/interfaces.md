# Branch interface contract

Fill this in and agree on it as a team **before** the CLIP branch, SRM branch,
and fusion head are coded independently. Changing a shape after both sides
have written code is the single most likely Day-2 integration stall — avoid
it by settling this on Day 1.

## 1. CLIP branch output (owned by CLIP Branch Lead) → Fusion head input

- Function: `models/backbone_clip.py::embed(images: Tensor[B, 3, H, W]) -> Tensor[B, D]`
- `D` = ? (e.g. 512 for ViT-B/16 via open_clip — confirm against the actual
  checkpoint, don't assume)
- Normalization expected on input images: TODO (CLIP's own preprocessing —
  confirm mean/std used by `open_clip.create_model_and_transforms`)
- Dtype: TODO (float32 vs float16 — matters for Colab memory)
- Whether embeddings are L2-normalized before returning: TODO

## 2. SRM/artifact branch output (owned by SRM Branch Lead) → Fusion head input

- Function: `models/artifact_branch.py::embed(images: Tensor[B, 3, H, W]) -> Tensor[B, D2]`
- `D2` = ? (depends on chosen shallow-CNN head — decide and record)
- Input expected: raw pixel images (SRM filters applied inside this module),
  or pre-filtered residual maps? TODO — pick one, document it
- Dtype: TODO

## 3. Fusion head input (owned by Training/Infra Lead)

- Function: `models/fusion_head.py::forward(clip_embed: Tensor[B, D], srm_embed: Tensor[B, D2]) -> Tensor[B]` (logit or probability — specify which)
- Concatenation order: `[clip_embed, srm_embed]` or the reverse? Pick one and
  keep it consistent with `explainability.py` and `error_analysis.py`.

## 4. Shared transform module (owned by Data Lead) → everyone

- `data/transforms.py` must expose the same transform functions/signatures
  used by BOTH training augmentation (`dataset.py`) and evaluation
  (`evaluate.py`). Do not let a second, slightly-different implementation of
  "gaussian blur" or "jpeg compress" exist anywhere else in the repo.
- Severities must match `configs/train.yaml`'s `augmentation` section exactly.

- `apply_named(image, name, severity)` dispatch names, fixed by `evaluate.py`
  (Eval & Deliverables Lead) — implement `apply_named` to route these exact
  strings to the matching function: `"jpeg"` -> `jpeg_compress`, `"blur"` ->
  `gaussian_blur`, `"resize"` -> `resize_then_upscale`, `"noise"` ->
  `gaussian_noise`, `"color_jitter"` -> `color_jitter`, `"crop"` ->
  `center_crop`. `severity` is the raw value from the corresponding
  `*_SEVERITIES`/`*_SCALES`/`*_PCT`/`*_FRAC` constant (e.g. `("jpeg", 70)`,
  `("crop", 0.80)`).

## 5. `infer.py` contract (owned by Eval & Deliverables Lead) — hard requirement

- CLI: `python infer.py --input_dir <dir> --out preds.json`
- Output: JSON array, one object per image:
  `{"image_path": "<path>", "pred": <float in [0, 1]>}`
- `pred` = probability/confidence the image is AI-generated (higher = more
  likely AIGC). This exact schema is graded — do not change field names.
