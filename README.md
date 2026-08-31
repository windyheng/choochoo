# AIGC Detector — Robust Detection of AI-Generated Images

Prototype that classifies images as AI-generated (AIGC) vs. authentic, robust to
realistic post-processing (JPEG re-compression, blur, resize/thumbnailing,
noise, color jitter, center-crop). Built for track #5 - Robust Detection of AI‑Generated Images Under Real‑World Transformations.

## Overview

- **Architecture**: two-branch fusion — frozen CLIP ViT-B/16 (semantic signal,
  robust to compression/blur) + SRM filter bank → shallow CNN (artifact
  signal, degrades under transforms but strong on clean images). See
  [`docs/specs/2026-08-26-aigc-detection-design.md`](docs/specs/2026-08-26-aigc-detection-design.md)
  for the full design rationale, robustness protocol, and evaluation plan.
- **Team roles**: see the table in the design doc — 5 roles mirroring the two
  branches (Data, CLIP-branch, SRM-branch, Training/Infra, Eval+Deliverables).
- **Branch interface contract**: [`docs/interfaces.md`](docs/interfaces.md) —
  the CLIP branch (`models/backbone_clip.py`), SRM branch
  (`models/artifact_branch.py`), and fusion head (`models/fusion_head.py`,
  design: [`docs/specs/2026-08-29-fusion-head-design.md`](docs/specs/2026-08-29-fusion-head-design.md))
  are all implemented against it now.

## Tech stack

- **Development tools**: VS Code (local development), Google Colab and Kaggle
  Notebooks (free GPU for training — see [`notebooks/`](notebooks/)), Jupyter,
  `pytest` (110 tests, including real non-mocked model integration tests).
- **Models**: CLIP ViT-B/16 (`openai` pretrained weights, via
  [`open_clip`](https://github.com/mlfoundations/open_clip), frozen) for the
  semantic branch; a fixed SRM high-pass filter bank (classical, not learned)
  feeding a shallow CNN for the artifact branch. See
  [`docs/specs/2026-08-26-aigc-detection-design.md`](docs/specs/2026-08-26-aigc-detection-design.md).
- **APIs**: HuggingFace Hub API (CLIP weights, SID_Set dataset), Kaggle API
  (CIFAKE dataset), Weights & Biases API (training run tracking).
- **Libraries/frameworks**: PyTorch, torchvision, `open_clip_torch`,
  OpenCV, scikit-learn, `pytorch-grad-cam`, matplotlib, `huggingface_hub`,
  `kaggle`, `wandb`, PyYAML, pyarrow, `python-dotenv` — see
  [`requirements.txt`](requirements.txt) for exact versions.
- **Datasets & assets**: SID_Set (HuggingFace), CIFAKE (Kaggle), WildFake
  (ModelScope, celebahq + DDIM subset) for training; COCO val2017 + DALL·E
  Advanced for demo/validation only, never used for training or
  hyperparameter selection. See the design doc for full details and links.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN (see below); KAGGLE_USERNAME/KAGGLE_KEY and
                        # WANDB_API_KEY are only needed if you re-run data/prepare_datasets.py
                        # to refresh/extend the dataset — train.py itself doesn't use them.
```

`HF_TOKEN` is optional but recommended — without it, downloading the CLIP
backbone weights (used by both `train.py` and `explainability.py`) works but
is rate-limited as an anonymous request. Get a free token at
[huggingface.co](https://huggingface.co) (account settings → Access Tokens).

Datasets (see the design doc for full details and links):
- SID_Set (HuggingFace) and CIFAKE (Kaggle) — downloaded via `data/prepare_datasets.py`.
- WildFake (ModelScope) — **must be translated via the site's translation button
  manually before download**; not automatable. Already downloaded and published
  to this repo under `data/raw/wildfake/` (celebahq + DDIM subsets), so this
  step is only needed if you're refreshing or extending the dataset.
- COCO val2017 + DALL·E Advanced — demo/validation only, **never used for
  training or hyperparameter selection**.

## Reproducing results

```bash
python data/prepare_datasets.py --config configs/train.yaml
python train.py --config configs/train.yaml   # needs a GPU (Colab or Kaggle notebooks both work — see notebooks/) — CPU-only is impractically slow for 90k+ images
python evaluate.py --checkpoint <path> --eval_csv data/cache/splits/test.csv --out results/robustness_table.csv
python infer.py --input_dir <image_dir> --out preds.json --checkpoint <path>
```

**Note:** no training run has been executed yet, so `<path>` above doesn't
exist yet — `evaluate.py`/`infer.py` are fully implemented and tested against
real (freshly-initialized) checkpoints, but haven't been run against an
actually-trained one. `--checkpoint` on `infer.py` defaults to
`checkpoints/best.pt`; both scripts also accept `--config` (default
`configs/train.yaml`) and, for `evaluate.py`, `--branch full|clip_only|artifact_only`
plus optional `--checkpoint_clip_only`/`--checkpoint_artifact_only` for the
ablation comparison (each branch is trained separately via `train.py --branch`).

`infer.py` is the required deliverable script: given an image directory, it
outputs a JSON file `[{"image_path": ..., "pred": <float 0-1>}, ...]` — one
confidence score per image, higher = more likely AI-generated.

### Try it now — no training required

The two explainability visualizations don't need a trained checkpoint (CLIP
is pretrained/frozen, the SRM filter bank is fixed) and work today:

```bash
python -c "
from explainability import clip_attention_overlay, srm_residual_overlay
clip_attention_overlay('path/to/image.jpg', save_path='clip_overlay.png')
srm_residual_overlay('path/to/image.jpg').save('srm_overlay.png')
"
```

`clip_overlay.png` shows CLIP's attention-rollout heatmap (which regions the
semantic branch focuses on); `srm_overlay.png` shows the SRM branch's raw
artifact-residual map.

A third, `explainability.grad_cam_overlay(image_path, checkpoint_path)`, is
prediction-specific — which regions pushed a *trained* model toward "AI-
generated" — so it needs a `train.py` checkpoint. Run it on the FP/FN images
`error_analysis.py` surfaces, for the error-analysis note and demo video.

## Repo layout

```
data/            dataset download/prep, shared transforms (train-aug + eval), Dataset
                  class, path-portability helpers (data/paths.py), CLIP embedding cache
models/          CLIP branch, SRM artifact branch, fusion head — all implemented
train.py         training loop + checkpoint/resume (Colab/Kaggle-safe), wired to real data
evaluate.py      robustness matrix across all transforms/severities + ablations
infer.py         required inference script (image dir -> JSON preds)
explainability.py  CLIP attention-rollout + Grad-CAM overlays, SRM residual viz
error_analysis.py FP/FN mining and bucketing by transform/severity
configs/         training config (YAML)
docs/            design specs (main design, fusion-head design) + branch interface contract
notebooks/       train_colab.ipynb / train_kaggle.ipynb (GPU training); no demo/walkthrough notebook yet
results/         robustness table, error-analysis thumbnails
tests/           pytest — 110 tests, including real (non-mocked) model integration tests
```

## Robustness Evaluation Summary

**Current status (as of 2026-08-31):** first real results, from the checkpoint
trained to **step 4,200 of the planned 14,150** (10 epochs — training was
stopped early by running out of free Kaggle/Colab GPU quota, not by
convergence; expect these numbers to improve with more training). Evaluated
against the full 19,410-image held-out test set, but only **10 of the
required 16 conditions** completed before the same GPU-quota constraint
forced a stop: `clean` + all 4 JPEG severities + all 3 blur severities + both
resize severities. **Not yet evaluated**: noise (3 severities), color jitter,
center crop, and the combined resize→JPEG condition — see
[`results/robustness_table.csv`](results/robustness_table.csv) for the raw
data once those land.

| Condition | AUROC | Accuracy | FPR | FNR |
|---|---|---|---|---|
| clean | 0.996 | 0.969 | 0.018 | 0.045 |
| jpeg_quality_90 | 0.996 | 0.965 | 0.016 | 0.055 |
| jpeg_quality_70 | 0.995 | 0.963 | 0.021 | 0.053 |
| jpeg_quality_50 | 0.989 | 0.940 | 0.025 | 0.096 |
| jpeg_quality_30 | 0.983 | 0.914 | 0.025 | 0.149 |
| blur_sigma_0.5 | 0.993 | 0.893 | 0.004 | 0.213 |
| blur_sigma_1.0 | 0.907 | 0.656 | 0.006 | 0.689 |
| blur_sigma_2.0 | 0.758 | 0.643 | 0.076 | 0.644 |
| resize_scale_0.5 | 0.859 | 0.657 | 0.012 | 0.683 |
| resize_scale_0.25 | 0.750 | 0.650 | 0.115 | 0.590 |

**Combined AUC (clean + mean of the 9 evaluated transformed conditions): 0.955**
— caveat: this excludes noise/color-jitter/crop entirely, so it reads more
optimistic than the true 16-condition score would once those are run.

**Operating threshold: kept at the default 0.5**, not tuned down. FPR is low
across every evaluated condition (worst case 11.5% at the most extreme
transform tested, typically under 2.5%) — appropriate for a moderation
use-case where wrongly flagging a real user's photo carries a real
trust/reputation cost. The weak spot is FNR under blur/resize (see the error
analysis note below), which looks like a training-progress gap rather than a
calibration problem, so more training is the better fix, not threshold
tuning — revisit once a more-trained checkpoint's FPR/FNR profile exists.

**Error analysis**: see [`docs/error_analysis_note.md`](docs/error_analysis_note.md)
for the full write-up (representative FP/FN patterns, trade-off discussion).

See [Limitations & what we'd improve](#limitations--what-wed-improve) below
for the full reflection.

## Limitations & what we'd improve

Known limitations, independent of training progress:

- **Training is incomplete.** The current checkpoint stopped at step 4,200 of
  the planned 14,150 (free Kaggle/Colab GPU quota ran out, not convergence).
- **Only 10 of the 16 required robustness conditions have been evaluated**
  (missing: noise ×3 severities, color jitter, center crop, the combined
  resize→JPEG condition) — same GPU-quota constraint.
- **Individual-transform robustness only.** Per the brief, transforms are
  evaluated individually (clean, then each transform/severity in isolation);
  only one combined condition (resize→JPEG, simulating a repost pipeline) is
  included as a bonus differentiator, not an exhaustive combination sweep —
  and that combined condition wasn't reached in this run either (see above).
- **Ablation branches (`clip_only`/`artifact_only`) are separately-trained
  models**, not one model with post-hoc ablation — `train.py --branch
  clip_only|artifact_only` must each be run and checkpointed independently
  before `evaluate.py`'s ablation comparison can run. Not done yet.
- **Class balance across dataset sources isn't verified yet.** CIFAKE is
  50/50, but SID_Set/WildFake's true balance hasn't been checked against
  `compute_pos_weight`'s output on a real run.

**With more time**, in priority order: (1) finish training to the full 14,150
steps — the blur/resize FNR gap looks like it would improve most from this;
(2) run the remaining 6 conditions (noise, color jitter, crop, combined) for
a true 16-condition score; (3) train the two ablation branches for the
CLIP-vs-SRM contribution comparison the design was built to answer.

## Demo video

_TBD — short end-to-end walkthrough (inference on a sample image directory,
the explainability overlays, and the robustness results above), uploaded to
YouTube as public and linked here and in the Devpost submission._

## Team & contributions

| Role | Owner | Scope |
|---|---|---|
| Data Lead | Darren | `data/prepare_datasets.py`, `data/transforms.py`, `data/dataset.py` |
| CLIP Branch Lead | Windy | `models/backbone_clip.py`, fusion head design, CLIP explainability |
| SRM Branch Lead | Elodie | `models/artifact_branch.py`, SRM explainability |
| Training/Infra Lead | Iniyan | `train.py`, `configs/`, checkpoint/resume, ablations |
| Eval & Deliverables Lead | Jerry | `evaluate.py`, `infer.py`, `error_analysis.py`, README/Devpost/demo video |

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch naming and PR flow.
