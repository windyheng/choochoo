# AIGC Detector — Robust Detection of AI-Generated Images

Prototype that classifies images as AI-generated (AIGC) vs. authentic, robust to
realistic post-processing (JPEG re-compression, blur, resize/thumbnailing,
noise, color jitter, center-crop). Built for [hackathon name/track — fill in].

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
python train.py --config configs/train.yaml   # needs a GPU (e.g. Colab) — CPU-only is impractically slow for 90k+ images
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

## Repo layout

```
data/            dataset download/prep, shared transforms (train-aug + eval), Dataset
                  class, path-portability helpers (data/paths.py), CLIP embedding cache
models/          CLIP branch, SRM artifact branch, fusion head — all implemented
train.py         training loop + checkpoint/resume (Colab-safe), wired to real data
evaluate.py      robustness matrix across all transforms/severities + ablations
infer.py         required inference script (image dir -> JSON preds)
explainability.py  attention-rollout overlay (CLIP branch) + SRM residual viz — both work today
error_analysis.py FP/FN mining and bucketing by transform/severity
configs/         training config (YAML)
docs/            design specs (main design, fusion-head design) + branch interface contract
notebooks/       demo notebook for the walkthrough video (not yet built)
results/         robustness table, error-analysis thumbnails
tests/           pytest — 110 tests, including real (non-mocked) model integration tests
```

## Limitations & future work

**Current status (as of 2026-08-30):** the full pipeline is implemented and
tested end-to-end — CLIP branch, SRM branch, fusion head (per
[`docs/specs/2026-08-29-fusion-head-design.md`](docs/specs/2026-08-29-fusion-head-design.md):
`BatchNorm1d` + `Dropout`, with a `temperature` buffer for post-training Platt
scaling), `train.py` wired to the real dataset and CLIP embedding cache, and
`infer.py`/`evaluate.py` wired to load a real `train.py` checkpoint and run
actual inference. **What's still missing is the training run itself** — no
checkpoint has been produced yet (CPU-only locally is impractical for
90k+ training images; needs to run on a GPU, e.g. Colab). The sections below
will be filled in with actual numbers once that run produces one — see the design doc's error-analysis section for what to look for
(FP/FN clustering by transform severity, chosen operating threshold and its
FPR/FNR trade-off).

Known limitations independent of a trained checkpoint:

- **Individual-transform robustness only.** Per the brief, transforms are
  evaluated individually (clean, then each transform/severity in isolation);
  only one combined condition (resize→JPEG, simulating a repost pipeline) is
  included as a bonus differentiator, not an exhaustive combination sweep.
- **Ablation branches (`clip_only`/`artifact_only`) are separately-trained
  models**, not one model with post-hoc ablation — `train.py --branch
  clip_only|artifact_only` must each be run and checkpointed independently
  before `evaluate.py`'s ablation comparison can run.
- **Class balance across dataset sources isn't verified yet.** CIFAKE is
  50/50, but SID_Set/WildFake's true balance hasn't been checked against
  `compute_pos_weight`'s output on a real run.

_To fill in once a real checkpoint exists: representative FP/FN examples per
transform/severity bucket, the chosen operating threshold and why (FPR/FNR
trade-off for a moderation use-case), and what the two-branch ablation
actually shows about CLIP vs. SRM contribution._

## Team & contributions

| Role | Owner | Scope |
|---|---|---|
| Data Lead | Darren | `data/prepare_datasets.py`, `data/transforms.py`, `data/dataset.py` |
| CLIP Branch Lead | Windy | `models/backbone_clip.py`, fusion head design, CLIP explainability |
| SRM Branch Lead | Elodie | `models/artifact_branch.py`, SRM explainability |
| Training/Infra Lead | Iniyan | `train.py`, `configs/`, checkpoint/resume, ablations |
| Eval & Deliverables Lead | Jerry | `evaluate.py`, `infer.py`, `error_analysis.py`, README/Devpost/demo video |

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch naming and PR flow.
