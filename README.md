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
  read this before implementing `models/backbone_clip.py`,
  `models/artifact_branch.py`, or `models/fusion_head.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in KAGGLE_USERNAME/KAGGLE_KEY, WANDB_API_KEY, HF_TOKEN
```

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
python train.py --config configs/train.yaml
python evaluate.py --checkpoint <path> --out results/robustness_table.csv
python infer.py --input_dir <image_dir> --out preds.json
```

`infer.py` is the required deliverable script: given an image directory, it
outputs a JSON file `[{"image_path": ..., "pred": <float 0-1>}, ...]` — one
confidence score per image, higher = more likely AI-generated.

## Repo layout

```
data/            dataset download/prep, shared transforms (train-aug + eval), Dataset class
models/          CLIP branch, SRM artifact branch, fusion head
train.py         training loop + checkpoint/resume (Colab-safe)
evaluate.py      robustness matrix across all transforms/severities + ablations
infer.py         required inference script (image dir -> JSON preds)
explainability.py  Grad-CAM/attention rollout (CLIP branch) + SRM residual viz
error_analysis.py FP/FN mining and bucketing by transform/severity
configs/         training config (YAML)
docs/            design spec + branch interface contract
notebooks/       demo notebook for the walkthrough video
results/         robustness table, error-analysis thumbnails
tests/           pytest — transform unit tests, infer.py JSON-contract test
```

## Limitations & future work

**Current status (as of 2026-08-29):** the CLIP branch, SRM branch, and a
placeholder fusion head are implemented and unit-tested, but no end-to-end
training run against real images has completed yet — `train.py` still needs
to be wired to the CLIP embedding cache (`data/clip_embedding_cache.py`) and
the real dataset instead of synthetic tensors. `evaluate.py`/`infer.py`'s
scoring paths are built and tested against dummy models, but not yet run
against a real checkpoint. The sections below will be filled in with actual
numbers once that training run produces one — see the design doc's
error-analysis section for what to look for (FP/FN clustering by transform
severity, chosen operating threshold and its FPR/FNR trade-off).

Known limitations independent of a trained checkpoint:

- **Fusion head architecture is provisional.** The current
  `models/fusion_head.py` (`Linear → ReLU → Linear`) is a placeholder;
  [`docs/specs/2026-08-29-fusion-head-design.md`](docs/specs/2026-08-29-fusion-head-design.md)
  proposes a `BatchNorm1d + Dropout` version to correct for CLIP and SRM
  embeddings living on very different scales, pending Training/Infra Lead
  sign-off.
- **Individual-transform robustness only.** Per the brief, transforms are
  evaluated individually (clean, then each transform/severity in isolation);
  only one combined condition (resize→JPEG, simulating a repost pipeline) is
  included as a bonus differentiator, not an exhaustive combination sweep.
- **Ablation branches (`clip_only`/`artifact_only`) aren't wired to a real
  model yet** — `evaluate.py::load_predict_fns` is still a stub pending the
  composite model class that combines the three branches under one
  checkpoint.

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
