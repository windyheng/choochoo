# Design: Robust Detection of AI-Generated Images Under Real-World Transformations

## 1. Problem

Classify images as AI-generated (AIGC) vs. authentic, robust to realistic
post-processing/redistribution — not just clean-data accuracy.

**Required robustness transforms** (test against a subset; train-time
augmentation should match):

| Transform | Parameters | Real-world analog |
|---|---|---|
| JPEG compression | quality = 90, 70, 50, 30 | social re-encode / messaging |
| Gaussian blur | σ = 0.5, 1.0, 2.0 | out-of-focus |
| Resize | 0.5×/0.25× then upscale | thumbnail generation |
| Gaussian noise | σ = 0.02, 0.05, 0.10 | low-light sensor noise |
| Color jitter | brightness/contrast/saturation ±20% | filter apps / auto-enhance |
| Center crop | crop 80% | profile-picture cropping |

**Hard constraint:** total model parameters **<2B**.

**Datasets (training):**
- SID_Set — https://huggingface.co/datasets/saberzl/SID_Set
- CIFAKE — https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- WildFake — https://modelscope.cn/datasets/hy2628982280/WildFake/summary
  (must be translated via the site's translation button before use)

**Validation/demo dataset — reference only, DO NOT TRAIN ON THIS:**
- Non-AIGC: COCO val2017 (4998 images) · AIGC: DALL·E Advanced (8843 images)
- Demonstrates iterative improvement only; does not contribute to the final score.

**Required deliverables:**
1. Written project description (Devpost): approach, tools, models/APIs, libraries, datasets.
2. Public GitHub repo: well-structured/commented code, a script that takes an
   image directory and outputs `preds.json` (`image_path`, `pred` per image),
   README (overview, setup, repro steps, limitations, contributions).
3. Demo video: end-to-end, public YouTube link in Devpost, no unlicensed
   third-party trademarked/copyrighted content.
4. Robustness Evaluation Summary: clean vs. transformed performance table/visual.
5. Error Analysis Note: representative FP/FN, trade-off discussion.

**Judging weights:** Technical Execution 35% · Innovation & Problem Insight 20%
· Impact & Relevance 20% · Feasibility & Practicality 15% · Presentation & Communication 10%.

## 2. Core insight

Two detector families trade off:
- **Low-level artifact signals** (frequency/noise residuals, GAN checkerboard
  patterns) generalize well within a generator family but are exactly what
  blur/JPEG/resize destroy.
- **High-level semantic signals** (CLIP embeddings) survive compression/blur
  far better and generalize across generator families (important given
  WildFake spans many).

**Decision:** build both, fused. Robustness is treated primarily as a
training-distribution problem (augment with the exact transform families
above), not an architecture trick alone.

## 3. Architecture (<2B params — this design is ~150-450M)

```
Image ──┬──> CLIP ViT-B/16 (frozen, ~150M) ──> [CLS embed] ──┐
         │                                                     ├──> concat ──> small MLP head ──> P(AIGC)
         └──> SRM high-pass filter bank (fixed) ──> shallow CNN (~5M) ─┘
```

- CLIP backbone **frozen** — only the small head + artifact-branch CNN are
  trained (cf. Ojha et al. 2023, "Towards Universal Fake Image Detectors,"
  for cross-generator generalization via a frozen CLIP + linear probe).
  Precompute and cache CLIP embeddings offline since the backbone never updates
  (`data/cache_clip_embeddings.py` -> `data/cache/clip_embeddings/<split>.npz`,
  float16, L2-normalized). Training augmentation changes ~50% of samples' pixels
  each epoch, so those embeddings are computed live against the frozen backbone
  during training; clean samples and all of val/test/eval/ablation are served
  from the cache.
- SRM (Spatial Rich Model) filter bank → shallow CNN is the artifact branch;
  cheap, strong on clean/lightly-transformed images; its degradation under
  blur/JPEG is the intended centerpiece of the error-analysis writeup.
- Output: sigmoid confidence, **calibrated** (Platt/temperature scaling on a
  held-out split) — the deliverable wants a real confidence score.
- Default to ViT-B/16; ViT-L/14 (~428M) is a fallback upgrade only if
  Colab VRAM/time headroom allows.

## 4. Data strategy

- Train on SID_Set + CIFAKE + WildFake (translated) — mix GAN and diffusion
  generator families deliberately so the model doesn't overfit one artifact
  signature.
- Internal held-out val/test split from these sources (never the COCO/DALL·E
  demo set).
- Check class balance per source dataset; weighted sampling/loss weighting as needed.
- Hash-based near-duplicate check across train/val splits.
- Cap WildFake subset size deliberately — generator-family diversity matters
  more than volume on free-tier compute.

## 5. Robustness protocol

- **Training-time:** each batch randomly applies one of the six transform
  families (or none) at a random severity from the brief's parameter list —
  augmentation-matched-to-eval, not leakage (using documented transform
  *types and severities*, not the actual test images).
- **Eval-time matrix** (becomes the Robustness Evaluation Summary deliverable):

  | | Clean | JPEG q90/70/50/30 | Blur σ0.5/1/2 | Resize 0.5×/0.25× | Noise σ0.02/0.05/0.10 | Color jitter | Crop 80% | Combined (resize→JPEG) |
  |---|---|---|---|---|---|---|---|---|
  | AUROC | | | | | | | | |
  | Accuracy | | | | | | | | |
  | FPR / FNR | | | | | | | | |

- Include at least one **combined transform** (e.g. resize→JPEG, simulating
  an actual repost pipeline).
- **Ablation:** full model vs. CLIP-branch-only vs. artifact-branch-only, per
  transform — evidence for the two-branch design decision and the required
  trade-off discussion.

## 6. Explainability

- CLIP branch: attention rollout / Grad-CAM over ViT patches.
- Artifact branch: visualize the SRM residual map directly.
- Overlay both in the demo video.

## 7. Error analysis

- Bucket errors by transform × severity; pull representative FP/FN thumbnails per bucket.
- Hypothesize, then confirm/refute with real examples: FPs likely cluster on
  grainy/low-light real photos (sensor noise mimics generative artifacts);
  FNs likely cluster on heavily-compressed diffusion images at JPEG q30 (both
  branches' signal degrades).
- Explicitly discuss the **false-positive cost trade-off** for a moderation
  use case — state and justify an operating threshold (e.g. optimize for
  FPR ≤ X% rather than raw accuracy).

## 8. Team split (5 people)

| Role | Owns | Notes |
|---|---|---|
| Data Lead | `data/prepare_datasets.py`, `data/transforms.py`, `data/dataset.py`, splits, dedup, class balance | `transforms.py` blocks everyone — finish day 1 |
| CLIP Branch Lead | `models/backbone_clip.py`, embedding cache pipeline, fusion head design | Later: CLIP-side explainability |
| SRM Branch Lead | `models/artifact_branch.py` | Not blocked on Data/CLIP — can prototype day 1. Later: SRM explainability |
| Training/Infra Lead | `train.py`, `configs/train.yaml`, checkpoint/resume, runs the real training jobs, ablations | Owns GPU-hours |
| Eval & Deliverables Lead | `evaluate.py`, `infer.py`, `error_analysis.py`, drives README/Devpost/demo video | Stub `infer.py` against a dummy model day 1 |

## 9. Timeline (5-person parallel, ~1 week on free-tier Colab)

Two hard sync points: (A) shared `transforms.py`/data ready, (B) branches
integrated into the fusion head + baseline pipeline validated end-to-end.
Everything else — including training, if using separate Colab accounts per
person for the full model + each ablation — runs concurrently.

| Day | Data | CLIP branch | SRM branch | Training/Infra | Eval/Deliverables |
|---|---|---|---|---|---|
| 1 | Datasets, translate WildFake, start `transforms.py` | CLIP wrapper + cache script vs. placeholders | SRM filters + shallow CNN vs. placeholders | `train.py` skeleton + checkpoint/resume vs. dummy tensors | `infer.py` contract test + `evaluate.py` metrics vs. dummy model |
| **sync A** | `transforms.py` + splits done | | | | |
| 2 | Finalize splits/dedup/balance | Cache real CLIP embeddings | Integrate SRM on real images | Assemble fusion head, run tiny no-aug baseline | Wire `infer.py` to baseline checkpoint |
| **sync B** | Baseline pipeline runs end-to-end | | | | |
| 3-4 | Support Eval | Run CLIP-only ablation, own Colab account | Run artifact-only ablation, own Colab account | Run full fused model, own Colab account | Finalize eval harness vs. baseline checkpoint |
| 5 | Draft README/dataset section | CLIP explainability | SRM explainability | Draft README/methodology | Real robustness matrix + ablation comparison |
| 6 | Contribute to writeup | Contribute to writeup | Contribute to writeup | Contribute to writeup | Error analysis, storyboard demo video |
| 7 | — | — | — | — | Record/edit demo video, final polish, submit |

**Biggest known risk:** free-tier Colab session/idle timeouts (~12hr cap,
random disconnects). Build checkpoint/resume into `train.py` on Day 1.

## 10. Open decisions (not yet made)

- Exact WildFake subset size/generator-family mix.
- Exact operating threshold for the FPR/FNR trade-off.
- ViT-B/16 vs ViT-L/14 (default B/16, upgrade only if compute allows).
