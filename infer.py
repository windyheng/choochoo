"""Owner: Eval & Deliverables Lead.

REQUIRED DELIVERABLE. Given an image directory, outputs a JSON file with one
entry per image: {"image_path": ..., "pred": <float 0-1>}. `pred` is the
probability the image is AI-generated (higher = more likely AIGC). This
schema is graded — do not rename the fields. AUC (the graded metric) only
depends on the ranking of these scores, not their calibration, so `predict`
does not need Platt/temperature scaling to score well — it still needs to be
a real probability-shaped number for the JSON contract and for the
threshold-based FPR/FNR discussion in error_analysis.py.

`load_model`/`predict` here are the single scoring path shared with
evaluate.py's robustness matrix (see load_predict_fns in evaluate.py) — the
"running script that produces confidence scores" and "the scores that feed
the evaluation matrix" must be provably the same code, not two
implementations that can drift apart.

Loads the real checkpoint train.py produces: a ModuleDict state_dict covering
"artifact_branch" (omitted for a clip_only-trained checkpoint) and
"fusion_head" — the frozen CLIP backbone is never checkpointed (see
train.py::build_model's docstring), so it's rebuilt from configs/train.yaml
and combined with the checkpoint at load time, exactly like train.py's
main() does. `--branch` must match how the checkpoint was actually trained
(train.py's own `--branch` flag) — there's no way to detect this from the
checkpoint file itself, so a mismatch fails loudly via a state_dict key
error rather than silently producing wrong scores.

Run: python infer.py --input_dir <dir> --out preds.json [--checkpoint <path>] [--config <path>] [--branch full|clip_only|artifact_only]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.dataset import CANONICAL_SIZE
from models.artifact_branch import ArtifactBranch
from models.backbone_clip import CLIPBackbone
from models.fusion_head import FusionHead
from train import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model(checkpoint_path: str, config_path: str = "configs/train.yaml", branch: str = "full"):
    """Loads the frozen CLIP backbone (rebuilt from config, never
    checkpointed) plus the trained artifact branch / fusion head (restored
    from checkpoint_path's "model_state_dict", written by
    train.py::save_checkpoint). Returns a dict consumed by predict() —
    {"backbone": CLIPBackbone | None, "artifact_branch": ArtifactBranch | None,
    "fusion_head": FusionHead}."""
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = None
    if branch != "artifact_only":
        backbone = CLIPBackbone.from_config(config).to(device)
        backbone.eval()

    artifact_branch = None
    modules = {}
    if branch != "clip_only":
        artifact_branch = ArtifactBranch(out_channels=config["model"]["srm_out_channels"])
        modules["artifact_branch"] = artifact_branch
    fusion_head = FusionHead.from_config(config, branch=branch)
    modules["fusion_head"] = fusion_head
    model = torch.nn.ModuleDict(modules).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {"backbone": backbone, "artifact_branch": artifact_branch, "fusion_head": fusion_head, "device": device}


def _to_model_input(image: Image.Image, device) -> torch.Tensor:
    """PIL.Image -> [1, 3, 224, 224] float32 in [0, 1], matching
    data/dataset.py::AIGCDataset's preprocessing exactly (same resize,
    scaling, and channel order the model was trained on)."""
    resized = image.resize(CANONICAL_SIZE, Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def predict(image: Image.Image, model) -> float:
    """Returns a confidence score in [0, 1] that `image` (already loaded, RGB)
    is AI-generated, using an already-loaded `model` (see load_model). Takes
    an in-memory image rather than a path so evaluate.py's robustness matrix
    can call this on transformed (JPEG/blurred/etc.) images without writing
    them to disk first."""
    tensor = _to_model_input(image, model["device"])

    with torch.no_grad():
        clip_embed = model["backbone"].embed(tensor) if model["backbone"] is not None else None
        srm_embed = model["artifact_branch"].embed(tensor) if model["artifact_branch"] is not None else None
        prob = model["fusion_head"].predict_proba(clip_embed, srm_embed)

    return float(prob.item())


def run_inference(
    input_dir: str, checkpoint_path: str, config_path: str = "configs/train.yaml", branch: str = "full"
) -> list[dict]:
    model = load_model(checkpoint_path, config_path, branch)
    image_paths = sorted(
        p for p in Path(input_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    results = []
    for p in image_paths:
        with Image.open(p) as raw:
            image = raw.convert("RGB")
        results.append({"image_path": str(p), "pred": predict(image, model)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out", default="preds.json")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--branch", choices=["full", "clip_only", "artifact_only"], default="full")
    args = parser.parse_args()

    results = run_inference(args.input_dir, args.checkpoint, args.config, args.branch)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
