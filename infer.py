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

The CLI/JSON plumbing below is complete; only `load_model()`/`predict()`
(loading the real checkpoint and running the fused model) are stubs. Build
and test the JSON contract (tests/test_infer_contract.py) against a dummy
`predict()` before the real model exists — don't leave this until Day 7.

Run: python infer.py --input_dir <dir> --out preds.json [--checkpoint <path>]
"""

import argparse
import json
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model(checkpoint_path: str):
    """Loads the trained fusion model (CLIP branch + SRM branch + fusion
    head) once from checkpoint_path. Depends on models/backbone_clip.py,
    models/artifact_branch.py, and models/fusion_head.py existing."""
    raise NotImplementedError


def predict(image: Image.Image, model) -> float:
    """Returns a confidence score in [0, 1] that `image` (already loaded, RGB)
    is AI-generated, using an already-loaded `model` (see load_model). Takes
    an in-memory image rather than a path so evaluate.py's robustness matrix
    can call this on transformed (JPEG/blurred/etc.) images without writing
    them to disk first."""
    raise NotImplementedError


def run_inference(input_dir: str, checkpoint_path: str) -> list[dict]:
    model = load_model(checkpoint_path)
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
    args = parser.parse_args()

    results = run_inference(args.input_dir, args.checkpoint)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
