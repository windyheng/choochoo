"""Owner: Eval & Deliverables Lead.

REQUIRED DELIVERABLE. Given an image directory, outputs a JSON file with one
entry per image: {"image_path": ..., "pred": <float 0-1>}. `pred` is the
probability the image is AI-generated (higher = more likely AIGC). This
schema is graded — do not rename the fields.

The CLI/JSON plumbing below is complete; only `predict()` (loading the real
checkpoint and running the fused model) is a stub. Build and test the JSON
contract (tests/test_infer_contract.py) against a dummy `predict()` before
the real model exists — don't leave this until Day 7.

Run: python infer.py --input_dir <dir> --out preds.json [--checkpoint <path>]
"""

import argparse
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def predict(image_path: Path, checkpoint_path: str) -> float:
    """Returns a confidence score in [0, 1] that the image is AI-generated."""
    raise NotImplementedError


def run_inference(input_dir: str, checkpoint_path: str) -> list[dict]:
    image_paths = sorted(
        p for p in Path(input_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [
        {"image_path": str(p), "pred": predict(p, checkpoint_path)}
        for p in image_paths
    ]


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
