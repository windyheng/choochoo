"""Owner: Eval & Deliverables Lead.

Builds the robustness matrix required by the "Robustness Evaluation Summary"
deliverable: clean + every transform/severity in data/transforms.py + at
least one combined transform (e.g. resize->JPEG). Also runs the ablation
(full model vs. CLIP-branch-only vs. artifact-branch-only) per transform.

Metric functions here can be unit-tested independently of a real checkpoint
(tests/test_transforms.py-style) before a trained model exists — build the
harness early against a dummy/random model, then re-run once real
checkpoints land. `evaluate_condition`/`build_robustness_matrix`/
`run_ablation` take a `predict_fn(PIL.Image) -> float` callable, so they're
already fully testable; only `load_eval_samples` (depends on the Data Lead's
split format) and `load_predict_fns` (depends on the trained checkpoint +
models/*) are stubbed.

Run: python evaluate.py --checkpoint <path> --eval_csv <path> --out results/robustness_table.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from data import transforms

# name -> list of severities to sweep. Names here are the contract that
# data/transforms.py::apply_named must dispatch on (see docs/interfaces.md
# section 4) — keep in sync with the Data Lead.
TRANSFORM_SEVERITIES = {
    "jpeg": transforms.JPEG_QUALITIES,
    "blur": transforms.BLUR_SIGMAS,
    "resize": transforms.RESIZE_SCALES,
    "noise": transforms.NOISE_SIGMAS,
    "color_jitter": [transforms.COLOR_JITTER_PCT],
    "crop": [transforms.CENTER_CROP_FRAC],
}

# Combined-transform conditions required by the brief ("redistribution
# scenarios", e.g. resize then re-compress). Each is a list of (name,
# severity) steps applied in sequence.
COMBINED_TRANSFORMS = {
    "resize0.5_then_jpeg70": [("resize", 0.5), ("jpeg", 70)],
}

ABLATION_BRANCHES = ("full", "clip_only", "artifact_only")


def compute_metrics(y_true, y_pred, threshold=0.5):
    """y_true: 0/1 labels. y_pred: confidence scores in [0, 1]. Returns
    AUROC, accuracy, FPR, FNR at the given threshold. AUROC is NaN if only
    one class is present in y_true (can't be computed)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    auroc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")

    y_hat = (y_pred >= threshold).astype(int)
    accuracy = float((y_hat == y_true).mean())

    negatives = y_true == 0
    positives = y_true == 1
    fpr = float((y_hat[negatives] == 1).mean()) if negatives.any() else float("nan")
    fnr = float((y_hat[positives] == 0).mean()) if positives.any() else float("nan")

    return {"auroc": auroc, "accuracy": accuracy, "fpr": fpr, "fnr": fnr}


def _apply_condition(image, steps):
    """steps: list of (transform_name, severity) applied in sequence via
    data.transforms.apply_named. Empty list = clean (no transform)."""
    for name, severity in steps:
        image = transforms.apply_named(image, name, severity)
    return image


def evaluate_condition(samples, predict_fn, condition_label, steps, threshold=0.5):
    """samples: list of (image_path, label) pairs. predict_fn: PIL.Image ->
    float confidence in [0, 1]. steps: list of (transform_name, severity) to
    apply before scoring (empty = clean). Returns a metrics dict tagged with
    the condition label."""
    y_true = []
    y_pred = []
    for image_path, label in samples:
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        image = _apply_condition(image, steps)
        y_true.append(label)
        y_pred.append(predict_fn(image))

    row = compute_metrics(y_true, y_pred, threshold)
    row["condition"] = condition_label
    return row


def iter_conditions():
    """Yields (condition_label, steps) for clean + every transform/severity +
    every combined condition, in the order they should appear in the table."""
    yield "clean", []
    for name, severities in TRANSFORM_SEVERITIES.items():
        for severity in severities:
            yield f"{name}_{severity}", [(name, severity)]
    for label, steps in COMBINED_TRANSFORMS.items():
        yield label, steps


def build_robustness_matrix(samples, predict_fn, threshold=0.5, branch="full"):
    """Runs predict_fn over every required condition (clean, each
    transform/severity, combined) and returns a list of metric-dict rows,
    each tagged with `condition` and `branch`."""
    rows = []
    for condition_label, steps in iter_conditions():
        row = evaluate_condition(samples, predict_fn, condition_label, steps, threshold)
        row["branch"] = branch
        rows.append(row)
    return rows


def run_ablation(samples, predict_fns, threshold=0.5):
    """predict_fns: dict mapping branch name ("full", "clip_only",
    "artifact_only") -> predict_fn. Returns the concatenated robustness
    matrix rows across all branches, so the same table drives both the
    headline robustness summary (branch="full") and the ablation
    comparison."""
    rows = []
    for branch, predict_fn in predict_fns.items():
        rows.extend(build_robustness_matrix(samples, predict_fn, threshold, branch=branch))
    return rows


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["branch", "condition", "auroc", "accuracy", "fpr", "fnr"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def load_eval_samples(eval_csv: str):
    """Loads (image_path, label) pairs for the held-out eval split. Format
    depends on the Data Lead's split CSV (data/dataset.py) — wire this once
    that lands."""
    raise NotImplementedError


def load_predict_fns(checkpoint_path: str):
    """Loads the trained fusion model from checkpoint_path and returns a
    dict of {"full": fn, "clip_only": fn, "artifact_only": fn} predict_fns
    for the ablation, each mapping a PIL.Image -> float confidence. Depends
    on models/backbone_clip.py, models/artifact_branch.py, and
    models/fusion_head.py existing."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="results/robustness_table.csv")
    args = parser.parse_args()

    samples = load_eval_samples(args.eval_csv)
    predict_fns = load_predict_fns(args.checkpoint)
    rows = run_ablation(samples, predict_fns, threshold=args.threshold)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
