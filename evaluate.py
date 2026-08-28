"""Owner: Eval & Deliverables Lead.

Builds the robustness matrix required by the "Robustness Evaluation Summary"
deliverable: clean + every transform/severity in data/transforms.py + at
least one combined transform (e.g. resize->JPEG), tested individually (per
the brief) rather than requiring the combined condition. Also runs the
ablation (full model vs. CLIP-branch-only vs. artifact-branch-only) per
transform.

The graded technical result is `combined_auc_summary`'s combined score
(clean AUC + mean transformed AUC), not accuracy/FPR/FNR — those stay in the
per-condition rows only to support the error_analysis.py trade-off
discussion, since AUC is ranking-based and doesn't depend on calibration or a
chosen threshold.

Metric functions here can be unit-tested independently of a real checkpoint
(tests/test_transforms.py-style) before a trained model exists — build the
harness early against a dummy/random model, then re-run once real
checkpoints land. `evaluate_condition`/`build_robustness_matrix`/
`run_ablation`/`collect_predictions` take a `predict_fn(PIL.Image) -> float`
callable, so they're already fully testable; only `load_eval_samples`
(depends on the Data Lead's split format) and `load_predict_fns` (depends on
the trained checkpoint + models/*) are stubbed.

`collect_predictions` dumps one row per (sample, condition) — image_path,
label, condition, pred — to results/predictions.csv. This is the input
error_analysis.py mines for representative false positives/negatives; the
aggregate robustness_table.csv (metrics only, no image_path) can't support
that since it has no per-image trace.

Run: python evaluate.py --checkpoint <path> --eval_csv <path> --out results/robustness_table.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from data import transforms

# name -> list of severities to sweep, reusing data/transforms.py's own
# canonical dict (its dispatch names, e.g. "jpeg_quality"/"blur_sigma", must
# match configs/train.yaml's augmentation keys exactly) rather than
# hardcoding a second copy here.
TRANSFORM_SEVERITIES = transforms.TRANSFORM_SEVERITIES

# Combined-transform conditions required by the brief ("redistribution
# scenarios", e.g. resize then re-compress). Each is a list of (name,
# severity) steps applied in sequence.
COMBINED_TRANSFORMS = {
    "resize0.5_then_jpeg70": [("resize_scale", 0.5), ("jpeg_quality", 70)],
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


def _score_samples(samples, predict_fn, steps):
    """Applies `steps` to each sample's image and scores it with predict_fn.
    Returns a list of (image_path, label, pred) tuples, shared by
    evaluate_condition (aggregate metrics) and collect_predictions
    (per-sample dump for error_analysis.py)."""
    scored = []
    for image_path, label in samples:
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
        image = _apply_condition(image, steps)
        scored.append((image_path, label, predict_fn(image)))
    return scored


def evaluate_condition(samples, predict_fn, condition_label, steps, threshold=0.5):
    """samples: list of (image_path, label) pairs. predict_fn: PIL.Image ->
    float confidence in [0, 1]. steps: list of (transform_name, severity) to
    apply before scoring (empty = clean). Returns a metrics dict tagged with
    the condition label."""
    scored = _score_samples(samples, predict_fn, steps)
    row = compute_metrics([label for _, label, _ in scored], [pred for _, _, pred in scored], threshold)
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


def combined_auc_summary(rows):
    """Reduces robustness-matrix rows (as produced by build_robustness_matrix
    / run_ablation) to the graded technical result per branch: clean AUC,
    mean AUC across every transformed condition (individual severities, not
    the combined resize->JPEG condition — that stays a bonus differentiator,
    not part of the graded score), and their average. NaN AUROC rows (single-
    class conditions) are excluded from the mean.

    Returns {branch: {"clean_auc", "transformed_auc", "combined_auc"}}."""
    by_branch = {}
    for row in rows:
        by_branch.setdefault(row["branch"], []).append(row)

    summary = {}
    for branch, branch_rows in by_branch.items():
        clean_auc = next(r["auroc"] for r in branch_rows if r["condition"] == "clean")
        transformed_aurocs = [
            r["auroc"]
            for r in branch_rows
            if r["condition"] != "clean"
            and r["condition"] not in COMBINED_TRANSFORMS
            and not np.isnan(r["auroc"])
        ]
        transformed_auc = float(np.mean(transformed_aurocs)) if transformed_aurocs else float("nan")
        summary[branch] = {
            "clean_auc": clean_auc,
            "transformed_auc": transformed_auc,
            "combined_auc": float(np.mean([clean_auc, transformed_auc])),
        }
    return summary


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["branch", "condition", "auroc", "accuracy", "fpr", "fnr"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def collect_predictions(samples, predict_fn, branch="full"):
    """Per-sample predictions across every condition (clean + each
    transform/severity + combined) — unlike build_robustness_matrix
    (aggregate metrics only), this keeps each image's individual prediction
    so error_analysis.py can trace a representative error back to a file and
    thumbnail it. Only the "full" branch is meaningful here; the ablation
    branches don't need per-image dumps."""
    rows = []
    for condition_label, steps in iter_conditions():
        for image_path, label, pred in _score_samples(samples, predict_fn, steps):
            rows.append(
                {
                    "image_path": str(image_path),
                    "label": label,
                    "condition": condition_label,
                    "branch": branch,
                    "pred": pred,
                }
            )
    return rows


def write_predictions_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "label", "condition", "branch", "pred"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def load_eval_samples(eval_csv: str):
    """Loads (image_path, label) pairs for the held-out eval split, in the
    image_path,label,source format written by
    data/prepare_datasets.py::write_split_csv."""
    with open(eval_csv, newline="") as f:
        return [(row["image_path"], int(row["label"])) for row in csv.DictReader(f)]


def load_predict_fns(checkpoint_path: str):
    """Loads the trained fusion model from checkpoint_path and returns a
    dict of {"full": fn, "clip_only": fn, "artifact_only": fn} predict_fns
    for the ablation, each mapping a PIL.Image -> float confidence. The
    "full" entry should wrap infer.py's load_model()/predict() (via
    `model = infer.load_model(checkpoint_path); lambda image: infer.predict(image, model)`)
    so the robustness matrix and the required inference script never
    diverge; "clip_only"/"artifact_only" need extra hooks into the fusion
    head to score a single branch, not yet exposed by infer.py. Depends on
    models/backbone_clip.py, models/artifact_branch.py, and
    models/fusion_head.py existing."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_csv", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default="results/robustness_table.csv")
    parser.add_argument("--predictions_out", default="results/predictions.csv")
    args = parser.parse_args()

    samples = load_eval_samples(args.eval_csv)
    predict_fns = load_predict_fns(args.checkpoint)
    rows = run_ablation(samples, predict_fns, threshold=args.threshold)
    write_csv(rows, args.out)

    predictions = collect_predictions(samples, predict_fns["full"], branch="full")
    write_predictions_csv(predictions, args.predictions_out)

    summary = combined_auc_summary(rows)
    for branch, metrics in summary.items():
        print(
            f"{branch}: clean_auc={metrics['clean_auc']:.4f} "
            f"transformed_auc={metrics['transformed_auc']:.4f} "
            f"combined_auc={metrics['combined_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
