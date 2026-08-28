"""Owner: Eval & Deliverables Lead.

Mines representative false positives/negatives from evaluate.py's per-sample
predictions (results/predictions.csv, written by
evaluate.py::write_predictions_csv), bucketed by condition (clean + each
transform/severity), and saves a thumbnail grid of the most-confidently-wrong
examples per bucket under results/error_analysis/. Feeds the required "Error
Analysis Note" deliverable — pair each bucket with the false-positive-cost
trade-off discussion (see docs/specs design doc, section 7): a false positive
(real photo flagged as AIGC) carries a reputational/trust cost in a
moderation use case, distinct from a false negative's cost, so the two are
reported and discussed separately rather than folded into one error count.

Run: python error_analysis.py --predictions results/predictions.csv --out results/error_analysis/
"""

import argparse
import csv
from pathlib import Path

from PIL import Image

THUMB_SIZE = (96, 96)
GRID_COLS = 4
TOP_K = 8


def load_predictions(predictions_path: str) -> list[dict]:
    """Reads the per-sample predictions CSV written by
    evaluate.py::write_predictions_csv (image_path, label, condition, branch,
    pred)."""
    with open(predictions_path, newline="") as f:
        return [
            {
                "image_path": row["image_path"],
                "label": int(row["label"]),
                "condition": row["condition"],
                "pred": float(row["pred"]),
            }
            for row in csv.DictReader(f)
        ]


def bucket_errors(rows, threshold=0.5):
    """Groups misclassified rows by condition, split into false positives
    (real image, label 0, predicted AIGC) and false negatives (AIGC image,
    label 1, predicted real). Returns {condition: {"fp": [rows], "fn":
    [rows]}}; correct predictions are dropped."""
    buckets = {}
    for row in rows:
        predicted_aigc = row["pred"] >= threshold
        is_fp = row["label"] == 0 and predicted_aigc
        is_fn = row["label"] == 1 and not predicted_aigc
        if not (is_fp or is_fn):
            continue
        bucket = buckets.setdefault(row["condition"], {"fp": [], "fn": []})
        bucket["fp" if is_fp else "fn"].append(row)
    return buckets


def select_representative(rows, kind, k=TOP_K):
    """The most representative errors are the most *confidently* wrong:
    highest pred for false positives (confidently called AIGC but real),
    lowest pred for false negatives (confidently called real but AIGC)."""
    reverse = kind == "fp"
    return sorted(rows, key=lambda r: r["pred"], reverse=reverse)[:k]


def save_thumbnail_grid(rows, out_path, thumb_size=THUMB_SIZE, cols=GRID_COLS):
    """Saves a grid image of each row's source image (resized to thumb_size)
    to out_path, for visual inspection in the error-analysis writeup. No-op
    if rows is empty (bucket had no errors)."""
    if not rows:
        return
    n_rows = -(-len(rows) // cols)  # ceil
    grid = Image.new("RGB", (thumb_size[0] * cols, thumb_size[1] * n_rows), "white")
    for i, row in enumerate(rows):
        with Image.open(row["image_path"]) as img:
            thumb = img.convert("RGB").resize(thumb_size)
        x = (i % cols) * thumb_size[0]
        y = (i // cols) * thumb_size[1]
        grid.paste(thumb, (x, y))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def mine_errors(predictions_path: str, out_dir: str, threshold: float = 0.5, k: int = TOP_K):
    """Orchestrates the module: load predictions, bucket FP/FN per condition,
    save a thumbnail grid of the top-k most-confident errors per bucket, and
    write results/error_analysis/error_summary.csv with bucket sizes (so
    error counts are visible even for buckets too small/large to eyeball from
    the thumbnail grid alone). Returns the summary rows."""
    rows = load_predictions(predictions_path)
    buckets = bucket_errors(rows, threshold)
    out_dir = Path(out_dir)

    summary = []
    for condition in sorted(buckets):
        for kind in ("fp", "fn"):
            bucket_rows = buckets[condition][kind]
            summary.append({"condition": condition, "kind": kind, "count": len(bucket_rows)})
            representative = select_representative(bucket_rows, kind, k)
            save_thumbnail_grid(representative, out_dir / f"{condition}_{kind}.png")

    summary_path = out_dir / "error_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "kind", "count"])
        writer.writeheader()
        writer.writerows(summary)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="results/error_analysis/")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()
    mine_errors(args.predictions, args.out, threshold=args.threshold, k=args.top_k)


if __name__ == "__main__":
    main()
