"""Owner: Eval & Deliverables Lead.

Tests error_analysis.py's bucketing/selection/thumbnailing logic against
synthetic predictions and tiny real images (no trained model needed — this
mirrors the Day-1 pattern used for evaluate.py/infer.py: build and prove the
harness before real checkpoint output exists).
"""

import csv

import pytest
from PIL import Image

import error_analysis


@pytest.fixture
def image_dir(tmp_path):
    paths = {}
    for name, color in [("real0", (255, 0, 0)), ("real1", (0, 255, 0)), ("fake0", (0, 0, 255))]:
        p = tmp_path / f"{name}.png"
        Image.new("RGB", (8, 8), color=color).save(p)
        paths[name] = p
    return paths


def test_bucket_errors_splits_fp_and_fn():
    rows = [
        {"image_path": "a.png", "label": 0, "condition": "clean", "pred": 0.9},  # FP
        {"image_path": "b.png", "label": 1, "condition": "clean", "pred": 0.1},  # FN
        {"image_path": "c.png", "label": 0, "condition": "clean", "pred": 0.1},  # correct
        {"image_path": "d.png", "label": 1, "condition": "clean", "pred": 0.9},  # correct
    ]
    buckets = error_analysis.bucket_errors(rows)
    assert [r["image_path"] for r in buckets["clean"]["fp"]] == ["a.png"]
    assert [r["image_path"] for r in buckets["clean"]["fn"]] == ["b.png"]


def test_bucket_errors_groups_by_condition():
    rows = [
        {"image_path": "a.png", "label": 0, "condition": "clean", "pred": 0.9},
        {"image_path": "b.png", "label": 0, "condition": "jpeg_quality_30", "pred": 0.9},
    ]
    buckets = error_analysis.bucket_errors(rows)
    assert set(buckets) == {"clean", "jpeg_quality_30"}


def test_bucket_errors_respects_threshold():
    rows = [{"image_path": "a.png", "label": 0, "condition": "clean", "pred": 0.6}]
    assert error_analysis.bucket_errors(rows, threshold=0.5) != {}
    assert error_analysis.bucket_errors(rows, threshold=0.7) == {}


def test_select_representative_fp_orders_by_highest_confidence():
    rows = [
        {"image_path": "a.png", "pred": 0.6},
        {"image_path": "b.png", "pred": 0.95},
        {"image_path": "c.png", "pred": 0.7},
    ]
    selected = error_analysis.select_representative(rows, "fp", k=2)
    assert [r["image_path"] for r in selected] == ["b.png", "c.png"]


def test_select_representative_fn_orders_by_lowest_confidence():
    rows = [
        {"image_path": "a.png", "pred": 0.4},
        {"image_path": "b.png", "pred": 0.05},
        {"image_path": "c.png", "pred": 0.3},
    ]
    selected = error_analysis.select_representative(rows, "fn", k=2)
    assert [r["image_path"] for r in selected] == ["b.png", "c.png"]


def test_load_predictions_reads_csv(tmp_path):
    # format written by evaluate.py::write_predictions_csv
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "image_path,label,condition,branch,pred\n"
        "a.png,0,clean,full,0.9\n"
        "b.png,1,jpeg_quality_30,full,0.2\n"
    )
    rows = error_analysis.load_predictions(str(csv_path))
    assert rows == [
        {"image_path": "a.png", "label": 0, "condition": "clean", "pred": 0.9},
        {"image_path": "b.png", "label": 1, "condition": "jpeg_quality_30", "pred": 0.2},
    ]


def test_save_thumbnail_grid_writes_expected_size(tmp_path, image_dir):
    rows = [{"image_path": str(image_dir["real0"])}, {"image_path": str(image_dir["real1"])}]
    out_path = tmp_path / "grid.png"
    error_analysis.save_thumbnail_grid(rows, out_path, thumb_size=(10, 10), cols=4)

    assert out_path.exists()
    with Image.open(out_path) as grid:
        assert grid.size == (40, 10)  # 2 images, cols=4 -> 1 row of 4 thumb-widths


def test_save_thumbnail_grid_skips_when_empty(tmp_path):
    out_path = tmp_path / "grid.png"
    error_analysis.save_thumbnail_grid([], out_path)
    assert not out_path.exists()


def test_mine_errors_writes_summary_and_thumbnails(tmp_path, image_dir):
    predictions_path = tmp_path / "predictions.csv"
    with predictions_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "condition", "branch", "pred"])
        writer.writeheader()
        writer.writerow(
            {"image_path": str(image_dir["real0"]), "label": 0, "condition": "clean", "branch": "full", "pred": 0.9}
        )
        writer.writerow(
            {"image_path": str(image_dir["fake0"]), "label": 1, "condition": "clean", "branch": "full", "pred": 0.9}
        )  # correct, should not appear anywhere

    out_dir = tmp_path / "error_analysis"
    summary = error_analysis.mine_errors(str(predictions_path), str(out_dir))

    assert {"condition": "clean", "kind": "fp", "count": 1} in summary
    assert {"condition": "clean", "kind": "fn", "count": 0} in summary
    assert (out_dir / "clean_fp.png").exists()
    assert not (out_dir / "clean_fn.png").exists()  # empty bucket -> no thumbnail file

    with (out_dir / "error_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {"condition": "clean", "kind": "fp", "count": "1"} in rows
