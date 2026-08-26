"""Owner: Eval & Deliverables Lead.

Tests evaluate.py's metric/harness logic against a dummy predict_fn and a
monkeypatched data.transforms.apply_named (still a stub owned by the Data
Lead), per the Day-1 plan: build and prove the harness before a real
checkpoint or real transforms exist, then re-run unchanged once they land.
"""

import csv
import math

import pytest
from PIL import Image

import evaluate


@pytest.fixture
def samples(tmp_path):
    paths = []
    for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), color=color).save(p)
        paths.append(p)
    # (image_path, label): first real (0), second AIGC (1)
    return [(paths[0], 0), (paths[1], 1)]


@pytest.fixture(autouse=True)
def identity_transform(monkeypatch):
    """apply_named is still NotImplementedError (Data Lead's stub) — patch it
    to a no-op identity so the harness is testable today; swap out once the
    real implementation lands."""
    monkeypatch.setattr(evaluate.transforms, "apply_named", lambda image, name, severity: image)


def test_compute_metrics_perfect_predictions():
    metrics = evaluate.compute_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["auroc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["fnr"] == 0.0


def test_compute_metrics_single_class_auroc_is_nan():
    metrics = evaluate.compute_metrics([0, 0, 0], [0.1, 0.4, 0.2])
    assert math.isnan(metrics["auroc"])
    assert metrics["accuracy"] == 1.0
    assert metrics["fpr"] == 0.0
    assert math.isnan(metrics["fnr"])  # no positives to compute FNR over


def test_compute_metrics_threshold_is_respected():
    metrics = evaluate.compute_metrics([0, 1], [0.6, 0.6], threshold=0.7)
    assert metrics["fpr"] == 0.0  # 0.6 < 0.7 -> predicted negative
    assert metrics["fnr"] == 1.0  # true positive predicted negative


def test_iter_conditions_starts_clean_and_includes_combined():
    conditions = list(evaluate.iter_conditions())
    labels = [label for label, _ in conditions]
    assert labels[0] == "clean"
    assert "resize0.5_then_jpeg70" in labels
    assert len(labels) == len(set(labels))  # no duplicate condition labels


def test_evaluate_condition_applies_each_step(monkeypatch, samples):
    calls = []

    def fake_apply_named(image, name, severity):
        calls.append((name, severity))
        return image

    monkeypatch.setattr(evaluate.transforms, "apply_named", fake_apply_named)
    evaluate.evaluate_condition(samples, lambda img: 0.5, "jpeg_70", [("jpeg", 70)])

    assert calls == [("jpeg", 70), ("jpeg", 70)]  # once per sample


def test_evaluate_condition_clean_skips_transform(monkeypatch, samples):
    calls = []
    monkeypatch.setattr(evaluate.transforms, "apply_named", lambda i, n, s: calls.append((n, s)) or i)
    evaluate.evaluate_condition(samples, lambda img: 0.5, "clean", [])
    assert calls == []


def test_build_robustness_matrix_covers_every_condition(samples):
    rows = evaluate.build_robustness_matrix(samples, lambda img: 0.5)
    expected_labels = {label for label, _ in evaluate.iter_conditions()}
    assert {row["condition"] for row in rows} == expected_labels
    assert all(row["branch"] == "full" for row in rows)


def test_run_ablation_tags_each_branch(samples):
    predict_fns = {
        "full": lambda img: 0.9,
        "clip_only": lambda img: 0.6,
        "artifact_only": lambda img: 0.4,
    }
    rows = evaluate.run_ablation(samples, predict_fns)
    n_conditions = len(list(evaluate.iter_conditions()))
    assert len(rows) == n_conditions * 3
    assert {row["branch"] for row in rows} == set(predict_fns)


def test_write_csv_roundtrip(tmp_path):
    rows = [
        {"branch": "full", "condition": "clean", "auroc": 0.95, "accuracy": 0.9, "fpr": 0.05, "fnr": 0.1},
    ]
    out_path = tmp_path / "nested" / "robustness_table.csv"
    evaluate.write_csv(rows, out_path)

    with out_path.open() as f:
        reader = list(csv.DictReader(f))
    assert reader == [
        {"branch": "full", "condition": "clean", "auroc": "0.95", "accuracy": "0.9", "fpr": "0.05", "fnr": "0.1"}
    ]
