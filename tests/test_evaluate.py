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


def test_iter_conditions_uses_transforms_dispatch_names():
    # every step name must be one apply_named actually recognizes, so the
    # matrix doesn't crash on a real (non-monkeypatched) transforms module
    valid_names = set(evaluate.transforms.TRANSFORM_SEVERITIES)
    for _, steps in evaluate.iter_conditions():
        for name, _ in steps:
            assert name in valid_names


def test_evaluate_condition_applies_each_step(monkeypatch, samples):
    calls = []

    def fake_apply_named(image, name, severity):
        calls.append((name, severity))
        return image

    monkeypatch.setattr(evaluate.transforms, "apply_named", fake_apply_named)
    evaluate.evaluate_condition(samples, lambda img: 0.5, "jpeg_70", [("jpeg_quality", 70)])

    assert calls == [("jpeg_quality", 70), ("jpeg_quality", 70)]  # once per sample


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


def test_load_eval_samples_reads_split_csv(tmp_path):
    # format written by data/prepare_datasets.py::write_split_csv
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "image_path,label,source\n"
        "data/raw/cifake/test/REAL/0.jpg,0,cifake\n"
        "data/raw/cifake/test/FAKE/0.jpg,1,cifake\n"
    )

    samples = evaluate.load_eval_samples(str(csv_path))

    assert samples == [
        ("data/raw/cifake/test/REAL/0.jpg", 0),
        ("data/raw/cifake/test/FAKE/0.jpg", 1),
    ]


def test_collect_predictions_covers_every_condition_per_sample(samples):
    rows = evaluate.collect_predictions(samples, lambda img: 0.5)
    n_conditions = len(list(evaluate.iter_conditions()))
    assert len(rows) == n_conditions * len(samples)
    assert {row["condition"] for row in rows} == {label for label, _ in evaluate.iter_conditions()}
    assert all(row["branch"] == "full" for row in rows)
    assert all(0.0 <= row["pred"] <= 1.0 for row in rows)


def test_collect_predictions_keeps_per_sample_label_and_path(samples):
    rows = evaluate.collect_predictions(samples, lambda img: 0.5)
    clean_rows = [r for r in rows if r["condition"] == "clean"]
    assert {(r["image_path"], r["label"]) for r in clean_rows} == {
        (str(path), label) for path, label in samples
    }


def test_write_predictions_csv_roundtrip(tmp_path):
    rows = [
        {"image_path": "img0.png", "label": 0, "condition": "clean", "branch": "full", "pred": 0.1},
        {"image_path": "img1.png", "label": 1, "condition": "clean", "branch": "full", "pred": 0.9},
    ]
    out_path = tmp_path / "nested" / "predictions.csv"
    evaluate.write_predictions_csv(rows, out_path)

    with out_path.open() as f:
        reader = list(csv.DictReader(f))
    assert reader == [
        {"image_path": "img0.png", "label": "0", "condition": "clean", "branch": "full", "pred": "0.1"},
        {"image_path": "img1.png", "label": "1", "condition": "clean", "branch": "full", "pred": "0.9"},
    ]


def test_combined_auc_summary_averages_clean_and_transformed():
    rows = [
        {"branch": "full", "condition": "clean", "auroc": 1.0},
        {"branch": "full", "condition": "jpeg_quality_90", "auroc": 0.8},
        {"branch": "full", "condition": "jpeg_quality_70", "auroc": 0.6},
        {"branch": "full", "condition": "resize0.5_then_jpeg70", "auroc": 0.4},
    ]
    summary = evaluate.combined_auc_summary(rows)
    assert summary["full"]["clean_auc"] == 1.0
    assert summary["full"]["transformed_auc"] == pytest.approx(0.7)  # mean(0.8, 0.6); combined excluded
    assert summary["full"]["combined_auc"] == pytest.approx(0.85)  # mean(1.0, 0.7)


def test_combined_auc_summary_skips_nan_transformed_rows():
    rows = [
        {"branch": "full", "condition": "clean", "auroc": 1.0},
        {"branch": "full", "condition": "jpeg_quality_90", "auroc": float("nan")},
        {"branch": "full", "condition": "jpeg_quality_70", "auroc": 0.5},
    ]
    summary = evaluate.combined_auc_summary(rows)
    assert summary["full"]["transformed_auc"] == 0.5


def test_combined_auc_summary_per_branch():
    rows = [
        {"branch": "full", "condition": "clean", "auroc": 0.9},
        {"branch": "full", "condition": "jpeg_quality_90", "auroc": 0.7},
        {"branch": "clip_only", "condition": "clean", "auroc": 0.8},
        {"branch": "clip_only", "condition": "jpeg_quality_90", "auroc": 0.75},
    ]
    summary = evaluate.combined_auc_summary(rows)
    assert set(summary) == {"full", "clip_only"}
    assert summary["clip_only"]["clean_auc"] == 0.8


def test_load_predict_fns_wraps_infer_for_each_branch_given(tmp_path):
    """Network-free: uses artifact_only (no CLIP download) to prove
    load_predict_fns builds a working predict_fn per checkpoint given,
    wrapping infer.py's real load_model/predict."""
    import train

    config_path = "configs/train.yaml"
    config = train.load_config(config_path)
    model = train.build_model(config, branch="artifact_only")
    optimizer = train.build_optimizer(model, config)
    ckpt_path = tmp_path / "ckpt.pt"
    train.save_checkpoint(ckpt_path, model, optimizer, epoch=0, global_step=1)

    predict_fns = evaluate.load_predict_fns({"artifact_only": str(ckpt_path)}, config_path)

    assert set(predict_fns) == {"artifact_only"}
    image = Image.new("RGB", (64, 64), color=(10, 20, 30))
    pred = predict_fns["artifact_only"](image)
    assert isinstance(pred, float)
    assert 0.0 <= pred <= 1.0


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
