"""Owner: Eval & Deliverables Lead. Locks the infer.py JSON contract — this
schema is graded, don't let it drift.
"""

import json

from PIL import Image

import infer


def _write_dummy_image(path):
    Image.new("RGB", (4, 4), color=(128, 64, 32)).save(path)


def test_run_inference_produces_required_schema(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("a.jpg", "b.png"):
        _write_dummy_image(img_dir / name)

    monkeypatch.setattr(infer, "load_model", lambda checkpoint_path, config_path, branch: "dummy-model")
    monkeypatch.setattr(infer, "predict", lambda image, model: 0.5)

    results = infer.run_inference(str(img_dir), checkpoint_path="unused")

    assert len(results) == 2
    for entry in results:
        assert set(entry.keys()) == {"image_path", "pred"}
        assert isinstance(entry["pred"], float)
        assert 0.0 <= entry["pred"] <= 1.0


def test_run_inference_loads_model_once(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("a.jpg", "b.png", "c.jpg"):
        _write_dummy_image(img_dir / name)

    load_calls = []
    monkeypatch.setattr(
        infer,
        "load_model",
        lambda checkpoint_path, config_path, branch: load_calls.append(checkpoint_path) or "dummy-model",
    )
    monkeypatch.setattr(infer, "predict", lambda image, model: 0.5)

    infer.run_inference(str(img_dir), checkpoint_path="ckpt.pt")

    assert load_calls == ["ckpt.pt"]  # loaded once, not once per image


def test_main_writes_valid_json(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    _write_dummy_image(img_dir / "a.jpg")
    out_path = tmp_path / "preds.json"

    monkeypatch.setattr(infer, "load_model", lambda checkpoint_path, config_path, branch: "dummy-model")
    monkeypatch.setattr(infer, "predict", lambda image, model: 0.9)
    monkeypatch.setattr(
        "sys.argv",
        ["infer.py", "--input_dir", str(img_dir), "--out", str(out_path)],
    )

    infer.main()

    data = json.loads(out_path.read_text())
    assert data == [{"image_path": str(img_dir / "a.jpg"), "pred": 0.9}]
