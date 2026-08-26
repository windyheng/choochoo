"""Owner: Eval & Deliverables Lead. Locks the infer.py JSON contract — this
schema is graded, don't let it drift.
"""

import json

import infer


def test_run_inference_produces_required_schema(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("a.jpg", "b.png"):
        (img_dir / name).write_bytes(b"not a real image, contract test only")

    monkeypatch.setattr(infer, "predict", lambda image_path, checkpoint_path: 0.5)

    results = infer.run_inference(str(img_dir), checkpoint_path="unused")

    assert len(results) == 2
    for entry in results:
        assert set(entry.keys()) == {"image_path", "pred"}
        assert isinstance(entry["pred"], float)
        assert 0.0 <= entry["pred"] <= 1.0


def test_main_writes_valid_json(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "a.jpg").write_bytes(b"not a real image, contract test only")
    out_path = tmp_path / "preds.json"

    monkeypatch.setattr(infer, "predict", lambda image_path, checkpoint_path: 0.9)
    monkeypatch.setattr(
        "sys.argv",
        ["infer.py", "--input_dir", str(img_dir), "--out", str(out_path)],
    )

    infer.main()

    data = json.loads(out_path.read_text())
    assert data == [{"image_path": str(img_dir / "a.jpg"), "pred": 0.9}]
