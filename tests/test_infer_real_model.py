"""Owner: Eval & Deliverables Lead.

Integration tests for infer.py's real load_model/predict against an actual
(freshly-initialized, untrained) checkpoint in train.py's real format —
proves the checkpoint contract holds (state_dict keys, branch handling,
preprocessing) without needing an actual trained model. `artifact_only`
needs no network access and runs by default; `full`/`clip_only` additionally
load the real CLIP backbone (~600MB download on first run), so those are
marked `slow` like tests/test_backbone_clip.py.
"""

import pytest

import infer
import train

CONFIG_PATH = "configs/train.yaml"


def _save_fake_checkpoint(path, branch):
    config = train.load_config(CONFIG_PATH)
    model = train.build_model(config, branch=branch)
    optimizer = train.build_optimizer(model, config)
    train.save_checkpoint(path, model, optimizer, epoch=0, global_step=1)
    return path


def test_load_model_and_predict_artifact_only(tmp_path):
    from PIL import Image

    ckpt = _save_fake_checkpoint(tmp_path / "ckpt_step_00000001.pt", "artifact_only")
    model = infer.load_model(str(ckpt), CONFIG_PATH, branch="artifact_only")

    assert model["backbone"] is None
    assert model["artifact_branch"] is not None

    image = Image.new("RGB", (64, 64), color=(120, 80, 40))
    pred = infer.predict(image, model)

    assert isinstance(pred, float)
    assert 0.0 <= pred <= 1.0


def test_load_model_rejects_dim_mismatch(tmp_path):
    """A checkpoint saved under one srm_out_channels config must fail loudly
    (state_dict shape mismatch) rather than silently loading into a
    differently-shaped model and producing wrong-scale predictions. Uses
    artifact_only for both sides so this stays network-free (no CLIP
    download) — a branch mismatch would hit the same load_state_dict check."""
    import yaml

    config = train.load_config(CONFIG_PATH)
    ckpt = _save_fake_checkpoint(tmp_path / "ckpt_step_00000001.pt", "artifact_only")

    bad_config = dict(config)
    bad_config["model"] = {**config["model"], "srm_out_channels": config["model"]["srm_out_channels"] + 8}
    bad_config_path = tmp_path / "bad_config.yaml"
    bad_config_path.write_text(yaml.safe_dump(bad_config))

    with pytest.raises(RuntimeError):
        infer.load_model(str(ckpt), str(bad_config_path), branch="artifact_only")


@pytest.mark.slow
def test_load_model_and_predict_full(tmp_path):
    from PIL import Image

    ckpt = _save_fake_checkpoint(tmp_path / "ckpt_step_00000001.pt", "full")
    model = infer.load_model(str(ckpt), CONFIG_PATH, branch="full")

    assert model["backbone"] is not None
    assert model["artifact_branch"] is not None

    image = Image.new("RGB", (64, 64), color=(120, 80, 40))
    pred = infer.predict(image, model)

    assert isinstance(pred, float)
    assert 0.0 <= pred <= 1.0
