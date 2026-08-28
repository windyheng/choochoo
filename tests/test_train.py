"""Tests for the checkpoint/resume mechanism in train.py, exercised against
dummy tensors and a tiny model."""

import random
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import train


def _tiny_model():
    return train.DummyFusionHead(clip_dim=4, srm_dim=4, hidden_dim=8)


def _tiny_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)


def _tiny_dataloader(n_samples=8, batch_size=2, seed=0):
    dataset = train.DummyEmbeddingDataset(n_samples, clip_dim=4, srm_dim=4, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def test_find_latest_checkpoint_missing_dir_returns_none(tmp_path):
    assert train.find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_find_latest_checkpoint_empty_dir_returns_none(tmp_path):
    assert train.find_latest_checkpoint(tmp_path) is None


def test_find_latest_checkpoint_picks_highest_step_not_mtime(tmp_path):
    low_step = tmp_path / "ckpt_step_00000005.pt"
    high_step = tmp_path / "ckpt_step_00000010.pt"
    (tmp_path / "notes.txt").write_text("not a checkpoint")

    high_step.write_bytes(b"high")
    time.sleep(0.01)
    low_step.write_bytes(b"low")  # newer mtime, lower step

    assert train.find_latest_checkpoint(tmp_path) == high_step


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model_a = _tiny_model()
    optimizer_a = _tiny_optimizer(model_a)
    batch = next(iter(_tiny_dataloader()))
    train.train_step(model_a, optimizer_a, batch, device="cpu")  # give optimizer real momentum state

    path = tmp_path / "ckpt_step_00000001.pt"
    train.save_checkpoint(path, model_a, optimizer_a, epoch=0, global_step=1)

    model_b = _tiny_model()
    optimizer_b = _tiny_optimizer(model_b)
    meta = train.load_checkpoint(path, model_b, optimizer_b, device="cpu")

    assert meta == {"epoch": 0, "global_step": 1}
    for key, value in model_a.state_dict().items():
        assert torch.equal(value, model_b.state_dict()[key])

    state_a = optimizer_a.state_dict()["state"]
    state_b = optimizer_b.state_dict()["state"]
    for param_id in state_a:
        assert torch.equal(state_a[param_id]["exp_avg"], state_b[param_id]["exp_avg"])
        assert torch.equal(state_a[param_id]["exp_avg_sq"], state_b[param_id]["exp_avg_sq"])


def test_rng_state_roundtrip():
    state = train.get_rng_state()
    a = (random.random(), np.random.rand(), torch.rand(1))
    train.set_rng_state(state)
    b = (random.random(), np.random.rand(), torch.rand(1))

    assert a[0] == b[0]
    assert a[1] == b[1]
    assert torch.equal(a[2], b[2])


def test_train_checkpoints_at_configured_step_interval(tmp_path):
    model = _tiny_model()
    optimizer = _tiny_optimizer(model)
    dataloader = _tiny_dataloader(n_samples=8, batch_size=2)  # 4 steps/epoch

    train.train(
        model,
        dataloader,
        optimizer,
        epochs=1,
        checkpoint_dir=tmp_path,
        checkpoint_every_steps=2,
        device="cpu",
    )

    assert (tmp_path / "ckpt_step_00000002.pt").exists()
    assert (tmp_path / "ckpt_step_00000004.pt").exists()
    assert not (tmp_path / "ckpt_step_00000001.pt").exists()
    assert not (tmp_path / "ckpt_step_00000003.pt").exists()


def test_train_resumes_and_matches_uninterrupted_run(tmp_path):
    dataloader = _tiny_dataloader(n_samples=8, batch_size=2, seed=0)

    torch.manual_seed(0)
    model_ref = _tiny_model()
    optimizer_ref = _tiny_optimizer(model_ref)
    train.train(
        model_ref,
        dataloader,
        optimizer_ref,
        epochs=2,
        checkpoint_dir=tmp_path / "ref",
        checkpoint_every_steps=4,
        device="cpu",
    )

    torch.manual_seed(0)
    model_x = _tiny_model()  # identical init to model_ref via the same seed
    optimizer_x = _tiny_optimizer(model_x)
    final_step = train.train(
        model_x,
        dataloader,
        optimizer_x,
        epochs=1,
        checkpoint_dir=tmp_path / "resume",
        checkpoint_every_steps=4,
        device="cpu",
    )
    assert final_step == 4

    latest = train.find_latest_checkpoint(tmp_path / "resume")
    assert latest == tmp_path / "resume" / "ckpt_step_00000004.pt"

    model_y = _tiny_model()  # deliberately NOT seeded to match model_x/model_ref
    optimizer_y = _tiny_optimizer(model_y)
    meta = train.load_checkpoint(latest, model_y, optimizer_y, device="cpu")
    assert meta["global_step"] == 4

    final_step = train.train(
        model_y,
        dataloader,
        optimizer_y,
        epochs=2,
        checkpoint_dir=tmp_path / "resume",
        checkpoint_every_steps=4,
        start_global_step=meta["global_step"],
        device="cpu",
    )

    assert final_step == 8
    for key, value in model_ref.state_dict().items():
        assert torch.allclose(value, model_y.state_dict()[key])
