"""Owner: Training/Infra Lead.

End-to-end smoke test for the real (non-Dummy) assembled pipeline:
build_model() + build_dataloader() + real_train_step() against a handful of
real images from data/cache/splits/train.csv. Loads the real CLIP ViT-B/16
checkpoint (~600 MB download on first run), so marked `slow` and skipped
unless `pytest --runslow` (see conftest.py and tests/test_backbone_clip.py,
which follows the same convention).
"""

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("open_clip")

import train
from data.paths import REPO_ROOT
from models.backbone_clip import CLIPBackbone

pytestmark = pytest.mark.slow

CONFIG = {
    "data": {"splits_dir": "data/cache/splits", "batch_size": 4, "num_workers": 0},
    "model": {
        "clip_model": "ViT-B-16",
        "clip_pretrained": "openai",
        "freeze_clip": True,
        "srm_out_channels": 32,
        "fusion_hidden_dim": 16,
    },
    "train": {"lr": 1e-3, "weight_decay": 1e-4},
}


@pytest.fixture(scope="module")
def backbone():
    return CLIPBackbone.from_config(CONFIG).eval()


def test_build_dataloader_yields_real_image_batches():
    dataloader = train.build_dataloader(CONFIG, split="train", augment=False)
    images, labels, paths = next(iter(dataloader))

    assert images.shape == (4, 3, 224, 224)
    assert images.dtype == torch.float32
    assert ((0.0 <= images) & (images <= 1.0)).all()
    assert labels.shape == (4,)
    assert len(paths) == 4
    assert all((REPO_ROOT / p).exists() for p in paths)


def test_real_train_step_updates_artifact_branch_and_fusion_head(backbone):
    model = train.build_model(CONFIG, branch="full")
    model.train()
    optimizer = train.build_optimizer(model, CONFIG)
    dataloader = train.build_dataloader(CONFIG, split="train", augment=False)
    batch = next(iter(dataloader))

    before = copy.deepcopy(model.state_dict())
    loss = train.real_train_step(model, optimizer, batch, "cpu", backbone=backbone, branch="full")
    after = model.state_dict()

    assert isinstance(loss, float)
    assert loss == loss  # not NaN
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert any(k.startswith("artifact_branch.") for k in changed), "artifact_branch did not train"
    assert any(k.startswith("fusion_head.") for k in changed), "fusion_head did not train"


def test_real_train_step_clip_only_does_not_build_or_train_artifact_branch(backbone):
    model = train.build_model(CONFIG, branch="clip_only")
    assert "artifact_branch" not in model
    model.train()
    optimizer = train.build_optimizer(model, CONFIG)
    dataloader = train.build_dataloader(CONFIG, split="train", augment=False)
    batch = next(iter(dataloader))

    loss = train.real_train_step(model, optimizer, batch, "cpu", backbone=backbone, branch="clip_only")
    assert isinstance(loss, float)


def test_real_train_step_artifact_only_skips_backbone():
    model = train.build_model(CONFIG, branch="artifact_only")
    model.train()
    optimizer = train.build_optimizer(model, CONFIG)
    dataloader = train.build_dataloader(CONFIG, split="train", augment=False)
    batch = next(iter(dataloader))

    loss = train.real_train_step(
        model, optimizer, batch, "cpu", backbone=None, branch="artifact_only"
    )
    assert isinstance(loss, float)


def test_tiny_no_aug_baseline_runs_end_to_end(tmp_path, backbone):
    """Mirrors what `python train.py --config configs/train.yaml` does for a
    few steps — the actual Day-2 deliverable: prove the assembled pipeline
    runs and checkpoints, not just that individual pieces work in isolation."""
    model = train.build_model(CONFIG, branch="full")
    model.train()
    optimizer = train.build_optimizer(model, CONFIG)
    dataloader = train.build_dataloader(CONFIG, split="train", augment=False)
    pos_weight = train.compute_pos_weight(dataloader.dataset)

    import functools

    step_fn = functools.partial(
        train.real_train_step, backbone=backbone, branch="full", pos_weight=pos_weight
    )

    final_step = train.train(
        model,
        dataloader,
        optimizer,
        epochs=1,
        checkpoint_dir=tmp_path,
        checkpoint_every_steps=2,
        device="cpu",
        max_steps=3,
        step_fn=step_fn,
    )

    assert final_step == 3
    assert train.find_latest_checkpoint(tmp_path) is not None
