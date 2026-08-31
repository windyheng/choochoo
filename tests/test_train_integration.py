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


# --- CLIP embedding cache in the training path ---


def _mini_train_csv(tmp_path, n=8):
    """A split CSV of the first n real train rows (repo-relative paths intact)."""
    import csv

    src = REPO_ROOT / "data" / "cache" / "splits" / "train.csv"
    rows = list(csv.DictReader(src.open(newline="")))[:n]
    dst = tmp_path / "train.csv"
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return dst, [r["image_path"] for r in rows]


def test_resolve_clip_embeddings_cache_matches_live(tmp_path, backbone):
    """A cache hit must return the same embedding the frozen backbone would
    compute live (within fp16 storage tolerance)."""
    from torch.utils.data import DataLoader

    from data import cache_clip_embeddings as cce
    from data.clip_embedding_cache import CLIPEmbeddingCache
    from data.dataset import AIGCDataset

    csv_path, _ = _mini_train_csv(tmp_path)
    cce.build_split_cache(csv_path, tmp_path / "cache" / "train.npz", backbone, batch_size=4)
    cache = CLIPEmbeddingCache.load(tmp_path / "cache", "train")

    loader = DataLoader(AIGCDataset(csv_path, augment=False, report_augmented=True), batch_size=8)
    images, _, paths, augmented = next(iter(loader))

    cached = train._resolve_clip_embeddings(
        images, paths, augmented, backbone=backbone, clip_cache=cache, device="cpu"
    )
    live = backbone.embed(images)
    assert torch.allclose(cached, live, atol=2e-3)


def test_resolve_clip_embeddings_only_embeds_augmented_rows_live(tmp_path, backbone, monkeypatch):
    """Augmented rows go through backbone.embed; cached clean rows do not."""
    from torch.utils.data import DataLoader

    from data import cache_clip_embeddings as cce
    from data.clip_embedding_cache import CLIPEmbeddingCache
    from data.dataset import AIGCDataset

    csv_path, _ = _mini_train_csv(tmp_path, n=4)
    cce.build_split_cache(csv_path, tmp_path / "cache" / "train.npz", backbone, batch_size=4)
    cache = CLIPEmbeddingCache.load(tmp_path / "cache", "train")

    loader = DataLoader(AIGCDataset(csv_path, augment=False, report_augmented=True), batch_size=4)
    images, _, paths, _ = next(iter(loader))
    forced_aug = torch.tensor([True, False, True, False])

    live_batch_sizes = []
    real_embed = backbone.embed
    monkeypatch.setattr(
        backbone, "embed", lambda imgs: (live_batch_sizes.append(imgs.shape[0]), real_embed(imgs))[1]
    )
    out = train._resolve_clip_embeddings(
        images, list(paths), forced_aug, backbone=backbone, clip_cache=cache, device="cpu"
    )

    assert live_batch_sizes == [2]           # only the 2 augmented rows embedded live
    assert out.shape == (4, 512)
    # the 2 cached rows still round-trip close to a fresh live embed
    monkeypatch.undo()
    live = backbone.embed(images)
    assert torch.allclose(out[1], live[1], atol=3e-3)
    assert torch.allclose(out[3], live[3], atol=3e-3)


def test_clip_only_train_step_with_cache_trains_fusion_head(tmp_path, backbone):
    from torch.utils.data import DataLoader

    from data import cache_clip_embeddings as cce
    from data.clip_embedding_cache import CLIPEmbeddingCache
    from data.dataset import AIGCDataset

    csv_path, _ = _mini_train_csv(tmp_path)
    cce.build_split_cache(csv_path, tmp_path / "cache" / "train.npz", backbone, batch_size=4)
    cache = CLIPEmbeddingCache.load(tmp_path / "cache", "train")

    model = train.build_model(CONFIG, branch="clip_only")
    model.train()
    optimizer = train.build_optimizer(model, CONFIG)
    loader = DataLoader(
        AIGCDataset(csv_path, augment=False, report_augmented=True), batch_size=8
    )
    batch = next(iter(loader))

    import copy

    before = copy.deepcopy(model.state_dict())
    loss = train.real_train_step(
        model, optimizer, batch, "cpu", backbone=backbone, branch="clip_only", clip_cache=cache
    )
    after = model.state_dict()

    assert isinstance(loss, float) and loss == loss
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert any(k.startswith("fusion_head.") for k in changed)
