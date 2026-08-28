"""Owner: CLIP Branch Lead. Tests for models/backbone_clip.py.

These load the real CLIP ViT-B/16 checkpoint (~600 MB download on first run),
so they are marked `slow` and skipped unless `pytest --runslow` (see conftest.py).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("open_clip")

from models.backbone_clip import CLIPBackbone

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def backbone():
    return CLIPBackbone(model_name="ViT-B-16", pretrained="openai")


def _fake_batch(n: int = 2, size: int = 224) -> "torch.Tensor":
    """A batch shaped like what data/dataset.py emits: float32, [0, 1], unnormalized."""
    return torch.rand(n, 3, size, size, dtype=torch.float32)


def test_embed_shape_and_dtype(backbone):
    out = backbone.embed(_fake_batch(3))
    assert out.shape == (3, 512)
    assert out.dtype == torch.float32


def test_embed_dim_property(backbone):
    assert backbone.embed_dim == 512


def test_backbone_is_frozen(backbone):
    assert all(not p.requires_grad for p in backbone.parameters())
    assert sum(p.numel() for p in backbone.parameters()) < 200_000_000  # ViT-B/16 visual ~86M


def test_stays_eval_after_train_call(backbone):
    backbone.train()  # simulate an enclosing model.train()
    assert not backbone.visual.training
    backbone.eval()


def test_embed_is_deterministic(backbone):
    batch = _fake_batch(2)
    assert torch.allclose(backbone.embed(batch), backbone.embed(batch))


def test_accepts_unnormalized_zero_one_input(backbone):
    out = backbone.embed(torch.full((1, 3, 224, 224), 0.5))
    assert torch.isfinite(out).all()


def test_l2_normalized_by_default(backbone):
    norms = backbone.embed(_fake_batch(4)).norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_l2_normalize_can_be_disabled():
    raw = CLIPBackbone(l2_normalize=False)
    norms = raw.embed(_fake_batch(3)).norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


def test_rejects_wrong_shape(backbone):
    with pytest.raises(ValueError):
        backbone.embed(torch.rand(3, 224, 224))


def test_small_perturbation_stays_closer_than_a_different_image(backbone):
    """A tiny pixel nudge should move the embedding less than swapping in an
    unrelated image does — the 'robust to mild changes' property in miniature."""
    torch.manual_seed(0)
    img = torch.rand(1, 3, 224, 224)
    nudged = (img + 0.01 * torch.randn_like(img)).clamp(0, 1)
    other = torch.rand(1, 3, 224, 224)
    e_img, e_nudged, e_other = (backbone.embed(x) for x in (img, nudged, other))
    assert (e_img * e_nudged).sum() > (e_img * e_other).sum()


def test_from_config():
    config = {"model": {"clip_model": "ViT-B-16", "clip_pretrained": "openai", "freeze_clip": True}}
    backbone = CLIPBackbone.from_config(config)
    assert backbone.embed_dim == 512
    assert backbone.frozen
