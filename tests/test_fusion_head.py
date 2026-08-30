import pytest
import torch

from models.fusion_head import FusionHead


def test_fusion_head_returns_logit_per_sample():
    head = FusionHead(clip_dim=512, srm_dim=32, hidden_dim=16)
    clip_embed = torch.randn(4, 512)
    srm_embed = torch.randn(4, 32)

    logits = head(clip_embed, srm_embed)

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_fusion_head_concatenates_clip_then_srm():
    head = FusionHead(clip_dim=2, srm_dim=3, hidden_dim=8)
    # zero the srm branch's contribution: srm inputs identical across samples,
    # any difference in output must come from the clip half of the concat.
    clip_embed = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    srm_embed = torch.zeros(2, 3)

    logits = head(clip_embed, srm_embed)

    assert logits[0].item() != logits[1].item()


def test_fusion_head_supports_backpropagation():
    head = FusionHead(clip_dim=8, srm_dim=4, hidden_dim=16)
    clip_embed = torch.randn(3, 8, requires_grad=True)
    srm_embed = torch.randn(3, 4, requires_grad=True)

    head(clip_embed, srm_embed).sum().backward()

    assert clip_embed.grad is not None
    assert srm_embed.grad is not None
    assert all(p.grad is not None for p in head.parameters())


def test_fusion_head_rejects_batch_size_mismatch():
    head = FusionHead(clip_dim=4, srm_dim=4)
    with pytest.raises(ValueError):
        head(torch.randn(2, 4), torch.randn(3, 4))


@pytest.mark.parametrize(
    "clip_embed,srm_embed",
    [(torch.randn(4), torch.randn(2, 4)), (torch.randn(2, 4), torch.randn(4))],
)
def test_fusion_head_rejects_non_2d_input(clip_embed, srm_embed):
    head = FusionHead(clip_dim=4, srm_dim=4)
    with pytest.raises(ValueError):
        head(clip_embed, srm_embed)


def test_fusion_head_rejects_both_dims_zero():
    with pytest.raises(ValueError):
        FusionHead(clip_dim=0, srm_dim=0)


def test_fusion_head_clip_only_ignores_srm_embed():
    head = FusionHead(clip_dim=4, srm_dim=0, hidden_dim=8)
    logits = head(torch.randn(3, 4), None)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_fusion_head_clip_only_rejects_missing_clip_embed():
    head = FusionHead(clip_dim=4, srm_dim=0)
    with pytest.raises(ValueError):
        head(None, None)


def test_fusion_head_artifact_only_ignores_clip_embed():
    head = FusionHead(clip_dim=0, srm_dim=4, hidden_dim=8)
    logits = head(None, torch.randn(3, 4))
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_fusion_head_predict_proba_default_temperature_is_plain_sigmoid():
    head = FusionHead(clip_dim=8, srm_dim=4, hidden_dim=16)
    head.eval()
    clip_embed = torch.randn(5, 8)
    srm_embed = torch.randn(5, 4)

    with torch.no_grad():
        logits = head(clip_embed, srm_embed)
        proba = head.predict_proba(clip_embed, srm_embed)

    assert torch.allclose(proba, torch.sigmoid(logits))
    assert torch.all((proba > 0) & (proba < 1))


def test_fusion_head_predict_proba_scales_with_temperature():
    head = FusionHead(clip_dim=8, srm_dim=4, hidden_dim=16)
    head.eval()
    clip_embed = torch.randn(5, 8)
    srm_embed = torch.randn(5, 4)

    with torch.no_grad():
        default_proba = head.predict_proba(clip_embed, srm_embed)
        head.temperature.fill_(2.0)
        scaled_proba = head.predict_proba(clip_embed, srm_embed)

    assert not torch.allclose(default_proba, scaled_proba)


@pytest.mark.parametrize(
    "branch,expected_clip_dim,expected_srm_dim",
    [("full", 512, 32), ("clip_only", 512, 0), ("artifact_only", 0, 32)],
)
def test_fusion_head_from_config_zeroes_correct_branch(branch, expected_clip_dim, expected_srm_dim):
    config = {"model": {"clip_model": "ViT-B-16", "srm_out_channels": 32, "fusion_hidden_dim": 16}}
    head = FusionHead.from_config(config, branch=branch)
    assert head.clip_dim == expected_clip_dim
    assert head.srm_dim == expected_srm_dim


def test_fusion_head_from_config_rejects_unknown_branch():
    config = {"model": {"clip_model": "ViT-B-16", "srm_out_channels": 32}}
    with pytest.raises(ValueError):
        FusionHead.from_config(config, branch="nonsense")
