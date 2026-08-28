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
