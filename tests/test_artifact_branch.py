import pytest
import torch

from models.artifact_branch import ArtifactBranch


def test_artifact_branch_returns_fixed_size_embedding_for_variable_images():
    branch = ArtifactBranch(out_channels=32)

    small = branch(torch.randn(2, 3, 32, 32))
    wide = branch(torch.randn(2, 3, 48, 64))

    assert small.shape == (2, 32)
    assert wide.shape == (2, 32)
    assert torch.isfinite(small).all()
    assert torch.isfinite(wide).all()


def test_srm_filters_are_fixed_and_cnn_is_trainable():
    branch = ArtifactBranch()

    assert branch.srm_kernels.requires_grad is False
    assert any(parameter.requires_grad for parameter in branch.encoder.parameters())


def test_artifact_branch_supports_backpropagation():
    branch = ArtifactBranch(out_channels=8)
    images = torch.randn(2, 3, 16, 16, requires_grad=True)

    branch(images).sum().backward()

    assert images.grad is not None
    assert all(
        parameter.grad is not None for parameter in branch.encoder.parameters()
    )


def test_embed_matches_forward():
    branch = ArtifactBranch()
    images = torch.randn(1, 3, 16, 16)

    assert torch.equal(branch(images), branch.embed(images))


@pytest.mark.parametrize(
    "images",
    [torch.randn(2, 1, 16, 16), torch.randn(2, 3, 2, 16), torch.randn(3, 16, 16)],
)
def test_artifact_branch_rejects_invalid_input(images):
    with pytest.raises(ValueError):
        ArtifactBranch()(images)