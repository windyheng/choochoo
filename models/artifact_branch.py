"""Owner: SRM Branch Lead.

Fixed SRM (Spatial Rich Model) high-pass filter bank -> shallow trainable
CNN. Cheap, strong on clean/lightly-transformed images; expected to degrade
under blur/JPEG/resize (that degradation is intentional signal for the
error-analysis writeup, not a bug to hide).

Can be prototyped against any small labeled image sample before the full
data pipeline (data/prepare_datasets.py) is ready — not blocked on Data Lead.

See docs/interfaces.md section 2 for the exact output shape/dtype contract
the fusion head expects.
"""

import torch
from torch import nn


class ArtifactBranch(nn.Module):
    def __init__(self, out_channels: int = 32):
        super().__init__()
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] -> [B, D2]. See docs/interfaces.md for D2."""
        raise NotImplementedError
