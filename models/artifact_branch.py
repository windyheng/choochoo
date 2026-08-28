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
import torch.nn.functional as F
from torch import nn


class ArtifactBranch(nn.Module):
    def __init__(self, out_channels: int = 32):
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        kernels = torch.tensor(
            [
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("srm_kernels", kernels.repeat(3, 1, 1, 1))

        self.encoder = nn.Sequential(
            nn.Conv2d(15, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] -> [B, D2]. See docs/interfaces.md for D2."""
        return self.embed(images)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Return SRM artifact embeddings for raw RGB images."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if images.shape[2] < 3 or images.shape[3] < 3:
            raise ValueError("images must be at least 3x3 pixels")

        residuals = F.conv2d(images, self.srm_kernels, padding=1, groups=3)
        return self.encoder(residuals).flatten(1)
