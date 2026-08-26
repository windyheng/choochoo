"""Owner: CLIP Branch Lead.

Frozen CLIP ViT-B/16 feature extractor. See docs/interfaces.md section 1 for
the exact output shape/dtype/normalization contract the fusion head expects
— fill that in as you implement this, don't leave it as TODO once decided.

Precompute and cache embeddings for the non-augmented portion of training
data (backbone is frozen, no reason to recompute every epoch).
"""

import torch
from torch import nn


class CLIPBackbone(nn.Module):
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai"):
        super().__init__()
        raise NotImplementedError

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] -> [B, D]. See docs/interfaces.md for D."""
        raise NotImplementedError
