"""Owner: Training/Infra Lead (assembles Day 2, once CLIP + SRM branches land).

Concatenates the CLIP branch embedding and SRM branch embedding, feeds
through a small trainable MLP -> P(AIGC). See docs/interfaces.md section 3
for the agreed concatenation order — keep it consistent with
explainability.py and error_analysis.py.
"""

import torch
from torch import nn


class FusionHead(nn.Module):
    def __init__(self, clip_dim: int, srm_dim: int, hidden_dim: int = 256):
        super().__init__()
        raise NotImplementedError

    def forward(self, clip_embed: torch.Tensor, srm_embed: torch.Tensor) -> torch.Tensor:
        """Returns logits, shape [B]. Caller applies sigmoid for probability."""
        raise NotImplementedError
