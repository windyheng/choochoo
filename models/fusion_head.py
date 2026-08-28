"""Owner: Training/Infra Lead (assembles Day 2, once CLIP + SRM branches land).

Concatenates the CLIP branch embedding and SRM branch embedding, feeds
through a small trainable MLP -> P(AIGC). Concatenation order is
[clip_embed, srm_embed] — matching train.py's DummyFusionHead, which already
committed to this order (see docs/interfaces.md section 3) — keep it
consistent with explainability.py and error_analysis.py.

Implemented ahead of the Training/Infra Lead to unblock infer.py/evaluate.py,
which need a real (non-Dummy) fusion head to run actual inference. Shape is a
straight port of DummyFusionHead's architecture (Linear -> ReLU -> Linear),
since that's already the shape train.py's checkpoint/resume loop and
configs/train.yaml's fusion_hidden_dim were built against — swap the
architecture here if the Training/Infra Lead wants something different, the
forward() contract (concat order, logits[B] output) is what everything else
depends on.
"""

import torch
from torch import nn


class FusionHead(nn.Module):
    def __init__(self, clip_dim: int, srm_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim + srm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_embed: torch.Tensor, srm_embed: torch.Tensor) -> torch.Tensor:
        """Returns logits, shape [B]. Caller applies sigmoid for probability."""
        if clip_embed.dim() != 2 or srm_embed.dim() != 2:
            raise ValueError(
                f"expected [B, D] embeddings, got clip_embed {tuple(clip_embed.shape)}, "
                f"srm_embed {tuple(srm_embed.shape)}"
            )
        if clip_embed.shape[0] != srm_embed.shape[0]:
            raise ValueError(
                f"batch size mismatch: clip_embed {clip_embed.shape[0]} vs srm_embed {srm_embed.shape[0]}"
            )
        x = torch.cat([clip_embed, srm_embed], dim=1)  # concat order per docs/interfaces.md section 3
        return self.net(x).squeeze(-1)
