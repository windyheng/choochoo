"""Owner: Training/Infra Lead (assembled Day 2, once CLIP + SRM branches landed).

Concatenates the CLIP branch embedding and SRM branch embedding, feeds
through a small trainable MLP -> P(AIGC). Concatenation order is
[clip_embed, srm_embed] — matching train.py's DummyFusionHead, which already
committed to this order (see docs/interfaces.md section 3) — keep it
consistent with explainability.py and error_analysis.py.

Architecture follows docs/specs/2026-08-29-fusion-head-design.md: BatchNorm1d
on the concatenated input (CLIP is a unit vector, SRM is unbounded
non-negative pooled activations — incompatible scales without it), Dropout
before the output layer, and a `temperature` buffer for post-training Platt
scaling (fit elsewhere, e.g. calibrate.py; defaults to 1.0 so an uncalibrated
checkpoint still produces sane output). `clip_dim`/`srm_dim` may be 0 (but not
both) to support CLIP-only / SRM-only ablation runs — the corresponding
forward() argument is then ignored and may be None.
"""

import torch
from torch import nn

from models.backbone_clip import CLIP_EMBED_DIMS

_BRANCH_ZEROS = {
    "full": (False, False),
    "clip_only": (False, True),  # zero out srm_dim
    "artifact_only": (True, False),  # zero out clip_dim
}


class FusionHead(nn.Module):
    def __init__(self, clip_dim: int, srm_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        if clip_dim <= 0 and srm_dim <= 0:
            raise ValueError("at least one of clip_dim/srm_dim must be positive")
        if clip_dim < 0 or srm_dim < 0:
            raise ValueError("clip_dim/srm_dim must be non-negative")
        self.clip_dim = clip_dim
        self.srm_dim = srm_dim

        input_dim = clip_dim + srm_dim
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, clip_embed: torch.Tensor | None, srm_embed: torch.Tensor | None) -> torch.Tensor:
        """Returns logits, shape [B]. Caller applies sigmoid for probability
        (or use predict_proba for the calibrated version).

        clip_embed/srm_embed may be None only for the side whose dim is 0
        (CLIP-only / SRM-only branch configurations)."""
        parts = []
        if self.clip_dim > 0:
            if clip_embed is None or clip_embed.dim() != 2:
                shape = None if clip_embed is None else tuple(clip_embed.shape)
                raise ValueError(f"expected clip_embed shaped [B, {self.clip_dim}], got {shape}")
            parts.append(clip_embed)
        if self.srm_dim > 0:
            if srm_embed is None or srm_embed.dim() != 2:
                shape = None if srm_embed is None else tuple(srm_embed.shape)
                raise ValueError(f"expected srm_embed shaped [B, {self.srm_dim}], got {shape}")
            parts.append(srm_embed)

        if len(parts) == 2 and parts[0].shape[0] != parts[1].shape[0]:
            raise ValueError(
                f"batch size mismatch: clip_embed {parts[0].shape[0]} vs srm_embed {parts[1].shape[0]}"
            )

        # concat order per docs/interfaces.md section 3: [clip_embed, srm_embed]
        x = torch.cat(parts, dim=1) if len(parts) == 2 else parts[0]
        x = self.input_norm(x)
        return self.net(x).squeeze(-1)

    def predict_proba(self, clip_embed: torch.Tensor | None, srm_embed: torch.Tensor | None) -> torch.Tensor:
        """Calibrated probability, shape [B] in (0, 1). Temperature defaults
        to 1.0 (plain sigmoid) until a calibration step sets it."""
        return torch.sigmoid(self.forward(clip_embed, srm_embed) / self.temperature)

    @classmethod
    def from_config(cls, config: dict, branch: str = "full") -> "FusionHead":
        """Build from a parsed configs/train.yaml dict (`model.*` keys) and a
        branch selection: "full" (default), "clip_only", or "artifact_only"."""
        if branch not in _BRANCH_ZEROS:
            raise ValueError(f"unknown branch {branch!r}, expected one of {sorted(_BRANCH_ZEROS)}")
        zero_clip, zero_srm = _BRANCH_ZEROS[branch]

        model_cfg = config["model"]
        clip_dim = 0 if zero_clip else CLIP_EMBED_DIMS[model_cfg.get("clip_model", "ViT-B-16")]
        srm_dim = 0 if zero_srm else model_cfg["srm_out_channels"]
        return cls(clip_dim=clip_dim, srm_dim=srm_dim, hidden_dim=model_cfg.get("fusion_hidden_dim", 256))
