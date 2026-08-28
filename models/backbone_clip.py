"""Owner: CLIP Branch Lead.

Frozen CLIP ViT-B/16 image feature extractor. This is the "semantic signal"
branch: robust to JPEG/blur/resize and generalizing across generator families,
where the SRM artifact branch is precise but fragile (see the design doc §2).

Contract with the fusion head (docs/interfaces.md §1):
- `embed(images: Tensor[B, 3, 224, 224]) -> Tensor[B, 512]`, float32
- input images are float32 in [0, 1], channel-first, and NOT normalized by the
  caller — that's exactly what data/dataset.py emits. CLIP's own mean/std is
  applied inside embed().
- returned embeddings are L2-normalized along dim=-1 (l2_normalize=True default).

The backbone is frozen, so a given image always maps to the same embedding.
Precompute and cache embeddings for the non-augmented training data rather than
recomputing every epoch (that's the embedding-cache pipeline, built next).
"""

from __future__ import annotations

import torch
from torch import nn

try:
    import open_clip
    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
except ImportError:  # keep `import models.backbone_clip` cheap where the dep is absent
    open_clip = None
    # OpenAI CLIP preprocessing constants (open_clip.constants) — fallback only.
    OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

# Projected image-embedding width per open_clip model name. Used for a documented
# sanity check; the real value is read from the loaded tower at runtime.
CLIP_EMBED_DIMS = {"ViT-B-16": 512, "ViT-B-32": 512, "ViT-L-14": 768}


class CLIPBackbone(nn.Module):
    """Wraps the open_clip image tower as a fixed feature extractor.

    Only the visual tower is kept — the text encoder is unused for this task and
    is dropped to save Colab memory.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze: bool = True,
        l2_normalize: bool = True,
    ):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "open_clip is required for CLIPBackbone — `pip install open_clip_torch`."
            )
        self.model_name = model_name
        self.l2_normalize = l2_normalize
        self.frozen = freeze

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        # Image-only feature extractor: `model.visual(x)` == `encode_image(x, normalize=False)`.
        self.visual = model.visual

        self.register_buffer("mean", torch.tensor(OPENAI_DATASET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(OPENAI_DATASET_STD).view(1, 3, 1, 1))

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad_(False)
            self.visual.eval()

    @classmethod
    def from_config(cls, config: dict) -> "CLIPBackbone":
        """Build from a parsed configs/train.yaml dict (`model.*` keys)."""
        model_cfg = config["model"]
        return cls(
            model_name=model_cfg.get("clip_model", "ViT-B-16"),
            pretrained=model_cfg.get("clip_pretrained", "openai"),
            freeze=model_cfg.get("freeze_clip", True),
        )

    def train(self, mode: bool = True):
        """Keep a frozen tower in eval() no matter what the parent module does,
        so an enclosing model.train() can't re-enable dropout in the backbone."""
        super().train(mode)
        if self.frozen:
            self.visual.eval()
        return self

    @property
    def embed_dim(self) -> int:
        return CLIP_EMBED_DIMS.get(self.model_name) or int(self.visual.output_dim)

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, 224, 224], float32 in [0, 1], unnormalized.

        Returns [B, embed_dim] float32, L2-normalized when l2_normalize is set.
        """
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images shaped [B, 3, H, W], got {tuple(images.shape)}")

        images = images.to(device=self.mean.device, dtype=torch.float32)
        images = (images - self.mean) / self.std
        features = self.visual(images)

        if self.l2_normalize:
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.float()
