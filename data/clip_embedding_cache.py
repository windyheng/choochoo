"""Owner: CLIP Branch Lead.

Read side of the CLIP embedding cache. The CLIP backbone is frozen, so a clean
image always maps to the same embedding — `data/cache_clip_embeddings.py`
precomputes them once per split and this loads them back for training / eval.

Cache file: `data/cache/clip_embeddings/<split>.npz`
  - `paths`      : (N,) unicode array — the image_path strings exactly as they
                   appear in the split CSV (lookup is exact-match on these)
  - `embeddings` : (N, D) float16, L2-normalized (matches
                   `CLIPBackbone(l2_normalize=True)`)

Usage in a training Dataset — the ~50% of samples that get augmented each epoch
change their pixels, so their embedding is NOT cacheable; compute those live
against a `CLIPBackbone`:

    cache = CLIPEmbeddingCache.load("data/cache/clip_embeddings", "train")
    ...
    if augmented or image_path not in cache:
        emb = backbone.embed(image_tensor[None])[0]
    else:
        emb = cache.get(image_path)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class CLIPEmbeddingCache:
    def __init__(self, paths, embeddings: np.ndarray):
        paths = list(paths)
        if len(paths) != len(embeddings):
            raise ValueError(
                f"paths ({len(paths)}) and embeddings ({len(embeddings)}) length mismatch"
            )
        self._embeddings = np.asarray(embeddings)
        self._index = {str(p): i for i, p in enumerate(paths)}

    @staticmethod
    def npz_path(cache_dir, split: str) -> Path:
        return Path(cache_dir) / f"{split}.npz"

    @classmethod
    def load(cls, cache_dir, split: str) -> "CLIPEmbeddingCache":
        path = cls.npz_path(cache_dir, split)
        if not path.exists():
            raise FileNotFoundError(
                f"no embedding cache at {path} — run `python data/cache_clip_embeddings.py`"
            )
        with np.load(path, allow_pickle=False) as data:
            return cls(data["paths"], data["embeddings"])

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, image_path) -> bool:
        return str(image_path) in self._index

    @property
    def dim(self) -> int:
        return int(self._embeddings.shape[1]) if self._embeddings.ndim == 2 else 0

    def get(self, image_path) -> torch.Tensor:
        """Cached embedding as a [D] float32 tensor. Raises KeyError if absent."""
        try:
            row = self._index[str(image_path)]
        except KeyError:
            raise KeyError(f"{image_path!r} not in embedding cache") from None
        return torch.from_numpy(self._embeddings[row].astype(np.float32))
