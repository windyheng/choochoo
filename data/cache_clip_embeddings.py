"""Owner: CLIP Branch Lead.

Precompute CLIP embeddings for the CLEAN (non-augmented) images in each split.
The backbone is frozen, so this runs once and is reused every epoch + for all
eval/ablation. The ~50% of training samples that get augmented change their
pixels (and thus their embedding) — those are computed live during training,
not cached (see docs/specs design doc §3 and data/clip_embedding_cache.py).

Output: data/cache/clip_embeddings/<split>.npz  — `paths` + `embeddings`
[N, 512] float16, L2-normalized.

Resumable: safe to re-run after a Colab disconnect. Already-cached paths are
skipped and the .npz is rewritten atomically every --flush-every batches.
Unreadable images are skipped with a warning, not fatal.

Run: python data/cache_clip_embeddings.py --config configs/train.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
# Allow `python data/cache_clip_embeddings.py ...` (script run from inside data/,
# so the repo root isn't on sys.path by default).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import CANONICAL_SIZE
from data.paths import resolve_image_path
from models.backbone_clip import CLIPBackbone
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache" / "clip_embeddings"


def read_split_paths(split_csv: Path) -> list[str]:
    with open(split_csv, newline="") as f:
        return [row["image_path"] for row in csv.DictReader(f)]


def load_image(image_path: str) -> np.ndarray:
    """Same preprocessing as data.dataset.AIGCDataset, minus augmentation:
    RGB, resized to CANONICAL_SIZE, float32 CHW in [0, 1]. image_path is
    REPO_ROOT-relative (see data/paths.py); resolved before opening."""
    with Image.open(resolve_image_path(image_path)) as im:
        im = im.convert("RGB").resize(CANONICAL_SIZE, Image.BILINEAR)
    return (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)


def _load_existing(npz_path: Path) -> tuple[list[str], list[np.ndarray]]:
    if not npz_path.exists():
        return [], []
    with np.load(npz_path, allow_pickle=False) as data:
        return [str(p) for p in data["paths"]], list(data["embeddings"])


def _atomic_save(npz_path: Path, paths: list[str], embeddings: list[np.ndarray]) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.parent / (npz_path.stem + ".tmp.npz")
    stacked = (
        np.stack(embeddings).astype(np.float16)
        if embeddings
        else np.empty((0, 0), dtype=np.float16)
    )
    np.savez(tmp, paths=np.array(paths, dtype=np.str_), embeddings=stacked)
    tmp.replace(npz_path)


def build_split_cache(
    split_csv: Path,
    npz_path: Path,
    backbone,
    *,
    batch_size: int = 256,
    device: str = "cpu",
    flush_every: int = 20,
) -> None:
    all_paths = read_split_paths(Path(split_csv))
    done_paths, done_embeds = _load_existing(Path(npz_path))
    done_set = set(done_paths)
    todo = [p for p in all_paths if p not in done_set]

    print(f"[{Path(split_csv).stem}] {len(all_paths)} images, {len(todo)} to compute, {len(done_set)} cached")
    if not todo:
        return

    backbone = backbone.to(device).eval()
    paths_out, embeds_out = list(done_paths), list(done_embeds)
    skipped: list[str] = []
    batches_since_flush = 0

    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        arrs, kept = [], []
        for p in chunk:
            try:
                arrs.append(load_image(p))
                kept.append(p)
            except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError):
                skipped.append(p)
        if not kept:
            continue

        batch = torch.from_numpy(np.stack(arrs)).to(device)
        emb = backbone.embed(batch).detach().cpu().numpy().astype(np.float16)
        paths_out.extend(kept)
        embeds_out.extend(emb)

        batches_since_flush += 1
        print(f"  {len(paths_out)}/{len(all_paths)}")
        if batches_since_flush >= flush_every:
            _atomic_save(Path(npz_path), paths_out, embeds_out)
            batches_since_flush = 0

    _atomic_save(Path(npz_path), paths_out, embeds_out)
    size_mb = Path(npz_path).stat().st_size / 1e6
    print(f"[{Path(split_csv).stem}] wrote {npz_path} ({len(paths_out)} embeddings, {size_mb:.1f} MB)")
    if skipped:
        print(f"[{Path(split_csv).stem}] WARNING: skipped {len(skipped)} unreadable images, e.g. {skipped[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    splits_dir = REPO_ROOT / config["data"]["splits_dir"]
    cache_dir = Path(args.cache_dir)

    backbone = CLIPBackbone.from_config(config)

    for split in args.splits:
        split_csv = splits_dir / f"{split}.csv"
        if not split_csv.exists():
            print(f"[{split}] {split_csv} not found, skipping")
            continue
        build_split_cache(
            split_csv,
            cache_dir / f"{split}.npz",
            backbone,
            batch_size=args.batch_size,
            device=args.device,
            flush_every=args.flush_every,
        )


if __name__ == "__main__":
    main()
