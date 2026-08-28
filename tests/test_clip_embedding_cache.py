"""Owner: CLIP Branch Lead. Tests for the CLIP embedding cache (build + load).

The fast tests use a FakeBackbone — no model download. One `slow` test exercises
the real CLIPBackbone end to end.
"""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

from data import cache_clip_embeddings as cce
from data.clip_embedding_cache import CLIPEmbeddingCache


class FakeBackbone:
    """Deterministic [B, 8] embedding from the per-image pixel mean. Matches the
    CLIPBackbone surface build_split_cache uses: .to(), .eval(), .embed()."""

    def to(self, device):
        return self

    def eval(self):
        return self

    def embed(self, batch: "torch.Tensor") -> "torch.Tensor":
        flat = batch.reshape(batch.shape[0], -1)
        v = flat.mean(dim=1, keepdim=True).repeat(1, 8) + torch.arange(8).float()
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@pytest.fixture
def tiny_split(tmp_path):
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    paths = []
    for i in range(3):
        p = img_dir / f"{i}.png"
        Image.new("RGB", (32, 32), color=(i * 50, i * 50, i * 50)).save(p)
        paths.append(str(p))
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "label", "source"])
        for p in paths:
            w.writerow([p, 0, "fake"])
    return csv_path, paths


def test_build_and_load(tmp_path, tiny_split):
    csv_path, paths = tiny_split
    cache_dir = tmp_path / "cache"
    cce.build_split_cache(csv_path, cache_dir / "train.npz", FakeBackbone(), batch_size=2)

    cache = CLIPEmbeddingCache.load(cache_dir, "train")
    assert len(cache) == 3
    assert cache.dim == 8
    for p in paths:
        assert p in cache
        e = cache.get(p)
        assert e.shape == (8,)
        assert e.dtype == torch.float32
    assert "/not/here.png" not in cache


def test_embeddings_stored_as_float16(tmp_path, tiny_split):
    csv_path, _ = tiny_split
    cache_dir = tmp_path / "cache"
    cce.build_split_cache(csv_path, cache_dir / "train.npz", FakeBackbone(), batch_size=8)
    with np.load(cache_dir / "train.npz", allow_pickle=False) as data:
        assert data["embeddings"].dtype == np.float16
        assert data["embeddings"].shape == (3, 8)


def test_resume_is_noop_when_complete(tmp_path, tiny_split, capsys):
    csv_path, _ = tiny_split
    npz = tmp_path / "cache" / "train.npz"
    cce.build_split_cache(csv_path, npz, FakeBackbone(), batch_size=2)
    capsys.readouterr()
    cce.build_split_cache(csv_path, npz, FakeBackbone(), batch_size=2)
    assert "0 to compute" in capsys.readouterr().out


def test_partial_resume_completes_cache(tmp_path, tiny_split):
    csv_path, paths = tiny_split
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    npz = cache_dir / "train.npz"

    fb = FakeBackbone()
    first = torch.from_numpy(cce.load_image(paths[0])[None])
    e0 = fb.embed(first).numpy().astype(np.float16)
    np.savez(npz, paths=np.array([paths[0]], dtype=np.str_), embeddings=e0)

    cce.build_split_cache(csv_path, npz, fb, batch_size=2)

    cache = CLIPEmbeddingCache.load(cache_dir, "train")
    assert len(cache) == 3
    assert torch.allclose(cache.get(paths[0]), torch.from_numpy(e0[0].astype(np.float32)))


def test_unreadable_images_are_skipped(tmp_path, tiny_split, capsys):
    csv_path, paths = tiny_split
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([str(tmp_path / "missing.png"), 0, "fake"])

    npz = tmp_path / "cache" / "train.npz"
    cce.build_split_cache(csv_path, npz, FakeBackbone(), batch_size=2)

    cache = CLIPEmbeddingCache.load(tmp_path / "cache", "train")
    assert len(cache) == 3
    assert "skipped 1 unreadable" in capsys.readouterr().out


def test_get_missing_key_raises(tmp_path, tiny_split):
    csv_path, _ = tiny_split
    npz = tmp_path / "cache" / "train.npz"
    cce.build_split_cache(csv_path, npz, FakeBackbone(), batch_size=2)
    cache = CLIPEmbeddingCache.load(tmp_path / "cache", "train")
    with pytest.raises(KeyError):
        cache.get("/nonexistent.png")


def test_load_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CLIPEmbeddingCache.load(tmp_path, "train")


def test_cli_runs_as_script_from_any_cwd(tmp_path):
    """`python data/cache_clip_embeddings.py --help` must work regardless of CWD
    (regression: importing data.* / models.* when run as a script)."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "data" / "cache_clip_embeddings.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


@pytest.mark.slow
def test_end_to_end_real_backbone(tmp_path, tiny_split):
    pytest.importorskip("open_clip")
    from models.backbone_clip import CLIPBackbone

    csv_path, paths = tiny_split
    cache_dir = tmp_path / "cache"
    cce.build_split_cache(csv_path, cache_dir / "val.npz", CLIPBackbone(), batch_size=2)

    cache = CLIPEmbeddingCache.load(cache_dir, "val")
    assert cache.dim == 512
    e = cache.get(paths[0])
    assert e.shape == (512,)
    assert abs(e.norm().item() - 1.0) < 1e-2  # L2-normalized, survived fp16 round-trip
