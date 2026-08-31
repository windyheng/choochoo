"""Owner: Training/Infra Lead.

Training loop with checkpoint/resume — REQUIRED for free-tier Colab, which
disconnects at the ~12hr session cap and sometimes earlier. Must detect and
resume from the latest checkpoint under configs' checkpoint_dir on restart,
not assume a single uninterrupted run. Build this before the real training
run, not after a disconnect eats progress.

build_model()/build_dataloader() assemble the real pipeline: models/backbone_clip.py
(frozen CLIP, held outside the checkpointed model), models/artifact_branch.py
and models/fusion_head.py (trainable, wrapped in an nn.ModuleDict). The
DummyFusionHead / DummyEmbeddingDataset classes below remain only as fast
scaffolding for tests/test_train.py's checkpoint/resume mechanism tests,
exercised via train_step()/train()'s default step_fn — real runs go through
real_train_step() instead (see main()).

Run: python train.py --config configs/train.yaml [--branch full|clip_only|artifact_only] [--max_steps N]
"""

import argparse
import functools
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.clip_embedding_cache import CLIPEmbeddingCache
from data.dataset import AIGCDataset
from data.paths import REPO_ROOT
from models.artifact_branch import ArtifactBranch
from models.backbone_clip import CLIPBackbone
from models.fusion_head import FusionHead

# Placeholder dims, used only by the Dummy* scaffolding below (the generic
# checkpoint/resume mechanism tests in tests/test_train.py exercise these,
# not the real model/dims).
DUMMY_CLIP_DIM = 512
DUMMY_SRM_DIM = 64
DUMMY_DATASET_SIZE = 512

CHECKPOINT_RE = re.compile(r"ckpt_step_(\d+)\.pt$")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restores RNG state saved by get_rng_state — only affects exact
    reproducibility of the random augmentation sequence across a resume, not
    training correctness. A checkpoint resumed on a different torch/numpy
    version than it was saved on (e.g. trained on Colab, resumed on Kaggle)
    can fail to deserialize here; that's a real cross-environment scenario,
    not a hypothetical, so it's logged and skipped rather than crashing the
    whole resume over a non-essential reproducibility guarantee."""
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except (TypeError, ValueError, RuntimeError) as e:
        print(f"warning: could not restore RNG state ({e!r}) — resuming with a fresh RNG state instead")


def checkpoint_path_for_step(checkpoint_dir, global_step: int) -> Path:
    return Path(checkpoint_dir) / f"ckpt_step_{global_step:08d}.pt"


def save_checkpoint(path, model, optimizer, *, epoch: int, global_step: int) -> None:
    """Atomic write (save to .tmp then rename) — a disconnect mid-write must
    never leave a corrupt .pt file that find_latest_checkpoint would pick up
    and load_checkpoint would then crash on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "rng_state": get_rng_state(),
        },
        tmp_path,
    )
    tmp_path.replace(path)


def load_checkpoint(path, model, optimizer=None, device="cpu") -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    set_rng_state(checkpoint["rng_state"])
    return {"epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"]}


def find_latest_checkpoint(checkpoint_dir):
    """Picks the checkpoint with the highest step number parsed from its
    filename — NOT by mtime, since a Drive-mounted checkpoint_dir can have
    unreliable mtimes across VM restarts. Returns None if checkpoint_dir is
    missing or contains no matching files."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    candidates = []
    for p in checkpoint_dir.iterdir():
        match = CHECKPOINT_RE.search(p.name)
        if match:
            candidates.append((int(match.group(1)), p))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


class DummyEmbeddingDataset(Dataset):
    """Placeholder standing in for the real CLIP+SRM embedding pipeline.
    Emits (clip_embed, srm_embed, label) tuples matching the shape
    FusionHead will consume, so swapping in real embeddings later only
    changes build_dataloader()'s body."""

    def __init__(self, n_samples, clip_dim, srm_dim, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.clip_embeds = torch.randn(n_samples, clip_dim, generator=generator)
        self.srm_embeds = torch.randn(n_samples, srm_dim, generator=generator)
        self.labels = torch.randint(0, 2, (n_samples,), generator=generator).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.clip_embeds[idx], self.srm_embeds[idx], self.labels[idx]


def build_dataloader(config, split: str, augment: bool, use_meta: bool = False) -> DataLoader:
    splits_dir = REPO_ROOT / config["data"]["splits_dir"]
    dataset = AIGCDataset(
        splits_dir / f"{split}.csv", augment=augment, report_augmented=use_meta
    )
    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=config["data"]["num_workers"],
        # FusionHead's BatchNorm1d errors on a batch of size 1 in train mode;
        # drop_last also silently drops a small tail batch every epoch
        # (e.g. 90598 % 64 = 6 samples on the full train split) — a
        # negligible, intentional trade-off, not a bug.
        drop_last=True,
    )


class DummyFusionHead(nn.Module):
    """Lightweight stand-in for models.fusion_head.FusionHead, kept around so
    tests/test_train.py can exercise the generic checkpoint/resume mechanism
    against tiny tensors without paying for BatchNorm1d's batch-size
    constraints or the real branches. Same forward(clip_embed, srm_embed) ->
    logits[B] contract as the real FusionHead."""

    def __init__(self, clip_dim, srm_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clip_dim + srm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clip_embed, srm_embed):
        x = torch.cat([clip_embed, srm_embed], dim=1)  # concat order per docs/interfaces.md section 3
        return self.net(x).squeeze(-1)


def build_model(config, branch: str = "full") -> nn.ModuleDict:
    """Assembles the trainable half of the pipeline: the SRM artifact branch
    (omitted for a clip_only ablation run) and the fusion head. The frozen
    CLIP backbone is built separately (see main()) and deliberately kept out
    of this module tree — it's ~150M deterministic, re-downloadable params
    that don't belong in every checkpoint written to a Drive-mounted
    checkpoint_dir."""
    modules = {}
    if branch != "clip_only":
        modules["artifact_branch"] = ArtifactBranch(out_channels=config["model"]["srm_out_channels"])
    modules["fusion_head"] = FusionHead.from_config(config, branch=branch)
    return nn.ModuleDict(modules)


def build_optimizer(model, config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )


def train_step(model, optimizer, batch, device) -> float:
    clip_embed, srm_embed, labels = batch
    clip_embed = clip_embed.to(device)
    srm_embed = srm_embed.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(clip_embed, srm_embed)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def compute_pos_weight(dataset: AIGCDataset) -> torch.Tensor:
    """n_neg / n_pos over dataset.samples, for BCEWithLogitsLoss's pos_weight
    — CIFAKE is 50/50 but SID_Set/WildFake may not be (see
    docs/specs/2026-08-29-fusion-head-design.md section 4)."""
    labels = [label for _, label in dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)


def _resolve_clip_embeddings(images, paths, augmented, *, backbone, clip_cache, device):
    """CLIP embeddings for a batch: a cache lookup for clean samples whose
    path is cached, a live (frozen) backbone.embed for the augmented ones and
    cache misses. Falls back to a full live embed when no cache / no aug flags."""
    if clip_cache is None or augmented is None:
        return backbone.embed(images)
    n = images.shape[0]
    out = [None] * n
    live_idx = []
    for i in range(n):
        if bool(augmented[i]) or paths[i] not in clip_cache:
            live_idx.append(i)
        else:
            out[i] = clip_cache.get(paths[i]).to(device)
    if live_idx:
        live = backbone.embed(images[live_idx])
        for j, i in enumerate(live_idx):
            out[i] = live[j]
    return torch.stack(out)


def real_train_step(model, optimizer, batch, device, *, backbone, branch, pos_weight=None, clip_cache=None) -> float:
    """train_step for the real assembled pipeline: batch is a raw
    (images, labels, image_paths) triple from data.dataset.AIGCDataset, model
    is the nn.ModuleDict from build_model(), backbone is the frozen
    CLIPBackbone built separately in main() (None for an artifact_only run).

    CLIP embeddings are computed with no_grad inside CLIPBackbone.embed
    (frozen); SRM embeddings are computed with grad enabled so
    model["artifact_branch"]'s trainable CNN actually trains."""
    images, labels, paths, *rest = batch
    augmented = rest[0] if rest else None
    images = images.to(device)
    labels = labels.to(device).float()  # AIGCDataset yields plain int labels; BCEWithLogitsLoss needs float
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    optimizer.zero_grad()
    clip_embed = (
        _resolve_clip_embeddings(
            images, paths, augmented, backbone=backbone, clip_cache=clip_cache, device=device
        )
        if branch != "artifact_only"
        else None
    )
    srm_embed = model["artifact_branch"].embed(images) if branch != "clip_only" else None
    logits = model["fusion_head"](clip_embed, srm_embed)
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    loss.backward()
    optimizer.step()
    return loss.item()


def log(metrics: dict, step: int) -> None:
    formatted = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"step {step} {formatted}")


def train(
    model,
    dataloader,
    optimizer,
    *,
    epochs,
    checkpoint_dir,
    checkpoint_every_steps,
    start_global_step=0,
    device="cpu",
    max_steps=None,
    step_fn=train_step,
) -> int:
    checkpoint_dir = Path(checkpoint_dir)
    steps_per_epoch = len(dataloader)
    total_steps = epochs * steps_per_epoch
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    global_step = start_global_step
    data_iter = iter(dataloader)
    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        loss = step_fn(model, optimizer, batch, device)
        global_step += 1
        epoch = global_step // steps_per_epoch

        if global_step % checkpoint_every_steps == 0 or global_step == total_steps:
            save_checkpoint(
                checkpoint_path_for_step(checkpoint_dir, global_step),
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
            )

        if global_step % 50 == 0 or global_step == total_steps:
            log({"loss": loss, "epoch": epoch}, global_step)

    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--branch", choices=["full", "clip_only", "artifact_only"], default="full"
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train-time augmentation (default on; --no-augment for a clean baseline)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["train"].get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(config["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    backbone = None
    if args.branch != "artifact_only":
        backbone = CLIPBackbone.from_config(config).to(device)
        backbone.eval()  # frozen; standalone here (not a submodule of model), so not auto-managed by model.train()

    model = build_model(config, branch=args.branch).to(device)
    model.train()  # FusionHead's BatchNorm1d/Dropout depend on train/eval mode, unlike the old dummy head
    optimizer = build_optimizer(model, config)

    start_global_step = 0
    if config["train"].get("resume_from_latest", True):
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest is not None:
            meta = load_checkpoint(latest, model, optimizer, device=device)
            start_global_step = meta["global_step"]
            print(f"resumed from {latest} at global_step={start_global_step}")
        else:
            print("no checkpoint found under checkpoint_dir, starting fresh")

    # CLIP embeddings for clean (non-augmented) samples are the same every
    # epoch (frozen backbone) — serve them from the precomputed cache if one
    # exists, and only run the ViT live for the augmented ~half + cache misses.
    # Build it with: python data/cache_clip_embeddings.py --config <config>
    clip_cache = None
    if args.branch != "artifact_only":
        cache_dir = REPO_ROOT / "data" / "cache" / "clip_embeddings"
        if CLIPEmbeddingCache.npz_path(cache_dir, "train").exists():
            clip_cache = CLIPEmbeddingCache.load(cache_dir, "train")
            print(f"CLIP embedding cache: {len(clip_cache)} entries")
        else:
            print(
                "no CLIP embedding cache at data/cache/clip_embeddings/train.npz "
                "— embedding live every step (slow); run data/cache_clip_embeddings.py"
            )

    dataloader = build_dataloader(
        config, split="train", augment=args.augment, use_meta=clip_cache is not None
    )
    pos_weight = compute_pos_weight(dataloader.dataset)
    step_fn = functools.partial(
        real_train_step,
        backbone=backbone,
        branch=args.branch,
        pos_weight=pos_weight,
        clip_cache=clip_cache,
    )
    train(
        model,
        dataloader,
        optimizer,
        epochs=config["train"]["epochs"],
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_steps=config["train"]["checkpoint_every_steps"],
        start_global_step=start_global_step,
        device=device,
        max_steps=args.max_steps,
        step_fn=step_fn,
    )


if __name__ == "__main__":
    main()
