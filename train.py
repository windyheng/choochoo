"""Owner: Training/Infra Lead.

Training loop with checkpoint/resume — REQUIRED for free-tier Colab, which
disconnects at the ~12hr session cap and sometimes earlier. Must detect and
resume from the latest checkpoint under configs' checkpoint_dir on restart,
not assume a single uninterrupted run. Build this before the real training
run, not after a disconnect eats progress.

The model/dataloader below (DummyFusionHead / DummyEmbeddingDataset) are
synthetic placeholders standing in for models/backbone_clip.py,
models/artifact_branch.py, and models/fusion_head.py, whose embedding dims
are still TODO in docs/interfaces.md. build_model()/build_dataloader() are
the seam: swap them for the real dataset/FusionHead once those land, the
loop and checkpoint/resume code below does not need to change.

Run: python train.py --config configs/train.yaml
"""

import argparse
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Placeholder dims. D/D2 are still TODO in docs/interfaces.md (real CLIP/SRM
# branches not built yet) — these are arbitrary stand-ins, not guesses at the
# eventual real values.
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
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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


def build_dataloader(config) -> DataLoader:
    # Swap: return a DataLoader over data.dataset.AIGCDataset (or precomputed
    # embeddings) instead — batching setup below stays the same.
    dataset = DummyEmbeddingDataset(DUMMY_DATASET_SIZE, DUMMY_CLIP_DIM, DUMMY_SRM_DIM)
    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=0,  # in-memory random tensors; num_workers in config is for real image decoding
    )


class DummyFusionHead(nn.Module):
    """Placeholder for models.fusion_head.FusionHead (still
    NotImplementedError, pending D/D2 in docs/interfaces.md). Same
    forward(clip_embed, srm_embed) -> logits[B] contract."""

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


def build_model(config) -> nn.Module:
    # Swap: return models.fusion_head.FusionHead(clip_dim, srm_dim,
    # config["model"]["fusion_hidden_dim"]) once implemented and D/D2 settled.
    return DummyFusionHead(DUMMY_CLIP_DIM, DUMMY_SRM_DIM, config["model"]["fusion_hidden_dim"])


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

        loss = train_step(model, optimizer, batch, device)
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
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["train"].get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(config["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config).to(device)
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

    dataloader = build_dataloader(config)
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
    )


if __name__ == "__main__":
    main()
