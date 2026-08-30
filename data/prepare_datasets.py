"""Owner: Data Lead.

Downloads/prepares SID_Set (HuggingFace), CIFAKE (Kaggle), and WildFake
(ModelScope — must be manually translated via the site's translation button
before this script can consume it; not automatable).

Responsibilities:
- Download raw data into data/raw/ (gitignored).
- Dedup/near-duplicate check across sources and against each other.
- Build fixed train/val/test split CSVs (image_path, label, source) under
  data/cache/splits/ — committed to the repo so the team shares one exact
  partition (deterministic given --seed); image_path is written REPO_ROOT-
  relative (see data/paths.py) so the CSVs are portable across machines/OSes.
- Check and report class balance per source.

Label convention (matches infer.py's "pred = P(AIGC)" contract):
  0 = authentic, 1 = AI-generated.

Dataset-specific notes:
- SID_Set ships 249 train shards (~140GB total) and a 3-class label (0=real,
  1=full-synthetic, 2=tampered). Tampered/locally-edited real photos don't
  map cleanly to either side of this binary task and the brief only asks
  about full generation detection, so label 2 rows are dropped. Shard count
  is deliberately capped (--sid-shards) — full volume is infeasible on
  free-tier Colab and generator-family diversity matters more (see the
  design doc).
- CIFAKE unzips to train/{REAL,FAKE} and test/{REAL,FAKE}.
- WildFake: only a manually-downloaded subset lives under data/raw/wildfake/
  (celebahq.zip = real, DDIM.zip = one diffusion generator family) — see
  data/raw/wildfake/README or team chat for how that subset was chosen.
  Each zip is single-label, so folder name -> label is a fixed mapping.
  Sample size is capped (--wildfake-cap) so it doesn't dwarf the other two
  sources once unzipped (celebahq ~30K + DDIM ~66K images raw).

Run: python data/prepare_datasets.py --config configs/train.yaml
"""

import argparse
import csv
import io
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image

_REPO_ROOT_FOR_PATH = Path(__file__).resolve().parent.parent
# Allow `python data/prepare_datasets.py ...` (script run from inside data/,
# so the repo root isn't on sys.path by default).
if str(_REPO_ROOT_FOR_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_PATH))

from data.paths import REPO_ROOT, to_repo_relative

RAW_DIR = REPO_ROOT / "data" / "raw"

REAL_LABEL = 0
FAKE_LABEL = 1

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SID_SET_REPO = "saberzl/SID_Set"
SID_SET_TOTAL_SHARDS = 249

CIFAKE_DATASET = "birdy654/cifake-real-and-ai-generated-synthetic-images"

WILDFAKE_ZIPS = {
    "celebahq.zip": ("celebahq", REAL_LABEL),
    "DDIM.zip": ("DDIM", FAKE_LABEL),
}


def download_cifake(raw_dir: Path) -> Path:
    """Downloads + unzips CIFAKE via the Kaggle API. Idempotent."""
    out_dir = raw_dir / "cifake"
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[cifake] already present at {out_dir}, skipping download")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    print("[cifake] downloading via Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(CIFAKE_DATASET, path=str(out_dir), unzip=True)
    return out_dir


def download_sid_set(raw_dir: Path, num_shards: int) -> Path:
    """Downloads a capped number of SID_Set train shards and decodes them to
    individual JPEGs + a labels.csv (real/full-synthetic only; tampered
    rows dropped). Idempotent per-shard via _shards_downloaded.txt."""
    out_dir = raw_dir / "sid_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_csv = out_dir / "labels.csv"
    progress_file = out_dir / "_shards_downloaded.txt"
    downloaded = set(progress_file.read_text().splitlines()) if progress_file.exists() else set()

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    write_header = not labels_csv.exists()
    with open(labels_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image_path", "label"])
        for shard_idx in range(num_shards):
            shard_name = f"train-{shard_idx:05d}-of-{SID_SET_TOTAL_SHARDS:05d}.parquet"
            if shard_name in downloaded:
                print(f"[sid_set] {shard_name} already ingested, skipping")
                continue
            print(f"[sid_set] downloading {shard_name}...")
            local_parquet = hf_hub_download(
                repo_id=SID_SET_REPO, repo_type="dataset", filename=f"data/{shard_name}"
            )
            table = pq.read_table(local_parquet, columns=["image", "label"])
            n_written = 0
            for i, row in enumerate(table.to_pylist()):
                label = int(row["label"])
                if label not in (REAL_LABEL, FAKE_LABEL):
                    continue  # drop tampered (label 2) — see module docstring
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                out_path = out_dir / f"{shard_idx:05d}_{i:05d}.jpg"
                img.save(out_path, "JPEG", quality=95)
                writer.writerow([to_repo_relative(out_path), label])
                n_written += 1
            print(f"[sid_set]   -> {n_written} usable images (tampered rows dropped)")
            downloaded.add(shard_name)
            progress_file.write_text("\n".join(sorted(downloaded)))
    return out_dir


def ingest_wildfake(raw_dir: Path) -> Path:
    """Unzips the manually-downloaded WildFake subset (celebahq.zip,
    DDIM.zip) so their images are on disk and accessible, same as the other
    two sources. Idempotent: skips a zip if its target folder already has
    files."""
    wildfake_dir = raw_dir / "wildfake"
    for zip_name, (folder_name, _label) in WILDFAKE_ZIPS.items():
        zip_path = wildfake_dir / zip_name
        target_dir = wildfake_dir / folder_name
        if target_dir.exists() and any(target_dir.rglob("*")):
            print(f"[wildfake] {folder_name} already unzipped, skipping")
            continue
        if not zip_path.exists():
            print(f"[wildfake] {zip_path} not found, skipping (see README setup steps)")
            continue
        print(f"[wildfake] unzipping {zip_name}...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(wildfake_dir)
    return wildfake_dir


def _manifest_cifake(cifake_dir: Path) -> list[dict]:
    entries = []
    if not cifake_dir.exists():
        return entries
    for split_dir in cifake_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for label_name, label in (("REAL", REAL_LABEL), ("FAKE", FAKE_LABEL)):
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            for p in label_dir.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    entries.append({"image_path": str(p), "label": label, "source": "cifake"})
    return entries


def _manifest_sid_set(sid_dir: Path) -> list[dict]:
    entries = []
    labels_csv = sid_dir / "labels.csv"
    if not labels_csv.exists():
        return entries
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            entries.append(
                {"image_path": row["image_path"], "label": int(row["label"]), "source": "sid_set"}
            )
    return entries


def _manifest_wildfake(wildfake_dir: Path, cap_per_class: int, rng: random.Random) -> list[dict]:
    entries = []
    for folder_name, label in WILDFAKE_ZIPS.values():
        folder = wildfake_dir / folder_name
        if not folder.exists():
            continue
        images = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        if len(images) > cap_per_class:
            images = rng.sample(images, cap_per_class)
        entries.extend({"image_path": str(p), "label": label, "source": "wildfake"} for p in images)
    return entries


def build_manifest(raw_dir: Path, wildfake_cap: int, rng: random.Random) -> list[dict]:
    manifest = []
    manifest.extend(_manifest_cifake(raw_dir / "cifake"))
    manifest.extend(_manifest_sid_set(raw_dir / "sid_set"))
    manifest.extend(_manifest_wildfake(raw_dir / "wildfake", wildfake_cap, rng))
    return manifest


def average_hash(image_path: str, hash_size: int = 8) -> int:
    with Image.open(image_path) as img:
        img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)
    bits = arr > arr.mean()
    return int.from_bytes(np.packbits(bits), byteorder="big")


def dedup(manifest: list[dict]) -> tuple[list[dict], dict]:
    """Drops near-duplicate images (identical 8x8 average-hash), keeping the
    first occurrence. Reports how many were dropped per source pair."""
    seen = {}
    kept = []
    dropped_pairs = defaultdict(int)
    for entry in manifest:
        h = average_hash(entry["image_path"])
        if h in seen:
            pair = tuple(sorted((seen[h]["source"], entry["source"])))
            dropped_pairs[pair] += 1
            continue
        seen[h] = entry
        kept.append(entry)
    return kept, dropped_pairs


def report_class_balance(manifest: list[dict]) -> None:
    counts = defaultdict(lambda: defaultdict(int))
    for entry in manifest:
        counts[entry["source"]][entry["label"]] += 1
    print("--- class balance per source ---")
    for source, label_counts in sorted(counts.items()):
        total = sum(label_counts.values())
        real = label_counts.get(REAL_LABEL, 0)
        fake = label_counts.get(FAKE_LABEL, 0)
        print(f"  {source}: total={total} real={real} ({real / total:.1%}) fake={fake} ({fake / total:.1%})")


def stratified_split(
    manifest: list[dict], val_frac: float, test_frac: float, rng: random.Random
) -> tuple[list[dict], list[dict], list[dict]]:
    """Splits independently per (source, label) bucket so both class balance
    and source mix are preserved across train/val/test."""
    buckets = defaultdict(list)
    for entry in manifest:
        buckets[(entry["source"], entry["label"])].append(entry)

    train, val, test = [], [], []
    for items in buckets.values():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        val.extend(items[:n_val])
        test.extend(items[n_val : n_val + n_test])
        train.extend(items[n_val + n_test :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_split_csv(entries: list[dict], out_path: Path) -> None:
    """Writes image_path as a REPO_ROOT-relative POSIX path (see data/paths.py)
    so the CSV is portable across machines/OSes rather than hard-wired to
    whoever generated it — data/dataset.py and data/cache_clip_embeddings.py
    resolve it back via resolve_image_path()."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "source"])
        for e in entries:
            writer.writerow([to_repo_relative(e["image_path"]), e["label"], e["source"]])


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--sid-shards",
        type=int,
        default=2,
        help="Number of SID_Set train shards to download (~840 images/shard, ~560MB/shard). "
        f"Full dataset is {SID_SET_TOTAL_SHARDS} shards / ~140GB — deliberately capped.",
    )
    parser.add_argument(
        "--wildfake-cap",
        type=int,
        default=5000,
        help="Max images per WildFake class after unzip, so it doesn't dwarf the other sources.",
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse whatever is already in data/raw/ instead of re-downloading/re-unzipping.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    splits_dir = REPO_ROOT / config["data"]["splits_dir"]

    if not args.skip_download:
        download_cifake(RAW_DIR)
        download_sid_set(RAW_DIR, args.sid_shards)
        ingest_wildfake(RAW_DIR)

    rng = random.Random(args.seed)
    manifest = build_manifest(RAW_DIR, args.wildfake_cap, rng)
    print(f"\nCollected {len(manifest)} images before dedup")

    manifest, dropped_pairs = dedup(manifest)
    n_dropped = sum(dropped_pairs.values())
    print(f"{len(manifest)} images after dedup ({n_dropped} near-duplicates dropped)")
    for (source_a, source_b), count in sorted(dropped_pairs.items()):
        print(f"  {source_a} <-> {source_b}: {count}")

    report_class_balance(manifest)

    train, val, test = stratified_split(manifest, args.val_frac, args.test_frac, rng)
    write_split_csv(train, splits_dir / "train.csv")
    write_split_csv(val, splits_dir / "val.csv")
    write_split_csv(test, splits_dir / "test.csv")
    print(f"\nWrote splits to {splits_dir}: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
