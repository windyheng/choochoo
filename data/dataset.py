"""Owner: Data Lead.

torch Dataset/DataLoader over the prepared SID_Set + CIFAKE + WildFake splits.
Applies data.transforms.random_transform for training augmentation. Must
produce (image_tensor, label, image_path) tuples — image_path is needed by
error_analysis.py to trace predictions back to source files.

Images are resized to a fixed CANONICAL_SIZE and returned as float32 tensors
in [0, 1], channel-first, WITHOUT branch-specific normalization — CLIP's own
mean/std (or lack of any for the SRM branch) is applied inside each branch's
embed(), per docs/interfaces.md.

split_csv's image_path is REPO_ROOT-relative (see data/paths.py); resolved
via resolve_image_path() before opening, but returned to the caller as the
original relative string (matching the split CSV / predictions.csv format
error_analysis.py traces back through).
"""

import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data import transforms
from data.paths import resolve_image_path

CANONICAL_SIZE = (224, 224)  # matches CLIP ViT-B/16's native input resolution


class AIGCDataset(Dataset):
    def __init__(self, split_csv: str, augment: bool = False, apply_prob: float = 0.5):
        with open(split_csv, newline="") as f:
            self.samples = [(row["image_path"], int(row["label"])) for row in csv.DictReader(f)]
        self.augment = augment
        self.apply_prob = apply_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(resolve_image_path(image_path)).convert("RGB")

        if self.augment:
            image = transforms.random_transform(image, self.apply_prob)

        image = image.resize(CANONICAL_SIZE, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

        return image_tensor, label, image_path
