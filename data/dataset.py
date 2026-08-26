"""Owner: Data Lead.

torch Dataset/DataLoader over the prepared SID_Set + CIFAKE + WildFake splits.
Applies data.transforms.random_transform for training augmentation. Must
produce (image_tensor, label, image_path) tuples — image_path is needed by
error_analysis.py to trace predictions back to source files.
"""

from torch.utils.data import Dataset


class AIGCDataset(Dataset):
    def __init__(self, split_csv: str, augment: bool = False):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
