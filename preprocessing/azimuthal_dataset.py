"""
preprocessing/azimuthal_dataset.py

PyTorch Dataset for azimuthal-projection images. Stored at 32x32 (small,
loaded fully into RAM since the whole file is ~105MB); resize to 224x224
and ImageNet normalization applied per-sample via torchvision transforms,
since pretrained EfficientNet-B0 expects that input format.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGES_PATH = Path("data/processed/azimuthal_images.npy")
META_PATH = Path("data/processed/eeg_denoised_metadata.npz")
SPLIT_PATH = Path("data/processed/subject_split.npz")

# ImageNet normalization stats -- required for pretrained EfficientNet-B0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AzimuthalDataset(Dataset):
    def __init__(self, split_name: str):
        if split_name not in ("train", "val", "test"):
            raise ValueError(f"split_name must be 'train'/'val'/'test', got {split_name!r}")

        self.images = np.load(IMAGES_PATH)  # small enough to load fully
        metadata = np.load(META_PATH, allow_pickle=True)
        split = np.load(SPLIT_PATH, allow_pickle=True)

        self.y_all = metadata["y"]
        subject_ids = metadata["subject_ids"]
        subjects_in_split = split[split_name]

        mask = np.isin(subject_ids, subjects_in_split)
        self.indices = np.where(mask)[0]

        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = torch.from_numpy(self.images[real_idx]).float()  # (3, 32, 32)
        x = self.transform(x)  # (3, 224, 224), normalized
        y = int(self.y_all[real_idx])
        return x, torch.tensor(y, dtype=torch.long)


if __name__ == "__main__":
    for name in ("train", "val", "test"):
        ds = AzimuthalDataset(name)
        x, y = ds[0]
        print(f"{name}: {len(ds)} samples, sample X shape {tuple(x.shape)}, sample y {y.item()}")