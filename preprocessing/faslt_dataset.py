from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

FEATURES_PATH = Path("data/processed/faslt_features_standardized.npy")
META_PATH = Path("data/processed/eeg_denoised_metadata.npz")
SPLIT_PATH = Path("data/processed/subject_split.npz")


class FASLTDataset(Dataset):
    def __init__(self, split_name: str):
        if split_name not in ("train", "val", "test"):
            raise ValueError(f"split_name must be 'train'/'val'/'test', got {split_name!r}")

        # mmap_mode="r" -- data is NOT loaded into RAM here, only on __getitem__
        self.features = np.load(FEATURES_PATH, mmap_mode="r")
        metadata = np.load(META_PATH, allow_pickle=True)
        split = np.load(SPLIT_PATH, allow_pickle=True)

        self.y_all = metadata["y"]
        subject_ids = metadata["subject_ids"]
        subjects_in_split = split[split_name]

        mask = np.isin(subject_ids, subjects_in_split)
        self.indices = np.where(mask)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # np.asarray() here pulls just THIS one epoch off disk into RAM
        x = np.asarray(self.features[real_idx]).astype(np.float32)  # (19, 40, 1000)
        y = int(self.y_all[real_idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


if __name__ == "__main__":
    for name in ("train", "val", "test"):
        ds = FASLTDataset(name)
        x, y = ds[0]
        print(f"{name}: {len(ds)} samples, sample X shape {tuple(x.shape)}, sample y {y.item()}")