"""
preprocessing/load_split_data.py

Final shared step: load SWT features for a given split (train/val/test),
using the subject-level split saved earlier. Both MHA and Bi-LSTM should
import load_split_data rather than re-implementing this.
"""

from pathlib import Path
import numpy as np

FEATURES_PATH = Path("data/processed/eeg_swt_features.npy")
METADATA_PATH = Path("data/processed/eeg_swt_metadata.npz")
SPLIT_PATH = Path("data/processed/subject_split.npz")


def load_split_data(split_name: str):
    """
    split_name: 'train', 'val', or 'test'

    Returns
    -------
    X : np.ndarray, shape (n_epochs_in_split, 19, 6, 992)
    y : np.ndarray, shape (n_epochs_in_split,)
    """
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"split_name must be 'train', 'val', or 'test', got {split_name!r}")

    features = np.load(FEATURES_PATH, mmap_mode="r")
    metadata = np.load(METADATA_PATH, allow_pickle=True)
    split = np.load(SPLIT_PATH, allow_pickle=True)

    y_all = metadata["y"]
    subject_ids = metadata["subject_ids"]
    subjects_in_split = split[split_name]

    mask = np.isin(subject_ids, subjects_in_split)
    indices = np.where(mask)[0]

    X = features[indices]
    y = y_all[indices]

    return X, y


if __name__ == "__main__":
    for name in ("train", "val", "test"):
        X, y = load_split_data(name)
        print(f"{name}: X {X.shape}, y {y.shape}, label counts {np.bincount(y)}")