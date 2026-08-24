"""
preprocessing/splitting.py

Phase 1, Step 4: subject-level train/val/test split.

Must split by subject, never by epoch -- epochs from the same subject
are correlated, so epoch-level splitting would leak information and
inflate validation/test accuracy. Both MHA and Bi-LSTM must load the
same saved split.
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SPLIT_PATH = Path("data/processed/subject_split.npz")


def make_subject_split(
    subject_ids: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Split unique subjects (not epochs) into train/val/test groups.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each a sorted list of
    subject ID strings.
    """
    unique_subjects = np.unique(subject_ids)

    # First split off test subjects
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(unique_subjects, groups=unique_subjects))
    trainval_subjects = unique_subjects[trainval_idx]
    test_subjects = unique_subjects[test_idx]

    # Then split remaining into train/val
    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(trainval_subjects, groups=trainval_subjects))
    train_subjects = trainval_subjects[train_idx]
    val_subjects = trainval_subjects[val_idx]

    split = {
        "train": sorted(train_subjects.tolist()),
        "val": sorted(val_subjects.tolist()),
        "test": sorted(test_subjects.tolist()),
    }

    # Sanity check: no overlap between sets
    assert set(split["train"]) & set(split["val"]) == set()
    assert set(split["train"]) & set(split["test"]) == set()
    assert set(split["val"]) & set(split["test"]) == set()

    return split


def save_split(split: dict, path: Path = SPLIT_PATH):
    np.savez(path, **{k: np.array(v) for k, v in split.items()})
    print(f"Saved split to {path}")
    for k, v in split.items():
        print(f"  {k}: {len(v)} subjects -> {v}")


def load_split(path: Path = SPLIT_PATH) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k].tolist() for k in data.files}


if __name__ == "__main__":
    DATA_PATH = Path("data/processed/eeg_epochs.npz")
    data = np.load(DATA_PATH, allow_pickle=True)
    subject_ids = data["subject_ids"]

    split = make_subject_split(subject_ids)
    save_split(split)
