import numpy as np

data = np.load("data/processed/eeg_epochs.npz", allow_pickle=True)
X, y, subject_ids = data["X"], data["y"], data["subject_ids"]

split = np.load("data/processed/subject_split.npz", allow_pickle=True)

for name in ["train", "val", "test"]:
    subjects = split[name]
    mask = np.isin(subject_ids, subjects)
    y_subset = y[mask]
    print(f"{name}: {mask.sum()} epochs, label counts {np.bincount(y_subset)}, "
          f"ratio {np.bincount(y_subset)[0] / np.bincount(y_subset)[1]:.2f}:1")
          