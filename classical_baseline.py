"""
Classical baseline: band-power features + Logistic Regression.
Purpose: isolate whether the rest-vs-stress bias seen in ConvMixer comes from
the FASLT features themselves, or from the ConvMixer architecture/training.
Uses the SAME subject-level split and SAME subject-level majority-vote
evaluation as train_convmixer.py, so results are directly comparable.
"""
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from preprocessing.faslt_dataset import FASLTDataset

# freqs = linspace(1, 40, 40) -- ~1Hz bins. Standard EEG band definitions:
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}
FREQS = np.linspace(1, 40, 40)


def band_indices(low, high):
    return np.where((FREQS >= low) & (FREQS < high))[0]


BAND_IDX = {name: band_indices(lo, hi) for name, (lo, hi) in BANDS.items()}


def extract_band_power_features(dataset: FASLTDataset):
    """
    For each epoch: input is (19 channels, 40 freq bins, 1000 time samples).
    Reduce to (19 channels x 5 bands) = 95 features by averaging over time,
    then averaging over each band's frequency bins.
    """
    n_samples = len(dataset)
    n_channels = 19
    n_bands = len(BANDS)
    X = np.zeros((n_samples, n_channels * n_bands), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    subjects = np.empty(n_samples, dtype=object)

    for i in range(n_samples):
        real_idx = dataset.indices[i]
        epoch = np.asarray(dataset.features[real_idx])  # (19, 40, 1000)
        time_avg = epoch.mean(axis=2)  # (19, 40) -- average over time

        feat = np.zeros((n_channels, n_bands), dtype=np.float32)
        for b_idx, (band_name, freq_idx) in enumerate(BAND_IDX.items()):
            feat[:, b_idx] = time_avg[:, freq_idx].mean(axis=1)  # avg within band

        X[i] = feat.flatten()
        y[i] = dataset.y_all[real_idx]
        subjects[i] = dataset.subject_ids_all[real_idx]

    return X, y, subjects


def evaluate_subject_level(preds, y_true, subjects):
    groups = defaultdict(list)
    for pred, true, subj in zip(preds, y_true, subjects):
        groups[(subj, true)].append(pred)

    correct, total = 0, 0
    print("\n  Subject-level (majority-vote) breakdown:")
    for (subj, true_label), group_preds in sorted(groups.items()):
        majority_pred = int(round(np.mean(group_preds)))
        is_correct = majority_pred == true_label
        correct += is_correct
        total += 1
        epoch_acc = np.mean(np.array(group_preds) == true_label)
        print(f"    subject={subj} true_label={true_label} "
              f"n_epochs={len(group_preds)} epoch_acc={epoch_acc:.2f} "
              f"majority_pred={majority_pred} {'OK' if is_correct else 'WRONG'}")

    subject_acc = correct / total
    print(f"  Subject-level accuracy: {correct}/{total} = {subject_acc:.4f}\n")
    return subject_acc


if __name__ == "__main__":
    print("Extracting band-power features...")
    train_ds = FASLTDataset("train")
    val_ds = FASLTDataset("val")

    X_train, y_train, _ = extract_band_power_features(train_ds)
    X_val, y_val, subjects_val = extract_band_power_features(val_ds)

    # Standardize using train stats only (same convention as the rest of the pipeline)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    epoch_acc = (val_preds == y_val).mean()
    print(f"\nEpoch-level val accuracy: {epoch_acc:.4f}")

    evaluate_subject_level(val_preds, y_val, subjects_val)