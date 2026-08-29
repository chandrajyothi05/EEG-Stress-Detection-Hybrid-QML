"""
preprocessing/standardize_faslt.py

Per-channel standardization of log-normalized FASLT features.
Mean/std computed from the TRAIN split only, then applied to
train/val/test -- never compute stats from val/test, or you leak
their distribution into training (same principle as subject-level
splitting: no information should cross from held-out data back
into what the model learns from).
"""

from pathlib import Path
import numpy as np

IN_PATH = Path("data/processed/faslt_features_log.npy")
META_PATH = Path("data/processed/eeg_denoised_metadata.npz")
SPLIT_PATH = Path("data/processed/subject_split.npz")

OUT_PATH = Path("data/processed/faslt_features_standardized.npy")
STATS_PATH = Path("data/processed/faslt_standardize_stats.npz")


def compute_train_stats(features, train_indices):
    """
    Returns per-channel mean and std, shape (n_channels,), computed
    only from epochs in train_indices.
    """
    n_channels = features.shape[1]
    means = np.zeros(n_channels, dtype=np.float64)
    stds = np.zeros(n_channels, dtype=np.float64)

    for ch in range(n_channels):
        # Pull only this channel, only train epochs, into RAM
        channel_data = np.asarray(features[train_indices, ch, :, :])
        means[ch] = channel_data.mean()
        stds[ch] = channel_data.std()
        print(f"Channel {ch}: mean={means[ch]:.4f}, std={stds[ch]:.4f}")

    stds[stds == 0] = 1.0  # guard against zero-variance channel
    return means.astype(np.float32), stds.astype(np.float32)


if __name__ == "__main__":
    features = np.load(IN_PATH, mmap_mode="r")
    metadata = np.load(META_PATH, allow_pickle=True)
    split = np.load(SPLIT_PATH, allow_pickle=True)

    subject_ids = metadata["subject_ids"]
    train_subjects = split["train"]
    train_mask = np.isin(subject_ids, train_subjects)
    train_indices = np.where(train_mask)[0]

    print(f"Computing stats from {len(train_indices)} train epochs...")
    means, stds = compute_train_stats(features, train_indices)

    np.savez(STATS_PATH, means=means, stds=stds)
    print(f"Saved stats to {STATS_PATH}")

    # Apply to ALL epochs (train, val, test alike -- using train-derived stats)
    n_epochs = features.shape[0]
    batch_size = 500

    standardized = np.lib.format.open_memmap(
        OUT_PATH, mode="w+", dtype=np.float32, shape=features.shape
    )

    means_r = means.reshape(1, -1, 1, 1)  # broadcast over (epoch, channel, freq, time)
    stds_r = stds.reshape(1, -1, 1, 1)

    for start in range(0, n_epochs, batch_size):
        end = min(start + batch_size, n_epochs)
        batch = np.asarray(features[start:end])
        standardized[start:end] = (batch - means_r) / stds_r
        print(f"Standardized epochs {start}-{end} / {n_epochs}")

    standardized.flush()
    print(f"Saved standardized features to {OUT_PATH}")

    # Quick sanity check on train epochs only -- should be ~0 mean, ~1 std
    sample = np.asarray(standardized[train_indices[:200]])
    print(f"Train sample check: mean={sample.mean():.4f}, std={sample.std():.4f}")