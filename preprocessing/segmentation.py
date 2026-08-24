"""
preprocessing/segmentation.py

Phase 1, Step 3: epoching + normalization, and a single entry point that
builds one subject's (X, y) arrays from raw EDF through to normalized
epochs. Both the MHA and Bi-LSTM branches should call
`build_subject_dataset` rather than re-implementing any of this --
that's the whole point of keeping this pipeline shared.

Label convention (see project notes): session "1" (background/resting)
-> 0, session "2" (mental arithmetic task) -> 1. This is a task-condition
proxy for stress, not a clinical diagnosis -- document that assumption
in the report.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mne

from load_eeg import load_recording
from filtering import FilterConfig, select_eeg_channels, apply_filters

DATA_DIR = Path("data/raw/eegmat")

SESSION_LABELS = {"1": 0, "2": 1}  # background -> 0, task -> 1


@dataclass
class EpochConfig:
    """Epoching parameters -- shared by both branches."""
    duration_sec: float = 2.0   # length of each epoch/window
    overlap_sec: float = 1.0    # overlap between consecutive epochs (0 = none)


def epoch_raw(raw: mne.io.Raw, config: EpochConfig = EpochConfig()) -> np.ndarray:
    """
    Split a continuous Raw recording into fixed-length epochs.

    Returns
    -------
    np.ndarray, shape (n_epochs, n_channels, n_samples_per_epoch)
    """
    epochs = mne.make_fixed_length_epochs(
        raw, duration=config.duration_sec, overlap=config.overlap_sec,
        preload=True, verbose="ERROR",
    )
    return epochs.get_data()  # (n_epochs, n_channels, n_samples)


def normalize_epochs(epochs_array: np.ndarray) -> np.ndarray:
    """
    Per-epoch, per-channel z-score normalization: each channel within
    each epoch is centered and scaled independently. This is done AFTER
    epoching (not on the continuous signal) so normalization can't leak
    information across epoch boundaries.

    Parameters
    ----------
    epochs_array : np.ndarray, shape (n_epochs, n_channels, n_samples)

    Returns
    -------
    np.ndarray, same shape, z-scored along the last axis.
    """
    mean = epochs_array.mean(axis=-1, keepdims=True)
    std = epochs_array.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0  # guard against flat/zero-variance channels
    return (epochs_array - mean) / std


def build_subject_dataset(
    subject: str,
    data_dir: Path = DATA_DIR,
    filter_config: FilterConfig = FilterConfig(),
    epoch_config: EpochConfig = EpochConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the full (X, y, subject_ids) arrays for ONE subject, combining
    both sessions (_1 background, _2 task).

    Returns
    -------
    X : np.ndarray, shape (n_epochs_total, n_channels, n_samples_per_epoch)
    y : np.ndarray, shape (n_epochs_total,) -- 0 = background, 1 = task
    subject_ids : np.ndarray, shape (n_epochs_total,) -- string subject id
        repeated for every epoch, so this can be concatenated across
        subjects and still support subject-wise splitting downstream.
    """
    X_parts, y_parts = [], []

    for session, label in SESSION_LABELS.items():
        raw = load_recording(subject, session, data_dir=data_dir)
        raw = select_eeg_channels(raw)
        raw = apply_filters(raw, filter_config)

        epochs_array = epoch_raw(raw, epoch_config)
        epochs_array = normalize_epochs(epochs_array)

        X_parts.append(epochs_array)
        y_parts.append(np.full(epochs_array.shape[0], label, dtype=int))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    subject_ids = np.full(X.shape[0], subject)

    return X, y, subject_ids
def build_full_dataset(
    subjects: list[str] | None = None,
    data_dir: Path = DATA_DIR,
    filter_config: FilterConfig = FilterConfig(),
    epoch_config: EpochConfig = EpochConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, subject_ids) across ALL subjects, concatenated into
    one dataset. Both MHA and Bi-LSTM should load from the saved
    output of this rather than re-running segmentation themselves.
    """
    if subjects is None:
        subjects = [f"Subject{i:02d}" for i in range(36)]  # Subject00..Subject35

    X_all, y_all, ids_all = [], [], []

    for subject in subjects:
        try:
            X, y, subject_ids = build_subject_dataset(
                subject, data_dir=data_dir,
                filter_config=filter_config, epoch_config=epoch_config,
            )
        except FileNotFoundError as e:
            print(f"Skipping {subject}: {e}")
            continue

        X_all.append(X)
        y_all.append(y)
        ids_all.append(subject_ids)

        print(f"{subject}: {X.shape[0]} epochs, labels {np.bincount(y)}")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    ids_all = np.concatenate(ids_all, axis=0)

    return X_all, y_all, ids_all


if __name__ == "__main__":
    OUT_PATH = Path("data/processed/eeg_epochs.npz")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    X, y, subject_ids = build_full_dataset()

    print(f"\nFinal X shape: {X.shape}")
    print(f"Final y shape: {y.shape}, label counts: {np.bincount(y)}")
    print(f"Unique subjects: {len(np.unique(subject_ids))}")

    np.savez(OUT_PATH, X=X, y=y, subject_ids=subject_ids)
    print(f"Saved combined dataset to {OUT_PATH}")