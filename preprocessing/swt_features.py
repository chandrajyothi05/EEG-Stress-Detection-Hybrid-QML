"""
preprocessing/swt_features.py

Phase 1, Step 5: Stationary Wavelet Transform (SWT) feature extraction.

Disk-based batching version.

Input:
    data/processed/eeg_X.npy
        Shape: (n_epochs, n_channels, n_samples)

Metadata:
    data/processed/eeg_epochs.npz
        Contains y and subject_ids

Output:
    data/processed/eeg_swt_features.npy
        Memory-mapped SWT feature array

    data/processed/eeg_swt_metadata.npz
        Labels and subject IDs
"""

import numpy as np
import pywt
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

@dataclass
class SWTConfig:
    wavelet: str = "db4"
    level: int = 5


# ============================================================
# Helper
# ============================================================

def _next_valid_length(n_samples: int, level: int) -> int:
    """
    Return the largest signal length <= n_samples
    that is divisible by 2**level.
    """

    factor = 2 ** level
    return (n_samples // factor) * factor


# ============================================================
# Single-channel SWT
# ============================================================

def swt_single_channel(
    signal: np.ndarray,
    config: SWTConfig
) -> np.ndarray:
    """
    Apply SWT to one EEG channel.

    Returns:
        Shape: (level + 1, valid_len)

    Ordering:
        [cD1, cD2, cD3, cD4, cD5, cA5]
    """

    valid_len = _next_valid_length(
        len(signal),
        config.level
    )

    if valid_len == 0:
        raise ValueError(
            f"Signal length {len(signal)} is too short "
            f"for SWT level {config.level}."
        )

    signal = signal[:valid_len]

    coeffs = pywt.swt(
        signal,
        wavelet=config.wavelet,
        level=config.level,
        trim_approx=True,
    )

    # pywt returns:
    # [cA5, cD5, cD4, cD3, cD2, cD1]

    cA_level = coeffs[0]

    details = coeffs[1:][::-1]

    # Reorder to:
    # [cD1, cD2, cD3, cD4, cD5, cA5]

    stacked = np.stack(
        details + [cA_level],
        axis=0
    )

    return stacked.astype(np.float32)


# ============================================================
# Batch SWT extraction
# ============================================================

def extract_swt_features(
    X: np.ndarray,
    config: SWTConfig
) -> np.ndarray:
    """
    Apply SWT to a batch of EEG epochs.

    Input:
        X:
            (n_epochs, n_channels, n_samples)

    Output:
        (n_epochs, n_channels, level + 1, valid_len)
    """

    X = np.asarray(X, dtype=np.float32)

    n_epochs, n_channels, n_samples = X.shape

    valid_len = _next_valid_length(
        n_samples,
        config.level
    )

    if valid_len == 0:
        raise ValueError(
            f"Signal length {n_samples} is too short "
            f"for SWT level {config.level}."
        )

    out = np.empty(
        (
            n_epochs,
            n_channels,
            config.level + 1,
            valid_len,
        ),
        dtype=np.float32,
    )

    for epoch_idx in range(n_epochs):

        for channel_idx in range(n_channels):

            out[epoch_idx, channel_idx] = (
                swt_single_channel(
                    X[epoch_idx, channel_idx],
                    config
                )
            )

    return out


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    NPZ_PATH = Path(
        "data/processed/eeg_epochs.npz"
    )

    INPUT_NPY_PATH = Path(
        "data/processed/eeg_X.npy"
    )

    FEATURES_PATH = Path(
        "data/processed/eeg_swt_features.npy"
    )

    METADATA_PATH = Path(
        "data/processed/eeg_swt_metadata.npz"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = SWTConfig()

    batch_size = 100

    print("=" * 60)
    print("SWT FEATURE EXTRACTION")
    print("=" * 60)

    print(f"Wavelet       : {config.wavelet}")
    print(f"SWT level     : {config.level}")
    print(f"Batch size    : {batch_size}")
    print()

    # --------------------------------------------------------
    # Create memory-mapped input if it does not exist
    # --------------------------------------------------------

    if not INPUT_NPY_PATH.exists():

        print(
            "Input .npy file not found."
        )

        print(
            "Creating memory-mapped input file..."
        )

        data = np.load(
            NPZ_PATH,
            allow_pickle=True
        )

        X_original = data["X"]

        print(
            f"Original X shape : {X_original.shape}"
        )

        print(
            f"Original X dtype : {X_original.dtype}"
        )

        # Create .npy file on disk
        X_disk = np.lib.format.open_memmap(
            INPUT_NPY_PATH,
            mode="w+",
            dtype=np.float32,
            shape=X_original.shape,
        )

        # Copy in small batches
        conversion_batch_size = 500

        for start in range(
            0,
            len(X_original),
            conversion_batch_size
        ):

            end = min(
                start + conversion_batch_size,
                len(X_original)
            )

            print(
                f"Converting input epochs "
                f"{start} - {end - 1}"
            )

            X_disk[start:end] = (
                X_original[start:end]
            )

            X_disk.flush()

        del X_disk
        del X_original
        del data

        print(
            f"Created: {INPUT_NPY_PATH}"
        )
        print()

    # --------------------------------------------------------
    # Open input as memory-mapped array
    # --------------------------------------------------------

    print(
        f"Opening input: {INPUT_NPY_PATH}"
    )

    X = np.load(
        INPUT_NPY_PATH,
        mmap_mode="r"
    )

    print(
        f"Input X shape : {X.shape}"
    )

    print(
        f"Input X dtype : {X.dtype}"
    )

    n_epochs, n_channels, n_samples = X.shape

    # --------------------------------------------------------
    # Calculate valid SWT length
    # --------------------------------------------------------

    valid_len = _next_valid_length(
        n_samples,
        config.level
    )

    n_coefficients = config.level + 1

    final_shape = (
        n_epochs,
        n_channels,
        n_coefficients,
        valid_len,
    )

    print()
    print(
        f"Valid signal length : {valid_len}"
    )

    print(
        f"Number of coefficients : {n_coefficients}"
    )

    print(
        f"Final SWT shape : {final_shape}"
    )

    # --------------------------------------------------------
    # Estimate disk size
    # --------------------------------------------------------

    estimated_gb = (
        np.prod(final_shape)
        * np.dtype(np.float32).itemsize
        / (1024 ** 3)
    )

    print(
        f"Estimated output size : "
        f"{estimated_gb:.2f} GB"
    )

    print()

    # --------------------------------------------------------
    # Create memory-mapped output
    # --------------------------------------------------------

    print(
        "Creating disk-based SWT output..."
    )

    features = np.lib.format.open_memmap(
        FEATURES_PATH,
        mode="w+",
        dtype=np.float32,
        shape=final_shape,
    )

    print(
        f"Output file created: "
        f"{FEATURES_PATH}"
    )

    print()

    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    print("=" * 60)
    print("STARTING SWT PROCESSING")
    print("=" * 60)
    print()

    for start in range(
        0,
        n_epochs,
        batch_size
    ):

        end = min(
            start + batch_size,
            n_epochs
        )

        print(
            f"Processing epochs "
            f"{start} - {end - 1} "
            f"/ {n_epochs - 1}"
        )

        # ----------------------------------------------------
        # Read only this batch from disk
        # ----------------------------------------------------

        batch_X = np.asarray(
            X[start:end],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # SWT extraction
        # ----------------------------------------------------

        batch_features = extract_swt_features(
            batch_X,
            config
        )

        # ----------------------------------------------------
        # Write batch directly to disk
        # ----------------------------------------------------

        features[start:end] = batch_features

        features.flush()

        # ----------------------------------------------------
        # Release batch RAM
        # ----------------------------------------------------

        del batch_X
        del batch_features

        print(
            f"Saved epochs "
            f"{start} - {end - 1}"
        )

        print()

    # --------------------------------------------------------
    # Close output memmap
    # --------------------------------------------------------

    del features

    # --------------------------------------------------------
    # Save labels and subject IDs
    # --------------------------------------------------------

    print(
        "Saving metadata..."
    )

    data = np.load(
        NPZ_PATH,
        allow_pickle=True
    )

    y = data["y"]
    subject_ids = data["subject_ids"]

    np.savez(
        METADATA_PATH,
        y=y,
        subject_ids=subject_ids,
    )

    del data
    del y
    del subject_ids

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VERIFYING SWT OUTPUT")
    print("=" * 60)

    saved_features = np.load(
        FEATURES_PATH,
        mmap_mode="r"
    )

    print(
        f"Final SWT shape : "
        f"{saved_features.shape}"
    )

    print(
        f"Final SWT dtype : "
        f"{saved_features.dtype}"
    )

    print(
        f"Features saved  : "
        f"{FEATURES_PATH}"
    )

    print(
        f"Metadata saved  : "
        f"{METADATA_PATH}"
    )

    print()

    expected_shape = (
        n_epochs,
        n_channels,
        config.level + 1,
        valid_len,
    )

    assert saved_features.shape == expected_shape, (
        f"Unexpected output shape: "
        f"{saved_features.shape}; "
        f"expected {expected_shape}"
    )

    assert saved_features.dtype == np.float32

    print(
        "Shape verification : PASSED"
    )

    print(
        "Dtype verification : PASSED"
    )

    print()
    print("=" * 60)
    print("SWT EXTRACTION COMPLETE")
    print("=" * 60)