"""
preprocessing/swt_features.py

Phase 1, Step 5: Stationary Wavelet Transform (SWT) feature extraction.

Applied per-epoch, per-channel, independently -- consistent with how
normalization was done (per-epoch), so no information leaks across
epochs or channels during this step.

Wavelet choice: db4 (Daubechies-4), decomposition level 5.
  - The PSD check (see check_psd.py) showed the signal is already
    band-limited to ~30-40 Hz, with negligible content above that.
  - SWT level 5 on a 500 Hz signal decomposes down into the following
    approximate frequency bands per level:
        Level 1 detail: ~125-250 Hz
        Level 2 detail: ~62.5-125 Hz
        Level 3 detail: ~31.25-62.5 Hz
        Level 4 detail: ~15.6-31.25 Hz
        Level 5 detail: ~7.8-15.6 Hz
        Level 5 approximation: ~0-7.8 Hz
  - Since almost nothing lives above ~40 Hz in this data, levels 1-2
    mostly capture noise, while levels 3-5 (approx) cover the bands
    most relevant to EEG rhythms (beta down to delta/theta) and to
    stress/cognitive-load literature specifically.
  - Unlike DWT, SWT does not downsample between levels (it's the
    "stationary"/undecimated version), so every output subband has
    the same length as the input epoch. This avoids alignment
    headaches when stacking subbands as features.

Usage (as a module):
    from preprocessing.swt_features import extract_swt_features

    # X: (n_epochs, n_channels, n_samples) from segmentation.py
    features = extract_swt_features(X)
"""

from dataclasses import dataclass

import numpy as np
import pywt


@dataclass
class SWTConfig:
    """SWT parameters -- shared by both MHA and Bi-LSTM branches."""
    wavelet: str = "db4"
    level: int = 5


def _next_valid_length(n_samples: int, level: int) -> int:
    """
    pywt.swt requires the signal length to be divisible by 2**level.
    Returns the largest length <= n_samples that satisfies this, so we
    can trim epochs to a valid length before transforming.
    """
    factor = 2 ** level
    return (n_samples // factor) * factor


def swt_single_channel(signal: np.ndarray, config: SWTConfig = SWTConfig()) -> np.ndarray:
    """
    Apply SWT to a single 1D signal (one channel of one epoch).

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    config : SWTConfig

    Returns
    -------
    np.ndarray, shape (level + 1, n_valid_samples)
        Subbands stacked as [detail_1, detail_2, ..., detail_level, approx_level].
        n_valid_samples may be slightly less than n_samples if the
        original length wasn't divisible by 2**level (see
        _next_valid_length) -- this is a fixed, deterministic trim,
        not data loss driven by content.
    """
    valid_len = _next_valid_length(len(signal), config.level)
    trimmed = signal[:valid_len]

    coeffs = pywt.swt(trimmed, wavelet=config.wavelet, level=config.level, trim_approx=True)
    # pywt.swt with trim_approx=True returns:
    #   [cA_level, cD_level, cD_level-1, ..., cD_1]
    # Reorder to [cD_1, cD_2, ..., cD_level, cA_level] for readability,
    # matching the module docstring's band ordering.
    cA_level = coeffs[0]
    details = coeffs[1:][::-1]  # now cD_1 ... cD_level
    stacked = np.stack(details + [cA_level], axis=0)  # (level+1, n_valid_samples)
    return stacked


def extract_swt_features(X: np.ndarray, config: SWTConfig = SWTConfig()) -> np.ndarray:
    """
    Apply SWT to every channel of every epoch.

    Parameters
    ----------
    X : np.ndarray, shape (n_epochs, n_channels, n_samples)
        Output of segmentation.py (already filtered + normalized).
    config : SWTConfig

    Returns
    -------
    np.ndarray, shape (n_epochs, n_channels, level + 1, n_valid_samples)
        Each channel is expanded into (level + 1) subbands.
    """
    n_epochs, n_channels, n_samples = X.shape
    valid_len = _next_valid_length(n_samples, config.level)

    out = np.empty((n_epochs, n_channels, config.level + 1, valid_len), dtype=X.dtype)

    for i in range(n_epochs):
        for c in range(n_channels):
            out[i, c] = swt_single_channel(X[i, c], config)

    return out


if __name__ == "__main__":
    # Quick manual check using the combined dataset from segmentation.py
    from pathlib import Path

    data = np.load(Path("data/processed/eeg_epochs.npz"), allow_pickle=True)
    X = data["X"]

    print(f"Input X shape: {X.shape}")

    config = SWTConfig()
    features = extract_swt_features(X[:5], config)  # just first 5 epochs as a sanity check
    print(f"SWT output shape (5-epoch sample): {features.shape}")
    print(f"Expected: (5, {X.shape[1]}, {config.level + 1}, ~{X.shape[2]})")