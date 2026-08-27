"""
preprocessing/swt_denoise.py

Replaces the old subband-stacking swt_features.py for this architecture.
This performs actual SWT denoising -- decompose each channel, soft-threshold
the detail coefficients (VisuShrink), and reconstruct -- producing a single
cleaned signal per channel, same shape as the input. Both FASLT and
Azimuthal Projection branches should load THIS output, not the old
6-subband stack.

Order in the full pipeline (per your diagram):
    segmentation.py (filter -> epoch -> normalize) -> THIS STEP -> branches
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pywt

IN_PATH = Path("data/processed/eeg_epochs.npz")
OUT_PATH = Path("data/processed/eeg_denoised.npy")
META_OUT_PATH = Path("data/processed/eeg_denoised_metadata.npz")


@dataclass
class DenoiseConfig:
    wavelet: str = "db4"
    level: int = 4
    threshold_mode: str = "soft"


def swt_denoise_channel(sig: np.ndarray, config: DenoiseConfig = DenoiseConfig()) -> np.ndarray:
    n = len(sig)
    factor = 2 ** config.level
    pad = (-n) % factor
    sig_padded = np.pad(sig, (0, pad), mode="reflect") if pad else sig

    coeffs = pywt.swt(sig_padded, config.wavelet, level=config.level, trim_approx=True)
    approx, details = coeffs[0], coeffs[1:]

    finest_detail = details[-1]
    sigma = np.median(np.abs(finest_detail)) / 0.6745
    uni_thresh = sigma * np.sqrt(2 * np.log(len(sig_padded)))

    denoised_details = [pywt.threshold(d, uni_thresh, mode=config.threshold_mode) for d in details]
    denoised = pywt.iswt([approx] + denoised_details, config.wavelet)
    return denoised[:n]


def swt_denoise_epoch(X: np.ndarray, config: DenoiseConfig = DenoiseConfig()) -> np.ndarray:
    """X: (n_channels, n_samples) -> same shape, denoised per channel."""
    return np.stack([swt_denoise_channel(X[ch], config) for ch in range(X.shape[0])])


if __name__ == "__main__":
    data = np.load(IN_PATH, allow_pickle=True)
    X, y, subject_ids = data["X"], data["y"], data["subject_ids"]
    print(f"Input X shape: {X.shape}")

    config = DenoiseConfig()

    n_epochs = X.shape[0]
    batch_size = 200

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    denoised_memmap = np.lib.format.open_memmap(
        OUT_PATH, mode="w+", dtype=np.float32, shape=X.shape
    )

    for start in range(0, n_epochs, batch_size):
        end = min(start + batch_size, n_epochs)
        for i in range(start, end):
            denoised_memmap[i] = swt_denoise_epoch(X[i], config)
        print(f"Denoised epochs {start}-{end-1} / {n_epochs}")

    denoised_memmap.flush()
    print(f"Saved denoised data to {OUT_PATH}")

    np.savez(META_OUT_PATH, y=y, subject_ids=subject_ids)
    print(f"Saved metadata to {META_OUT_PATH}")

    print(f"Final denoised shape: {denoised_memmap.shape}")