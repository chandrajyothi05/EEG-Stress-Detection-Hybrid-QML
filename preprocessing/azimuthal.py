preprocessing/azimuthal.py

Azimuthal Projection branch, following Mane & Shinde (StressNet, 2023):
theta/alpha/beta band power per electrode -> azimuthal equidistant
projection -> interpolated 2D grid per band -> stacked as a 3-channel
"RGB" image -> resized to 224x224 for pretrained EfficientNet-B0.

Requires exact channel names (see module-level CH_NAMES) matching the
column order used in eeg_denoised.npy.
"""

from pathlib import Path
import numpy as np
import mne
from scipy.signal import welch
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# !! MUST match the exact channel order in eeg_denoised.npy - fill this in
# after running the ch_names check above.
# ---------------------------------------------------------------------------
python
# Raw channel names as they appear in eeg_denoised.npy (with "EEG " prefix
# and old-style T3/T4/T5/T6 naming)
RAW_CH_NAMES = [
    'EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8',
    'EEG T3', 'EEG T4', 'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6',
    'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz',
]

# Map old 10-20 names to the modern names MNE's standard_1020 montage expects
_OLD_TO_MODERN = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def clean_channel_name(raw_name: str) -> str:
    """Strip 'EEG ' prefix and remap old-style temporal channel names."""
    name = raw_name.replace("EEG ", "").strip()
    return _OLD_TO_MODERN.get(name, name)


CH_NAMES = [clean_channel_name(ch) for ch in RAW_CH_NAMES]

BANDS = {
    "theta": (4, 7),
    "alpha": (8, 12),
    "beta": (13, 30),
}

GRID_SIZE = 32  # interpolation resolution before final resize to 224x224
FS = 500.0


def get_2d_electrode_positions(ch_names: list) -> np.ndarray:
    """
    Uses MNE's standard 10-20 montage to get 3D positions, then applies
    azimuthal equidistant projection to flatten them to 2D -- same
    approach as EEGLearn (Bashivan et al.) and consistent with the
    "azimuthal projection" described in the StressNet paper.

    Returns
    -------
    np.ndarray, shape (n_channels, 2)
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    montage_positions = montage.get_positions()["ch_pos"]

    missing = [ch for ch in ch_names if ch not in montage_positions]
    if missing:
        raise ValueError(
            f"Channels not found in standard_1020 montage: {missing}. "
            "Check for naming mismatches (e.g. 'T3' vs 'T7')."
        )

    xyz = np.array([montage_positions[ch] for ch in ch_names])  # (n_ch, 3)

    # Azimuthal equidistant projection (EEGLearn-style):
    # project each 3D point onto a 2D plane based on its angle from vertical.
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))  # polar angle from top of head
    phi = np.arctan2(y, x)                     # azimuthal angle

    proj_x = theta * np.cos(phi)
    proj_y = theta * np.sin(phi)

    return np.stack([proj_x, proj_y], axis=1)  # (n_ch, 2)


def band_power(signal: np.ndarray, fs: float, band: tuple) -> float:
    """
    Average power in a frequency band via Welch's method.
    signal: 1D array (one channel, one epoch).
    """
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return psd[band_mask].mean()


def epoch_band_powers(epoch: np.ndarray, fs: float = FS) -> dict:
    """
    epoch: (n_channels, n_samples)
    Returns dict: {band_name: np.ndarray of shape (n_channels,)}
    """
    n_channels = epoch.shape[0]
    powers = {band: np.zeros(n_channels) for band in BANDS}
    for ch in range(n_channels):
        for band_name, band_range in BANDS.items():
            powers[band_name][ch] = band_power(epoch[ch], fs, band_range)
    return powers


def interpolate_to_grid(values: np.ndarray, positions_2d: np.ndarray,
                         grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    values: (n_channels,) -- e.g. theta power per electrode
    positions_2d: (n_channels, 2) -- from get_2d_electrode_positions
    Returns: (grid_size, grid_size) interpolated image, NaNs outside
    the electrode hull filled with 0.
    """
    grid_x, grid_y = np.mgrid[
        positions_2d[:, 0].min():positions_2d[:, 0].max():complex(grid_size),
        positions_2d[:, 1].min():positions_2d[:, 1].max():complex(grid_size),
    ]
    grid = griddata(positions_2d, values, (grid_x, grid_y), method="cubic", fill_value=0.0)
    return grid


def epoch_to_image(epoch: np.ndarray, positions_2d: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Full per-epoch pipeline: band powers -> 3 interpolated grids -> stacked
    as a 3-channel image.

    Returns
    -------
    np.ndarray, shape (3, grid_size, grid_size) -- (theta, alpha, beta) as channels
    """
    powers = epoch_band_powers(epoch, fs)
    channels = [interpolate_to_grid(powers[band], positions_2d) for band in ("theta", "alpha", "beta")]
    return np.stack(channels, axis=0)  # (3, grid_size, grid_size)


if __name__ == "__main__":
    if CH_NAMES is None:
        raise RuntimeError("Set CH_NAMES at the top of this file first (see instructions).")

    positions_2d = get_2d_electrode_positions(CH_NAMES)
    print(f"Electrode 2D positions shape: {positions_2d.shape}")

    denoised = np.load("data/processed/eeg_denoised.npy", mmap_mode="r")
    one_epoch = np.asarray(denoised[0])

    image = epoch_to_image(one_epoch, positions_2d)
    print(f"Single-epoch image shape: {image.shape} (expect (3, {GRID_SIZE}, {GRID_SIZE}))")
    print(f"Min: {image.min():.4f}, Max: {image.max():.4f}")
