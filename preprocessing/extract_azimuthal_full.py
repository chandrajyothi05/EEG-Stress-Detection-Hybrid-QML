"""
preprocessing/extract_azimuthal_full.py

Runs the azimuthal projection pipeline across all 8604 epochs.
Output is small (~105MB at 32x32x3xfloat32), so no memmap needed --
straightforward in-memory loop with progress printing.
"""

from pathlib import Path
import numpy as np

from preprocessing.azimuthal import get_2d_electrode_positions, epoch_to_image, CH_NAMES

IN_PATH = Path("data/processed/eeg_denoised.npy")
META_PATH = Path("data/processed/eeg_denoised_metadata.npz")
OUT_PATH = Path("data/processed/azimuthal_images.npy")

if __name__ == "__main__":
    positions_2d = get_2d_electrode_positions(CH_NAMES)
    denoised = np.load(IN_PATH, mmap_mode="r")
    metadata = np.load(META_PATH, allow_pickle=True)

    n_epochs = denoised.shape[0]
    images = np.zeros((n_epochs, 3, 32, 32), dtype=np.float32)

    for i in range(n_epochs):
        epoch = np.asarray(denoised[i])
        images[i] = epoch_to_image(epoch, positions_2d)
        if (i + 1) % 500 == 0 or i == n_epochs - 1:
            print(f"Processed {i + 1}/{n_epochs}")

    np.save(OUT_PATH, images)
    print(f"Saved {OUT_PATH}, shape {images.shape}")

    # NaN/Inf check
    print(f"NaN check: {'FAILED' if np.isnan(images).any() else 'PASSED'}")
    print(f"Inf check: {'FAILED' if np.isinf(images).any() else 'PASSED'}")
    print(f"Global min: {images.min():.4f}, max: {images.max():.4f}")

    # y/subject_ids already saved in eeg_denoised_metadata.npz -- reuse directly,
    # no need to duplicate since alignment is guaranteed (same source, same order)