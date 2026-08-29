import time
import numpy as np
from preprocessing.azimuthal import get_2d_electrode_positions, epoch_to_image, CH_NAMES

positions_2d = get_2d_electrode_positions(CH_NAMES)
denoised = np.load("data/processed/eeg_denoised.npy", mmap_mode="r")

one_epoch = np.asarray(denoised[0])

start = time.time()
image = epoch_to_image(one_epoch, positions_2d)
elapsed = time.time() - start

print(f"One epoch took {elapsed:.4f} s")
print(f"Estimated total time for 8604 epochs: {elapsed * 8604 / 60:.1f} minutes")