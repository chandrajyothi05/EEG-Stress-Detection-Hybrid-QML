# test_faslt.py

import time
import numpy as np
import time
import sys
from pathlib import Path

# Allow Python to find faslt.py in the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from faslt import faslt_multichannel
# --- Load already-denoised EEG (memory-mapped, not loaded fully into RAM) ---
denoised = np.load(
    "data/processed/eeg_denoised.npy",
    mmap_mode="r"
)

fs = 500.0

# Starting frequency bank — revisit before final experiment
freqs = np.linspace(1, 40, 40)

# --- Take ONE epoch only ---
one_epoch = np.array(denoised[0])  # shape should be (19, 1000)
print("Input shape:", one_epoch.shape)

# --- Run FASLT and time it ---
start = time.time()

_, tensor = faslt_multichannel(
    one_epoch,
    fs,
    freqs,
    c1=3.0,
    order_min=1.0,
    order_max=11.0
)

elapsed = time.time() - start

print("FASLT output shape:", tensor.shape)   # expect (19, 40, 1000)
print(f"Time for one epoch: {elapsed:.2f} seconds")

# --- Extrapolate to full dataset ---
n_epochs_total = 8604
estimated_min = elapsed * n_epochs_total / 60
print(f"Estimated time for all {n_epochs_total} epochs: {estimated_min:.1f} minutes")

# --- Sanity checks on output values ---
print("Min:", tensor.min())
print("Max:", tensor.max())
print("Mean:", tensor.mean())
print("NaN:", np.isnan(tensor).any())
print("Inf:", np.isinf(tensor).any())
import faslt
print("Cache entries after one epoch:", len(faslt._kernel_cache))