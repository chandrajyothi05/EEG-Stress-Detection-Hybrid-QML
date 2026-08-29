"""
preprocessing/log_normalize_faslt.py

Compresses FASLT's wide dynamic range (0 to ~687,851 seen in your data)
using log1p, batch-wise to avoid loading the full 3+ GB array into RAM
at once. Run AFTER verify_faslt.py confirms no NaN/Inf.
"""

from pathlib import Path
import numpy as np

IN_PATH = Path("data/processed/faslt_features.npy")
OUT_PATH = Path("data/processed/faslt_features_log.npy")

if __name__ == "__main__":
    faslt = np.load(IN_PATH, mmap_mode="r")
    n_epochs = faslt.shape[0]
    batch_size = 500

    log_features = np.lib.format.open_memmap(
        OUT_PATH, mode="w+", dtype=np.float32, shape=faslt.shape
    )

    for start in range(0, n_epochs, batch_size):
        end = min(start + batch_size, n_epochs)
        batch = np.asarray(faslt[start:end])
        log_features[start:end] = np.log1p(batch)
        print(f"Log-normalized epochs {start}-{end} / {n_epochs}")

    log_features.flush()
    print(f"Saved to {OUT_PATH}")

    # quick sanity check
    sample = np.asarray(log_features[:100])
    print(f"Post-log1p range (first 100 epochs): min={sample.min():.4f}, max={sample.max():.4f}")