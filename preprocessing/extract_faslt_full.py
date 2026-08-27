# extract_faslt_full.py

import time
import numpy as np
from multiprocessing import Pool, cpu_count
from faslt import faslt_multichannel

INPUT_PATH = "data/processed/eeg_denoised.npy"
OUTPUT_PATH = "data/processed/faslt_features.npy"
fs = 500.0
freqs = np.linspace(1, 40, 40)
N_WORKERS = max(cpu_count() - 1, 1)

def process_epoch(args):
    idx, epoch = args
    _, tensor = faslt_multichannel(epoch, fs, freqs, c1=3.0, order_min=1.0, order_max=11.0)
    return idx, tensor

if __name__ == "__main__":
    denoised = np.load(INPUT_PATH, mmap_mode="r")
    n_epochs, n_channels, n_samples = denoised.shape
    n_freqs = len(freqs)

    output = np.lib.format.open_memmap(
        OUTPUT_PATH, mode="w+", dtype=np.float64,
        shape=(n_epochs, n_channels, n_freqs, n_samples)
    )

    tasks = ((i, np.array(denoised[i])) for i in range(n_epochs))

    start = time.time()
    with Pool(N_WORKERS) as pool:
        for count, (idx, tensor) in enumerate(pool.imap_unordered(process_epoch, tasks, chunksize=4)):
            output[idx] = tensor
            if (count + 1) % 200 == 0:
                elapsed = time.time() - start
                rate = (count + 1) / elapsed
                remaining = (n_epochs - (count + 1)) / rate
                print(f"[{count+1}/{n_epochs}] elapsed={elapsed/60:.1f}min "
                      f"est. remaining={remaining/60:.1f}min")

    output.flush()
    print("Done:", OUTPUT_PATH)