"""
preprocessing/check_psd.py

Quick one-off check: plot the power spectral density (PSD) of a recording
to see what filtering, if any, was already applied at the source -- before
we design preprocessing/filtering.py. Not part of the main pipeline.

Usage:
    python preprocessing/check_psd.py --subject Subject00 --session 1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from load_eeg import load_recording  # reuse the loader from step 1

DATA_DIR = Path("data/raw/eegmat")


def main():
    parser = argparse.ArgumentParser(description="Plot PSD of one EEGMAT recording.")
    parser.add_argument("--subject", default="Subject00")
    parser.add_argument("--session", default="1", choices=["1", "2"])
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    raw = load_recording(args.subject, args.session, data_dir=Path(args.data_dir))
    raw.pick(["eeg"])  # drop ECG / reference channel for this check

    psd = raw.compute_psd(fmin=0.5, fmax=100, picks="eeg", verbose="ERROR")
    fig = psd.plot(show=False)
    out_path = Path(f"{args.subject}_{args.session}_psd.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved PSD plot to {out_path}")

    # Also print a few numeric summary points so it's easy to paste back
    freqs = psd.freqs
    data = psd.get_data().mean(axis=0)  # average across channels
    for target_freq in [1, 5, 10, 20, 30, 40, 50, 60, 80]:
        idx = (abs(freqs - target_freq)).argmin()
        print(f"~{freqs[idx]:.1f} Hz: mean power = {data[idx]:.3e}")


if __name__ == "__main__":
    main()