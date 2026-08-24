"""
preprocessing/load_eeg.py

Phase 1, Step 1: load a single EEGMAT recording with MNE, verify the header
info against the dataset documentation, and plot a short segment.

This does NOT filter, epoch, or label the data yet -- it only confirms we
can read the files correctly and that the header matches what the EEGMAT
README claims (channel count, sampling frequency, duration). Everything
here is deliberately explicit rather than assumed, since some of the
README's wording (e.g. the filtering already applied) is ambiguous and
should be checked against the actual signal.

Usage:
    python preprocessing/load_eeg.py --subject Subject00 --session 1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne

# ---------------------------------------------------------------------------
# Config (move to preprocessing/config.py once more scripts need these)
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw/eegmat")
PLOT_SECONDS = 10.0  # how much signal to preview


def load_recording(subject: str, session: str, data_dir: Path = DATA_DIR) -> mne.io.Raw:
    """
    Load one EEGMAT EDF recording.

    Parameters
    ----------
    subject : str
        e.g. "Subject00"
    session : str
        "1" for background/resting EEG, "2" for mental arithmetic task EEG.
    data_dir : Path
        Directory containing the EEGMAT .edf files.

    Returns
    -------
    mne.io.Raw
        The loaded raw EEG object (not yet filtered or epoched).
    """
    if session not in ("1", "2"):
        raise ValueError(f"session must be '1' (background) or '2' (task), got {session!r}")

    edf_path = data_dir / f"{subject}_{session}.edf"
    if not edf_path.exists():
        raise FileNotFoundError(
            f"Could not find {edf_path}. Confirm the EEGMAT files are under {data_dir}."
        )

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    return raw


def inspect_recording(raw: mne.io.Raw, subject: str, session: str) -> dict:
    """
    Print and return key header info for sanity-checking against the
    EEGMAT documentation (23-channel Neurocom system, 500 Hz claimed
    sampling rate, ~60 s recordings).
    """
    info = {
        "subject": subject,
        "session": session,
        "n_channels": len(raw.ch_names),
        "channel_names": raw.ch_names,
        "sfreq": raw.info["sfreq"],
        "n_samples": raw.n_times,
        "duration_sec": raw.n_times / raw.info["sfreq"],
    }

    print(f"--- {subject}_{session}.edf ---")
    print(f"Channels ({info['n_channels']}): {info['channel_names']}")
    print(f"Sampling frequency: {info['sfreq']} Hz")
    print(f"Number of samples: {info['n_samples']}")
    print(f"Duration: {info['duration_sec']:.2f} s")

    return info


def plot_preview(raw: mne.io.Raw, subject: str, session: str, seconds: float = PLOT_SECONDS):
    """
    Plot the first `seconds` of every channel using matplotlib directly
    (avoids relying on MNE's interactive browser, which doesn't work well
    non-interactively / in scripts).
    """
    sfreq = raw.info["sfreq"]
    n_preview_samples = int(seconds * sfreq)
    data, times = raw[:, :n_preview_samples]  # data shape: (n_channels, n_samples)

    fig, ax = plt.subplots(figsize=(12, 8))
    # Offset each channel vertically so traces don't overlap
    offset_step = 3 * data.std()
    for i, ch_name in enumerate(raw.ch_names):
        ax.plot(times, data[i] + i * offset_step, linewidth=0.6, label=ch_name)

    ax.set_yticks([i * offset_step for i in range(len(raw.ch_names))])
    ax.set_yticklabels(raw.ch_names, fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{subject}_{session} — first {seconds:.0f}s")
    fig.tight_layout()

    out_path = Path(f"{subject}_{session}_preview.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved preview plot to {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Load and inspect one EEGMAT recording.")
    parser.add_argument("--subject", default="Subject00", help="e.g. Subject00")
    parser.add_argument("--session", default="1", choices=["1", "2"],
                         help="1 = background/resting, 2 = mental arithmetic task")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to data/raw/eegmat")
    args = parser.parse_args()

    raw = load_recording(args.subject, args.session, data_dir=Path(args.data_dir))
    inspect_recording(raw, args.subject, args.session)
    plot_preview(raw, args.subject, args.session)


if __name__ == "__main__":
    main()