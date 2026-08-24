"""
preprocessing/filtering.py

Phase 1, Step 2: channel selection + light filtering.

Based on the PSD check (see check_psd.py) run on Subject00_1:
  - Strong power at 1, 5, 10 Hz rules out a genuine ~30 Hz high-pass.
  - A steep rolloff between 30-40 Hz plus a very deep, narrow dip at
    50 Hz indicates the source data is already low-pass filtered
    (~30-40 Hz) and notch-filtered (50 Hz mains).
  - Because of that, our own filtering here is deliberately light:
    mainly a gentle high-pass to remove DC / baseline drift, which
    nothing upstream appears to have addressed. We do NOT re-apply a
    hard low-pass or redesign the notch, since that work already
    appears to be done at the source.

If you run this against other subjects and see a materially different
PSD shape (e.g. clear 50 Hz mains noise, or content above 40 Hz), revisit
these defaults -- they are set from Subject00_1 only, not the whole
dataset. Verify on a few more subjects before trusting this everywhere.

Usage (as a module):
    from preprocessing.load_eeg import load_recording
    from preprocessing.filtering import select_eeg_channels, apply_filters

    raw = load_recording("Subject00", "1")
    raw = select_eeg_channels(raw)
    raw = apply_filters(raw)
"""

from dataclasses import dataclass

import mne


@dataclass
class FilterConfig:
    """Filtering parameters -- keep these here rather than scattered
    through the codebase so both the MHA and Bi-LSTM branches use
    identical values."""
    l_freq: float = 0.5     # high-pass cutoff (Hz) -- removes DC/baseline drift
    h_freq: float | None = None  # low-pass cutoff (Hz) -- None = don't re-filter;
                                   # source data already appears band-limited ~30-40Hz
    apply_notch: bool = False     # source data already appears notch-filtered at 50Hz;
                                   # set True only if you see mains noise on other subjects
    notch_freq: float = 50.0      # Ukraine mains frequency, if apply_notch is True


def select_eeg_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Keep only the 19 scalp EEG channels; drop the reference channel
    (A2-A1) and ECG. Modifies and returns raw (MNE's pick is in-place).
    """
    raw = raw.copy()
    raw.pick(picks="eeg")  # MNE identifies EEG-typed channels automatically
    return raw


def apply_filters(raw: mne.io.Raw, config: FilterConfig = FilterConfig()) -> mne.io.Raw:
    """
    Apply the configured high-pass (and optionally low-pass / notch)
    filtering to an EEG-only Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        Should already be restricted to EEG channels (see select_eeg_channels).
    config : FilterConfig
        Filtering parameters. Defaults are intentionally light -- see
        module docstring for why.

    Returns
    -------
    mne.io.Raw
        A new, filtered Raw object (original is not modified).
    """
    raw = raw.copy()

    raw.filter(l_freq=config.l_freq, h_freq=config.h_freq, picks="eeg", verbose="ERROR")

    if config.apply_notch:
        raw.notch_filter(freqs=config.notch_freq, picks="eeg", verbose="ERROR")

    return raw