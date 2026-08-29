"""
Fractional Adaptive Superlet Transform (FASLT)
================================================
Implementation of the method from:
Bârzan, Moca, Ichim, Mureșan (2020) "Fractional Superlets", EUSIPCO 2020.

This follows eqs. (6)-(10) for the base superlet transform and
eqs. (13)-(15) for the fractional / adaptive fractional extension.

Designed to slot into a pipeline like:
    raw EEG -> filtering -> SWT denoising -> **FASLT** -> ConvMixer -> ...

Usage
-----
    from faslt import faslt_scalogram

    # x: 1D numpy array, single-channel EEG segment (after SWT denoising)
    freqs, L = faslt_scalogram(
        x, fs=1024,
        freqs=np.linspace(4, 40, 60),   # frequency axis you want in the output
        c1=3,
        order_min=1, order_max=11,
    )
    # L.shape == (len(freqs), len(x))  -> this is your time-frequency "image"
"""
import time
import sys
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from multiprocessing import Pool, cpu_count

# Allow Python to find faslt.py in the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

_kernel_cache = {}
# ---------------------------------------------------------------------------
# 1. Child wavelet (eq. 8) -- Morlet wavelet with omega_m normalized to 1
# ---------------------------------------------------------------------------
def _morlet_child_wavelet(c_m: float, omega: float, fs: float,
                           n_std: float = 5.0) -> np.ndarray:
    key = (round(c_m, 6), round(omega, 6), fs, n_std)
    if key in _kernel_cache:
        return _kernel_cache[key]

    sigma_t = (2 * np.pi * c_m) / (5.0 * omega)
    half_len = int(np.ceil(n_std * sigma_t * fs))
    half_len = max(half_len, 1)
    t = np.arange(-half_len, half_len + 1) / fs

    amp = (5.0 * omega) / (c_m * (2 * np.pi) ** 1.5)
    kernel = amp * np.exp(-0.5 * ((5.0 * omega * t) / (2 * np.pi * c_m)) ** 2) \
                 * np.exp(1j * omega * t)

    _kernel_cache[key] = kernel
    return kernel

# ---------------------------------------------------------------------------
# 2. R_x(c_i; t, omega) -- eq. (10): convolution of signal with child wavelet
# ---------------------------------------------------------------------------
def _R(x: np.ndarray, c_i: float, omega: float, fs: float) -> np.ndarray:
    """
    R_x(c_i; t, omega) = sqrt(2) * (x convolved with psi(c_i; ., omega))

    Returns a complex array the same length as x (same-length convolution,
    centered kernel -> 'same' mode keeps time alignment with x).
    """
    kernel = _morlet_child_wavelet(c_i, omega, fs)
    conv = fftconvolve(x, kernel, mode="same")
    return np.sqrt(2.0) * conv


# ---------------------------------------------------------------------------
# 3. Fractional Superlet Transform at a single frequency -- eq. (14)
# ---------------------------------------------------------------------------
def fslt_single_freq(x: np.ndarray, omega: float, fs: float,
                      c1: float, order_f: float) -> np.ndarray:
    """
    Computes FSLT_{x,c1,o_f}(t, omega) for one frequency omega and one
    (possibly fractional) order o_f, per eq. (14):

        FSLT = [ R_x(c1*(o_i+1); t, omega)^alpha * prod_{i=1..o_i} R_x(c1*i; t, omega) ]^(1/o_f)

    where o_f = o_i + alpha,  o_i = floor(o_f) (o_i >= 1), alpha in [0,1)

    Returns complex array (same length as x). Take |.|^2 for the scalogram.
    """
    order_f = max(order_f, 1.0)  # order must be >= 1 (o=1 reduces to CWT)
    o_i = int(np.floor(order_f))
    alpha = order_f - o_i

    # product of the integer-order terms R_x(c1*i; t, omega), i = 1..o_i
    prod = np.ones_like(x, dtype=complex)
    for i in range(1, o_i + 1):
        prod = prod * _R(x, c1 * i, omega, fs)

    if alpha > 1e-9:
        # weighted extra term for the fractional part (the (o_i+1)-th wavelet)
        r_extra = _R(x, c1 * (o_i + 1), omega, fs)
        combined = prod * (r_extra ** alpha)
    else:
        combined = prod

    return combined ** (1.0 / order_f)


# ---------------------------------------------------------------------------
# 4. Adaptive fractional order across the frequency axis -- eq. (15)
# ---------------------------------------------------------------------------
def _adaptive_order(freqs: np.ndarray, order_min: float, order_max: float) -> np.ndarray:
    """
    o_f(omega) = o_min + (o_max - o_min) * (omega - omega_min) / (omega_max - omega_min)

    Linear ramp of order across the given frequency axis (eq. 15, the
    fractional/continuous version -- no rounding, unlike the classic ASLT
    eq. 12 which uses `round(...)` and causes banding).
    """
    f_min, f_max = freqs.min(), freqs.max()
    if f_max == f_min:
        return np.full_like(freqs, order_min, dtype=float)
    return order_min + (order_max - order_min) * (freqs - f_min) / (f_max - f_min)


# ---------------------------------------------------------------------------
# 5. Full FASLT scalogram over a bank of frequencies
# ---------------------------------------------------------------------------
def faslt_scalogram(x: np.ndarray, fs: float, freqs: np.ndarray,
                     c1: float = 3.0, order_min: float = 1.0, order_max: float = 11.0):
    """
    Computes the Fractional Adaptive Superlet Transform scalogram of a
    1D signal x across the frequency bank `freqs`.

    Parameters
    ----------
    x         : 1D numpy array, single-channel time series (post SWT denoising)
    fs        : sampling rate in Hz
    freqs     : 1D array of frequencies (Hz) to compute (your desired freq axis)
    c1        : base number of cycles (paper recommends 1-3)
    order_min : superlet order at the lowest frequency in `freqs`
    order_max : superlet order at the highest frequency in `freqs`

    Returns
    -------
    freqs : the same freq axis passed in (for convenience)
    L     : 2D real array, shape (len(freqs), len(x)) -- the power scalogram,
            L(t, omega) = |FSLT(t, omega)|^2   (eq. 11)
    """
    freqs = np.asarray(freqs, dtype=float)
    orders = _adaptive_order(freqs, order_min, order_max)

    L = np.zeros((len(freqs), len(x)), dtype=float)
    for k, (f, o_f) in enumerate(zip(freqs, orders)):
        omega = 2 * np.pi * f
        fslt = fslt_single_freq(x, omega, fs, c1, o_f)
        L[k, :] = np.abs(fslt) ** 2

    return freqs, L


# ---------------------------------------------------------------------------
# 6. Multi-channel convenience wrapper (for building ConvMixer input tensors)
# ---------------------------------------------------------------------------
def faslt_multichannel(X: np.ndarray, fs: float, freqs: np.ndarray,
                        c1: float = 3.0, order_min: float = 1.0, order_max: float = 11.0):
    """
    X : 2D array, shape (n_channels, n_samples) -- an EEG epoch, post SWT.

    Returns
    -------
    freqs : frequency axis
    stack : 3D array, shape (n_channels, len(freqs), n_samples)
            Ready to be treated as a (C, H, W)-style tensor for ConvMixer,
            with C = EEG channels, H = frequency bins, W = time samples.
    """
    n_channels, n_samples = X.shape
    stack = np.zeros((n_channels, len(freqs), n_samples), dtype=float)
    for ch in range(n_channels):
        _, L = faslt_scalogram(X[ch], fs, freqs, c1, order_min, order_max)
        stack[ch] = L
    return freqs, stack


if __name__ == "__main__":
    # quick smoke test: a 20 Hz tone in noise
    fs = 250
    t = np.arange(0, 2, 1 / fs)
    x = np.sin(2 * np.pi * 20 * t) + 0.3 * np.random.randn(len(t))

    freqs = np.linspace(4, 40, 40)
    freqs, L = faslt_scalogram(x, fs, freqs, c1=3, order_min=1, order_max=11)
    print("Scalogram shape:", L.shape)
    print("Peak frequency bin:", freqs[np.argmax(L.mean(axis=1))], "Hz (expected ~20 Hz)")