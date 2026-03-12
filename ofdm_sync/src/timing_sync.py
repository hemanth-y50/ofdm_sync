"""
timing_sync.py
==============
Schmidl–Cox Timing Synchronization for OFDM Systems.

The Schmidl–Cox algorithm exploits the cyclic prefix (CP) and a training
preamble with two identical halves to estimate the coarse timing offset.

Reference:
    T. M. Schmidl and D. C. Cox, "Robust frequency and timing synchronization
    for OFDM," IEEE Trans. Commun., vol. 45, no. 12, pp. 1613–1621, Dec. 1997.
"""

import numpy as np
from scipy.signal import correlate


def schmidl_cox_timing(rx_signal: np.ndarray, N: int, cp_len: int) -> dict:
    """
    Estimate timing offset using the Schmidl–Cox metric.

    Parameters
    ----------
    rx_signal : np.ndarray (complex)
        Received baseband signal.
    N : int
        OFDM FFT size (number of subcarriers).
    cp_len : int
        Cyclic prefix length in samples.

    Returns
    -------
    dict with keys:
        'timing_offset'  : int   – estimated start of OFDM symbol
        'metric'         : np.ndarray – SC timing metric over all lags
        'plateau_start'  : int   – start of the flat plateau
        'plateau_end'    : int   – end of the flat plateau
    """
    L = N // 2  # half-preamble length
    n_samples = len(rx_signal)

    P = np.zeros(n_samples, dtype=complex)
    R = np.zeros(n_samples, dtype=float)

    # Sliding-window correlator
    for d in range(n_samples - N):
        # Cross-correlation of two L-length halves separated by L samples
        seg1 = rx_signal[d:d + L]
        seg2 = rx_signal[d + L:d + 2 * L]
        P[d] = np.dot(seg2, seg1.conj())
        R[d] = np.real(np.dot(seg2, seg2.conj()))

    # SC metric: |P(d)|^2 / R(d)^2  (plateau around true timing)
    with np.errstate(divide='ignore', invalid='ignore'):
        metric = np.where(R > 0, (np.abs(P) ** 2) / (R ** 2 + 1e-12), 0.0)

    # Plateau detection: find the region where metric > 0.95 * max
    threshold = 0.95 * np.max(metric)
    plateau_mask = metric >= threshold
    plateau_indices = np.where(plateau_mask)[0]

    if len(plateau_indices) == 0:
        timing_offset = int(np.argmax(metric))
        plateau_start = timing_offset
        plateau_end = timing_offset
    else:
        plateau_start = int(plateau_indices[0])
        plateau_end = int(plateau_indices[-1])
        # The SC plateau spans CP samples ending at the true symbol start.
        # Use the END of the plateau as the timing estimate.
        timing_offset = plateau_end

    return {
        'timing_offset': timing_offset,
        'metric': metric,
        'plateau_start': plateau_start,
        'plateau_end': plateau_end,
    }


def cross_correlation_timing(rx_signal: np.ndarray,
                              known_preamble: np.ndarray) -> dict:
    """
    Fine timing using cross-correlation with a known preamble sequence.

    Parameters
    ----------
    rx_signal      : complex received signal
    known_preamble : known transmitted preamble (time-domain)

    Returns
    -------
    dict with 'timing_offset' and 'correlation_magnitude'
    """
    corr = correlate(rx_signal, known_preamble, mode='full')
    corr_mag = np.abs(corr)
    # Align: full-mode offset adjustment
    lag_offset = len(known_preamble) - 1
    timing_offset = int(np.argmax(corr_mag)) - lag_offset

    return {
        'timing_offset': max(0, timing_offset),
        'correlation_magnitude': corr_mag,
    }


def estimate_timing_error_rate(true_offset: int,
                                estimated_offset: int,
                                N: int) -> float:
    """
    Normalised timing error (fraction of FFT size).
    """
    return abs(true_offset - estimated_offset) / N
