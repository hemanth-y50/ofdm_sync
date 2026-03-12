"""
cfo_estimation.py
=================
Carrier Frequency Offset (CFO) Estimation for OFDM Systems.

Three-stage CFO correction pipeline:
  1. Fractional CFO  – Schmidl–Cox phase-based estimate (|ε| ≤ 0.5 subcarrier)
  2. Fine CFO        – Moose algorithm using repeated preamble blocks
  3. Integer CFO     – Pilot-tone-based residual integer offset search

References:
    Schmidl & Cox (1997); Moose (1994); van de Beek et al. (1997).
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Fractional CFO  (Schmidl–Cox)
# ---------------------------------------------------------------------------

def estimate_fractional_cfo(rx_signal: np.ndarray,
                             N: int,
                             timing_offset: int) -> dict:
    """
    Estimate fractional CFO from the angle of the Schmidl–Cox correlator.

    The preamble consists of two identical halves of length N/2.
    The phase rotation between them carries the fractional CFO:
        ε_frac = angle(P) / π   (normalised to subcarrier spacing)

    Parameters
    ----------
    rx_signal      : complex baseband received signal
    N              : FFT size
    timing_offset  : sample index of preamble start (from timing sync)

    Returns
    -------
    dict: 'cfo_normalized' (ε, fraction of Δf), 'cfo_hz' requires Δf externally
    """
    L = N // 2
    d = timing_offset

    seg1 = rx_signal[d: d + L]
    seg2 = rx_signal[d + L: d + 2 * L]

    P = np.dot(seg2, seg1.conj())
    cfo_normalized = np.angle(P) / np.pi   # ε ∈ (-1, 1) × 0.5 Δf

    return {
        'cfo_normalized': cfo_normalized,
        'correlator_phase': np.angle(P),
        'P': P,
    }


# ---------------------------------------------------------------------------
# 2. Fine CFO  (Moose Algorithm)
# ---------------------------------------------------------------------------

def estimate_fine_cfo_moose(rx_signal: np.ndarray,
                             N: int,
                             timing_offset: int,
                             cp_len: int = 0) -> dict:
    """
    Moose (1994) fine CFO estimator using two repeated N-point blocks.

    The two consecutive identical OFDM blocks in the preamble give:
        ε_fine = angle(Σ r2[k]·r1*[k]) / (2π)

    This has a range of ±0.5 subcarrier spacings with higher accuracy
    than the Schmidl–Cox fractional estimate.

    Parameters
    ----------
    rx_signal     : complex received signal
    N             : FFT size (one block length, CP already stripped)
    timing_offset : sample index after CP removal

    Returns
    -------
    dict: 'cfo_normalized' normalised CFO estimate
    """
    d = timing_offset

    # Each Moose block = CP + N samples; strip CP before correlating
    block1_start = d + cp_len
    block2_start = d + (N + cp_len) + cp_len   # second symbol starts after first full symbol

    if block2_start + N > len(rx_signal):
        raise ValueError("Not enough samples for Moose estimator.")

    block1 = rx_signal[block1_start: block1_start + N]
    block2 = rx_signal[block2_start: block2_start + N]

    cross = np.dot(block2, block1.conj())
    cfo_normalized = np.angle(cross) / (2 * np.pi)

    return {
        'cfo_normalized': cfo_normalized,
        'cross_product': cross,
    }


# ---------------------------------------------------------------------------
# 3. Integer CFO  (Pilot-based)
# ---------------------------------------------------------------------------

def estimate_integer_cfo(rx_freq: np.ndarray,
                          pilot_indices: list,
                          pilot_values: np.ndarray,
                          search_range: int = 10) -> dict:
    """
    Integer CFO estimation by exhaustive search over pilot subcarriers.

    For each candidate integer offset k ∈ [-search_range, search_range],
    correlate the received pilots (shifted by k) with known pilots and
    find the shift that maximises the metric.

    Parameters
    ----------
    rx_freq       : FFT output of received OFDM symbol (complex, length N)
    pilot_indices : list of pilot subcarrier indices
    pilot_values  : known complex pilot symbols (same length as pilot_indices)
    search_range  : ±maximum integer shift to test

    Returns
    -------
    dict: 'integer_cfo' in subcarriers, 'metric_profile'
    """
    N = len(rx_freq)
    pilot_values = np.asarray(pilot_values)
    metric = np.zeros(2 * search_range + 1)

    for i, k in enumerate(range(-search_range, search_range + 1)):
        shifted_indices = [(p + k) % N for p in pilot_indices]
        rx_pilots = rx_freq[shifted_indices]
        metric[i] = np.abs(np.dot(rx_pilots, pilot_values.conj()))

    best_idx = int(np.argmax(metric))
    integer_cfo = best_idx - search_range

    return {
        'integer_cfo': integer_cfo,
        'metric_profile': metric,
        'search_range': search_range,
    }


# ---------------------------------------------------------------------------
# 4. CFO Correction
# ---------------------------------------------------------------------------

def apply_cfo_correction(rx_signal: np.ndarray,
                          cfo_normalized: float,
                          N: int,
                          cp_len: int) -> np.ndarray:
    """
    Correct CFO by multiplying the time-domain signal with a counter-rotating
    complex exponential.

    Parameters
    ----------
    rx_signal      : complex received signal
    cfo_normalized : total estimated CFO in units of subcarrier spacing
    N              : FFT size
    cp_len         : cyclic prefix length

    Returns
    -------
    cfo_corrected  : complex signal with CFO removed
    """
    symbol_len = N + cp_len
    n = np.arange(len(rx_signal))
    # Phase ramp: 2π·ε·n / N  (ε normalised to Δf)
    correction = np.exp(-1j * 2 * np.pi * cfo_normalized * n / N)
    return rx_signal * correction


# ---------------------------------------------------------------------------
# 5. CFO Estimation Error Metrics
# ---------------------------------------------------------------------------

def cfo_estimation_mse(true_cfo: float,
                        estimated_cfos: np.ndarray) -> dict:
    """
    Compute MSE and RMSE for CFO estimates across Monte-Carlo trials.

    Parameters
    ----------
    true_cfo        : ground-truth CFO (normalised)
    estimated_cfos  : array of estimates from multiple trials

    Returns
    -------
    dict: 'mse', 'rmse', 'bias', 'std'
    """
    errors = estimated_cfos - true_cfo
    mse = float(np.mean(errors ** 2))
    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'bias': float(np.mean(errors)),
        'std': float(np.std(errors)),
    }
