"""
ofdm_utils.py
=============
OFDM Signal Generation, Channel Models, and Utility Functions.

Provides:
  • OFDM modulator / demodulator (with CP)
  • Schmidl–Cox preamble generator
  • AWGN + multipath + Doppler channel
  • SNR / EbN0 helpers
"""

import numpy as np
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# 1. OFDM Modulator / Demodulator
# ---------------------------------------------------------------------------

def ofdm_modulate(symbols: np.ndarray,
                   N: int,
                   cp_len: int,
                   pilot_indices: Optional[list] = None,
                   pilot_values: Optional[np.ndarray] = None) -> np.ndarray:
    """
    OFDM modulation: IFFT + cyclic prefix insertion.

    Parameters
    ----------
    symbols       : data symbols in frequency domain (length ≤ N)
    N             : FFT size
    cp_len        : cyclic prefix length
    pilot_indices : optional pilot subcarrier indices
    pilot_values  : corresponding pilot values

    Returns
    -------
    tx_signal : time-domain OFDM symbol (length N + cp_len)
    """
    freq_domain = np.zeros(N, dtype=complex)

    # Insert data
    data_indices = [i for i in range(N) if
                    pilot_indices is None or i not in pilot_indices]
    n_data = min(len(symbols), len(data_indices))
    for i, idx in enumerate(data_indices[:n_data]):
        freq_domain[idx] = symbols[i]

    # Insert pilots
    if pilot_indices is not None and pilot_values is not None:
        for idx, val in zip(pilot_indices, pilot_values):
            freq_domain[idx] = val

    # IFFT (normalised)
    time_domain = np.fft.ifft(freq_domain, N) * np.sqrt(N)

    # Cyclic prefix
    cp = time_domain[-cp_len:]
    return np.concatenate([cp, time_domain])


def ofdm_demodulate(rx_symbol: np.ndarray,
                     N: int,
                     cp_len: int) -> np.ndarray:
    """
    OFDM demodulation: CP removal + FFT.

    Parameters
    ----------
    rx_symbol : received OFDM symbol (length N + cp_len)
    N         : FFT size
    cp_len    : cyclic prefix length

    Returns
    -------
    freq_domain : complex subcarrier values (length N)
    """
    # Remove CP
    time_domain = rx_symbol[cp_len: cp_len + N]
    freq_domain = np.fft.fft(time_domain, N) / np.sqrt(N)
    return freq_domain


# ---------------------------------------------------------------------------
# 2. Preamble Generation (Schmidl–Cox)
# ---------------------------------------------------------------------------

def generate_sc_preamble(N: int, cp_len: int, seed: int = 42) -> np.ndarray:
    """
    Generate a Schmidl–Cox training preamble.

    The preamble consists of two identical halves in the time domain.
    Achieved by placing BPSK symbols only on even subcarriers and zero
    on odd subcarriers, then taking IFFT.

    Returns
    -------
    preamble : complex time-domain preamble (length N + cp_len)
    """
    rng = np.random.default_rng(seed)
    freq = np.zeros(N, dtype=complex)
    # Only even subcarriers are non-zero → two identical halves in time
    n_even = N // 2
    bpsk = (2 * rng.integers(0, 2, n_even) - 1).astype(float)
    freq[0::2] = bpsk / np.sqrt(n_even)   # power normalised

    time = np.fft.ifft(freq, N) * np.sqrt(N)
    cp = time[-cp_len:]
    return np.concatenate([cp, time])


def generate_moose_preamble(N: int, cp_len: int, seed: int = 42) -> np.ndarray:
    """
    Generate a Moose preamble: two back-to-back identical OFDM symbols.

    Returns
    -------
    preamble : complex time-domain signal of length 2*(N + cp_len)
    """
    rng = np.random.default_rng(seed)
    bpsk = (2 * rng.integers(0, 2, N) - 1).astype(complex)
    symbol = ofdm_modulate(bpsk, N, cp_len)
    return np.concatenate([symbol, symbol])


# ---------------------------------------------------------------------------
# 3. Channel Models
# ---------------------------------------------------------------------------

def awgn_channel(signal: np.ndarray, snr_db: float,
                  seed: Optional[int] = None) -> np.ndarray:
    """
    Add complex AWGN to signal at the specified SNR (dB).

    Parameters
    ----------
    signal : complex baseband signal
    snr_db : signal-to-noise ratio in dB (per sample, complex noise)
    seed   : optional RNG seed

    Returns
    -------
    noisy_signal : signal + noise
    """
    rng = np.random.default_rng(seed)
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / snr_linear
    # Complex AWGN: noise_power split equally between I and Q
    noise = rng.normal(0, np.sqrt(noise_power / 2), len(signal)) + \
            1j * rng.normal(0, np.sqrt(noise_power / 2), len(signal))
    return signal + noise


def multipath_channel(signal: np.ndarray,
                       tap_delays: List[int],
                       tap_gains: List[complex]) -> np.ndarray:
    """
    Apply a static multipath channel (FIR filter).

    Parameters
    ----------
    signal      : input complex signal
    tap_delays  : list of delay taps (in samples)
    tap_gains   : complex gain for each tap

    Returns
    -------
    output : channel-filtered signal (same length as input, zero-padded)
    """
    max_delay = max(tap_delays) if tap_delays else 0
    output = np.zeros(len(signal) + max_delay, dtype=complex)
    for delay, gain in zip(tap_delays, tap_gains):
        output[delay: delay + len(signal)] += gain * signal
    return output[:len(signal)]


def doppler_channel(signal: np.ndarray,
                     fd_normalized: float) -> np.ndarray:
    """
    Apply a pure Doppler shift (frequency offset due to mobility).

    Parameters
    ----------
    signal        : complex input signal
    fd_normalized : normalised Doppler frequency  fd·T  (fd / subcarrier Δf)

    Returns
    -------
    doppler_shifted : signal multiplied by exp(j·2π·fd·n)
    """
    n = np.arange(len(signal))
    return signal * np.exp(1j * 2 * np.pi * fd_normalized * n)


# ---------------------------------------------------------------------------
# 4. Pilot Utilities
# ---------------------------------------------------------------------------

def generate_pilot_pattern(N: int, pilot_spacing: int = 8,
                             seed: int = 0) -> Tuple[list, np.ndarray]:
    """
    Generate evenly-spaced pilot subcarriers with BPSK values.

    Returns
    -------
    pilot_indices : list of subcarrier indices
    pilot_values  : complex pilot symbols (±1)
    """
    rng = np.random.default_rng(seed)
    pilot_indices = list(range(0, N, pilot_spacing))
    pilot_values = (2 * rng.integers(0, 2, len(pilot_indices)) - 1).astype(complex)
    return pilot_indices, pilot_values


# ---------------------------------------------------------------------------
# 5. SNR Utilities
# ---------------------------------------------------------------------------

def measure_snr(clean: np.ndarray, received: np.ndarray) -> float:
    """Measure effective SNR (dB) given clean reference and received signal."""
    noise_power = np.mean(np.abs(received - clean) ** 2)
    signal_power = np.mean(np.abs(clean) ** 2)
    if noise_power < 1e-15:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def qpsk_symbols(n: int, seed: int = 0) -> np.ndarray:
    """Generate n random QPSK symbols (±1/√2 ± j/√2)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 4, n)
    angles = bits * np.pi / 2 + np.pi / 4
    return np.exp(1j * angles)
