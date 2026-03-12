"""
phase_tracking.py
=================
PLL-based Phase Tracking and Phase-Noise Mitigation for OFDM.

Implements:
  • Lorentzian / masked phase-noise model (oscillator PSD)
  • Decision-directed PLL for Common Phase Error (CPE) tracking
  • ICI self-cancellation scheme for Inter-Carrier Interference reduction
  • Pilot-aided CPE estimation

References:
    Tomba (1998); Petrovic et al. (2007); Armada (2001).
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Phase Noise Model
# ---------------------------------------------------------------------------

def generate_phase_noise(n_samples: int,
                          linewidth_hz: float,
                          fs: float,
                          seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a Wiener-process (random-walk) phase noise sequence.

    For a free-running oscillator with 3-dB linewidth Δν, the discrete-time
    phase increments are i.i.d. Gaussian with variance:
        σ²_Δφ = 2π · Δν / fs

    Parameters
    ----------
    n_samples    : number of samples to generate
    linewidth_hz : oscillator 3-dB linewidth (Hz)  e.g. 100–10 000 Hz
    fs           : sampling frequency (Hz)
    seed         : optional RNG seed for reproducibility

    Returns
    -------
    phase_noise : np.ndarray (real, radians) of length n_samples
    """
    rng = np.random.default_rng(seed)
    sigma2 = 2 * np.pi * linewidth_hz / fs
    increments = rng.normal(0, np.sqrt(sigma2), n_samples)
    phase_noise = np.cumsum(increments)
    return phase_noise


def apply_phase_noise(signal: np.ndarray,
                       phase_noise: np.ndarray) -> np.ndarray:
    """Multiply signal by exp(j·φ_noise) to impose phase noise."""
    return signal * np.exp(1j * phase_noise[:len(signal)])


# ---------------------------------------------------------------------------
# 2. CPE Estimation (Pilot-Aided)
# ---------------------------------------------------------------------------

def estimate_cpe_pilots(rx_freq: np.ndarray,
                         pilot_indices: list,
                         pilot_values: np.ndarray) -> dict:
    """
    Estimate the Common Phase Error (CPE) from pilot subcarriers.

    CPE is the phase term common to all subcarriers caused by the DC component
    of phase noise.  Using pilots:
        φ_CPE = angle( Σ_p  rx[p] · tx*[p] )

    Parameters
    ----------
    rx_freq       : received OFDM symbol in frequency domain (after FFT)
    pilot_indices : list of pilot subcarrier indices
    pilot_values  : known transmitted pilot symbols

    Returns
    -------
    dict: 'cpe_rad' (CPE estimate in radians), 'corrected_rx'
    """
    pilot_values = np.asarray(pilot_values, dtype=complex)
    rx_pilots = rx_freq[pilot_indices]
    cross = rx_pilots * pilot_values.conj()
    cpe_rad = float(np.angle(np.sum(cross)))

    # Correct all subcarriers
    corrected_rx = rx_freq * np.exp(-1j * cpe_rad)

    return {
        'cpe_rad': cpe_rad,
        'corrected_rx': corrected_rx,
    }


# ---------------------------------------------------------------------------
# 3. Decision-Directed PLL
# ---------------------------------------------------------------------------

class PLLPhaseTracker:
    """
    Second-order decision-directed PLL for continuous phase tracking.

    The loop filter is a proportional-integral (PI) type:
        e[n]   = angle( r[n] · conj(ŝ[n]) )   ← phase error
        v[n]   = Kp·e[n] + Ki·Σe             ← loop filter output
        θ[n+1] = θ[n] + v[n]                 ← NCO update

    Parameters
    ----------
    Kp : proportional gain
    Ki : integral gain
    """

    def __init__(self, Kp: float = 0.05, Ki: float = 0.001):
        self.Kp = Kp
        self.Ki = Ki
        self._integrator = 0.0
        self._phase = 0.0
        self.phase_history: list = []
        self.error_history: list = []

    def reset(self):
        self._integrator = 0.0
        self._phase = 0.0
        self.phase_history = []
        self.error_history = []

    def step(self, rx_sample: complex, decision: complex) -> complex:
        """
        Process one sample through the PLL.

        Parameters
        ----------
        rx_sample : received (noisy) complex sample
        decision  : hard decision / known pilot for this sample

        Returns
        -------
        corrected_sample : phase-corrected complex sample
        """
        # NCO correction
        corrected = rx_sample * np.exp(-1j * self._phase)

        # Phase detector
        error = np.angle(corrected * decision.conj())

        # Loop filter (PI)
        self._integrator += error
        v = self.Kp * error + self.Ki * self._integrator

        # NCO update
        self._phase += v

        self.phase_history.append(self._phase)
        self.error_history.append(error)

        return corrected

    def process_block(self, rx_block: np.ndarray,
                       decisions: np.ndarray) -> np.ndarray:
        """Apply PLL to an entire block of samples."""
        out = np.zeros_like(rx_block)
        for i, (r, d) in enumerate(zip(rx_block, decisions)):
            out[i] = self.step(r, d)
        return out


# ---------------------------------------------------------------------------
# 4. ICI Self-Cancellation
# ---------------------------------------------------------------------------

def ici_self_cancellation_encode(tx_symbols: np.ndarray) -> np.ndarray:
    """
    ICI self-cancellation encoding: each data symbol is mapped onto two
    adjacent subcarriers as [S, -S].  This doubles the bandwidth but
    significantly reduces ICI power.

    Parameters
    ----------
    tx_symbols : data symbols (length N/2 for N total subcarriers)

    Returns
    -------
    encoded : length-N array with [S_k, -S_k] pattern
    """
    N = len(tx_symbols)
    encoded = np.zeros(2 * N, dtype=complex)
    encoded[0::2] = tx_symbols
    encoded[1::2] = -tx_symbols
    return encoded


def ici_self_cancellation_decode(rx_freq: np.ndarray) -> np.ndarray:
    """
    ICI self-cancellation decoding: combine adjacent subcarriers to cancel ICI.
        Ŝ_k = R[2k] - R[2k+1]   (maximum ratio combining of the two branches)
    """
    N = len(rx_freq) // 2
    decoded = (rx_freq[0::2] - rx_freq[1::2]) / 2
    return decoded[:N]


# ---------------------------------------------------------------------------
# 5. Phase Noise Impact Metrics
# ---------------------------------------------------------------------------

def compute_evm_phase_noise(clean_symbols: np.ndarray,
                              noisy_symbols: np.ndarray) -> float:
    """
    Error Vector Magnitude (EVM) due to phase noise (%).

    EVM = sqrt( mean|error|² / mean|ref|² ) × 100
    """
    error_power = np.mean(np.abs(noisy_symbols - clean_symbols) ** 2)
    ref_power = np.mean(np.abs(clean_symbols) ** 2)
    return float(np.sqrt(error_power / (ref_power + 1e-15)) * 100)


def snr_degradation_phase_noise(phase_noise_rad: np.ndarray,
                                 N: int) -> float:
    """
    Estimate SNR degradation (dB) caused by phase noise in an N-subcarrier
    OFDM system.

    Approximate formula (Pollet et al., 1995):
        ΔSNR ≈ -10 log10(1 - σ²_φ)   for small σ²_φ
    where σ²_φ is the phase-noise variance per OFDM symbol.
    """
    # Phase noise variance per symbol (average over one symbol period)
    var_phi = float(np.var(phase_noise_rad[:N]))
    snr_loss_db = -10 * np.log10(max(1 - var_phi, 1e-6))
    return snr_loss_db
