"""
sync_pipeline.py
================
End-to-End OFDM Synchronization Pipeline.

Integrates:
  1. Schmidl–Cox timing
  2. Fractional CFO (Schmidl–Cox)
  3. Fine CFO (Moose)
  4. Integer CFO (Pilot search)
  5. PLL phase tracking
  6. CPE correction

Usage
-----
    from src.sync_pipeline import OFDMSyncPipeline

    pipeline = OFDMSyncPipeline(N=64, cp_len=16, pilot_spacing=8)
    result = pipeline.synchronize(rx_signal)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .timing_sync import schmidl_cox_timing
from .cfo_estimation import (estimate_fractional_cfo,
                               estimate_fine_cfo_moose,
                               estimate_integer_cfo,
                               apply_cfo_correction)
from .phase_tracking import PLLPhaseTracker, estimate_cpe_pilots
from .ofdm_utils import (ofdm_demodulate, generate_pilot_pattern)


@dataclass
class SyncResult:
    """Container for all synchronization estimates."""
    timing_offset: int = 0
    fractional_cfo: float = 0.0
    fine_cfo: float = 0.0
    integer_cfo: int = 0
    total_cfo: float = 0.0
    cpe_rad: float = 0.0
    cfo_corrected_signal: Optional[np.ndarray] = None
    rx_freq_domain: Optional[np.ndarray] = None
    timing_metric: Optional[np.ndarray] = None
    success: bool = True
    error_msg: str = ""


class OFDMSyncPipeline:
    """
    Complete OFDM synchronization pipeline.

    Parameters
    ----------
    N              : FFT size (number of subcarriers)
    cp_len         : cyclic prefix length
    pilot_spacing  : subcarrier spacing between pilots (default 8)
    pll_kp         : PLL proportional gain
    pll_ki         : PLL integral gain
    enable_moose   : use Moose fine CFO (requires extra preamble symbol)
    enable_pll     : enable PLL phase tracking
    """

    def __init__(self,
                 N: int = 64,
                 cp_len: int = 16,
                 pilot_spacing: int = 8,
                 pll_kp: float = 0.05,
                 pll_ki: float = 0.001,
                 enable_moose: bool = True,
                 enable_pll: bool = True):
        self.N = N
        self.cp_len = cp_len
        self.symbol_len = N + cp_len
        self.pilot_indices, self.pilot_values = generate_pilot_pattern(
            N, pilot_spacing)
        self.pll = PLLPhaseTracker(Kp=pll_kp, Ki=pll_ki)
        self.enable_moose = enable_moose
        self.enable_pll = enable_pll

    # ------------------------------------------------------------------
    def synchronize(self, rx_signal: np.ndarray,
                    known_pilots: Optional[np.ndarray] = None) -> SyncResult:
        """
        Run the full synchronization chain on a received signal.

        Parameters
        ----------
        rx_signal    : complex received baseband signal
        known_pilots : optional override for pilot values

        Returns
        -------
        SyncResult dataclass with all estimates and corrected signals
        """
        result = SyncResult()

        if known_pilots is not None:
            self.pilot_values = known_pilots

        # ── Stage 1: Timing ──────────────────────────────────────────────
        try:
            timing_result = schmidl_cox_timing(rx_signal, self.N, self.cp_len)
            result.timing_offset = timing_result['timing_offset']
            result.timing_metric = timing_result['metric']
        except Exception as e:
            result.success = False
            result.error_msg = f"Timing failed: {e}"
            return result

        d = result.timing_offset

        # ── Stage 2: Fractional CFO (Schmidl–Cox) ────────────────────────
        try:
            frac_result = estimate_fractional_cfo(rx_signal, self.N, d)
            result.fractional_cfo = frac_result['cfo_normalized']
        except Exception as e:
            result.error_msg += f" | Frac CFO failed: {e}"

        # ── Stage 3: Fine CFO (Moose) ────────────────────────────────────
        if self.enable_moose:
            try:
                moose_result = estimate_fine_cfo_moose(rx_signal, self.N, d, self.cp_len)
                result.fine_cfo = moose_result['cfo_normalized']
            except Exception as e:
                result.fine_cfo = result.fractional_cfo
                result.error_msg += f" | Moose CFO failed: {e}"
        else:
            result.fine_cfo = result.fractional_cfo

        # ── Stage 4: CFO Correction (fractional + fine) ──────────────────
        total_frac = (result.fractional_cfo + result.fine_cfo) / 2
        cfo_corrected = apply_cfo_correction(
            rx_signal, total_frac, self.N, self.cp_len)
        result.cfo_corrected_signal = cfo_corrected

        # ── Stage 5: FFT demodulation ─────────────────────────────────────
        sym_start = d + self.cp_len   # after CP
        rx_sym = cfo_corrected[d: d + self.symbol_len]
        rx_freq = ofdm_demodulate(rx_sym, self.N, self.cp_len)

        # ── Stage 6: Integer CFO ──────────────────────────────────────────
        try:
            int_result = estimate_integer_cfo(
                rx_freq, self.pilot_indices, self.pilot_values)
            result.integer_cfo = int_result['integer_cfo']
        except Exception as e:
            result.error_msg += f" | Int CFO failed: {e}"

        result.total_cfo = total_frac + result.integer_cfo

        # ── Stage 7: Apply integer CFO correction (circular shift) ────────
        if result.integer_cfo != 0:
            rx_freq = np.roll(rx_freq, -result.integer_cfo)

        # ── Stage 8: CPE estimation and correction ────────────────────────
        try:
            cpe_result = estimate_cpe_pilots(
                rx_freq, self.pilot_indices, self.pilot_values)
            rx_freq = cpe_result['corrected_rx']
            result.cpe_rad = cpe_result['cpe_rad']
        except Exception as e:
            result.error_msg += f" | CPE failed: {e}"

        # ── Stage 9: PLL phase tracking (pilot subcarriers) ──────────────
        if self.enable_pll:
            self.pll.reset()
            rx_pilots = rx_freq[self.pilot_indices]
            corrected_pilots = self.pll.process_block(
                rx_pilots, self.pilot_values)
            # Apply PLL phase estimate to all subcarriers
            pll_phase = np.mean(self.pll.phase_history) if self.pll.phase_history else 0
            rx_freq = rx_freq * np.exp(-1j * pll_phase)

        result.rx_freq_domain = rx_freq
        return result

    # ------------------------------------------------------------------
    def reset(self):
        """Reset PLL state for new packet."""
        self.pll.reset()
