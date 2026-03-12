"""
OFDM Synchronization Module
============================
Timing, CFO Estimation, Phase Noise Mitigation & Tracking.
"""

from .timing_sync import schmidl_cox_timing, cross_correlation_timing
from .cfo_estimation import (
    estimate_fractional_cfo,
    estimate_fine_cfo_moose,
    estimate_integer_cfo,
    apply_cfo_correction,
    cfo_estimation_mse,
)
from .phase_tracking import (
    generate_phase_noise,
    apply_phase_noise,
    estimate_cpe_pilots,
    PLLPhaseTracker,
    ici_self_cancellation_encode,
    ici_self_cancellation_decode,
    compute_evm_phase_noise,
    snr_degradation_phase_noise,
)
from .ofdm_utils import (
    ofdm_modulate,
    ofdm_demodulate,
    generate_sc_preamble,
    generate_moose_preamble,
    awgn_channel,
    multipath_channel,
    doppler_channel,
    generate_pilot_pattern,
    measure_snr,
    qpsk_symbols,
)
from .sync_pipeline import OFDMSyncPipeline, SyncResult

__all__ = [
    "schmidl_cox_timing", "cross_correlation_timing",
    "estimate_fractional_cfo", "estimate_fine_cfo_moose",
    "estimate_integer_cfo", "apply_cfo_correction", "cfo_estimation_mse",
    "generate_phase_noise", "apply_phase_noise", "estimate_cpe_pilots",
    "PLLPhaseTracker", "ici_self_cancellation_encode",
    "ici_self_cancellation_decode", "compute_evm_phase_noise",
    "snr_degradation_phase_noise",
    "ofdm_modulate", "ofdm_demodulate", "generate_sc_preamble",
    "generate_moose_preamble", "awgn_channel", "multipath_channel",
    "doppler_channel", "generate_pilot_pattern", "measure_snr", "qpsk_symbols",
    "OFDMSyncPipeline", "SyncResult",
]
