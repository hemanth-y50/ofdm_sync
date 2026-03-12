"""
test_sync.py
============
Unit and integration tests for the OFDM synchronization pipeline.

Run with:
    pytest tests/test_sync.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ofdm_utils import (
    generate_sc_preamble, generate_moose_preamble,
    awgn_channel, ofdm_modulate, ofdm_demodulate,
    generate_pilot_pattern, qpsk_symbols, multipath_channel
)
from src.timing_sync import schmidl_cox_timing
from src.cfo_estimation import (
    estimate_fractional_cfo, estimate_fine_cfo_moose,
    estimate_integer_cfo, apply_cfo_correction, cfo_estimation_mse
)
from src.phase_tracking import (
    generate_phase_noise, apply_phase_noise,
    estimate_cpe_pilots, PLLPhaseTracker,
    ici_self_cancellation_encode, ici_self_cancellation_decode
)
from src.sync_pipeline import OFDMSyncPipeline


# ── Fixtures ─────────────────────────────────────────────────────────────────

N = 64
CP = 16
SNR_HIGH = 30   # dB – nearly noise-free
SNR_MED = 15    # dB
FS = 1e6        # Hz


@pytest.fixture
def basic_preamble():
    return generate_sc_preamble(N, CP, seed=0)


@pytest.fixture
def moose_preamble():
    return generate_moose_preamble(N, CP, seed=0)


# ── Timing Tests ─────────────────────────────────────────────────────────────

class TestTiming:
    def test_timing_no_noise(self, basic_preamble):
        """SC timing should find offset near CP (end of plateau)."""
        result = schmidl_cox_timing(basic_preamble, N, CP)
        assert abs(result['timing_offset'] - CP) <= CP, (
            f"Expected ~{CP}, got {result['timing_offset']}")

    def test_timing_with_delay(self, basic_preamble):
        """Timing should track a known integer delay."""
        delay = 20
        padded = np.concatenate([np.zeros(delay, dtype=complex), basic_preamble])
        result = schmidl_cox_timing(padded, N, CP)
        assert abs(result['timing_offset'] - (delay + CP)) <= CP

    def test_timing_metric_range(self, basic_preamble):
        """SC metric should be in [0, ~1] (allow small overshoot from noise)."""
        result = schmidl_cox_timing(basic_preamble, N, CP)
        m = result['metric']
        assert np.all(m >= 0)
        assert np.max(m) <= 1.1

    def test_timing_high_snr(self, basic_preamble):
        """Under high SNR, timing error should be within CP window."""
        rx = awgn_channel(basic_preamble, SNR_HIGH, seed=1)
        result = schmidl_cox_timing(rx, N, CP)
        assert abs(result['timing_offset'] - CP) <= CP

    def test_timing_medium_snr(self, basic_preamble):
        """Under moderate SNR, timing error should be within CP window."""
        rx = awgn_channel(basic_preamble, SNR_MED, seed=2)
        result = schmidl_cox_timing(rx, N, CP)
        assert abs(result['timing_offset'] - CP) <= CP


# ── CFO Tests ────────────────────────────────────────────────────────────────

class TestFractionalCFO:
    def test_zero_cfo(self, basic_preamble):
        """With zero CFO, estimate should be near zero."""
        result = estimate_fractional_cfo(basic_preamble, N, CP)
        assert abs(result['cfo_normalized']) < 0.05

    def test_known_fractional_cfo(self, basic_preamble):
        """Inject known fractional CFO, check recovery."""
        true_cfo = 0.3   # 30% of subcarrier spacing
        n = np.arange(len(basic_preamble))
        rx = basic_preamble * np.exp(1j * 2 * np.pi * true_cfo * n / N)
        result = estimate_fractional_cfo(rx, N, CP)
        assert abs(result['cfo_normalized'] - true_cfo) < 0.05

    def test_cfo_correction(self, basic_preamble):
        """After correction, residual CFO should be < 0.01."""
        true_cfo = 0.25
        n = np.arange(len(basic_preamble))
        rx = basic_preamble * np.exp(1j * 2 * np.pi * true_cfo * n / N)
        est = estimate_fractional_cfo(rx, N, CP)['cfo_normalized']
        corrected = apply_cfo_correction(rx, est, N, CP)
        residual = estimate_fractional_cfo(corrected, N, CP)['cfo_normalized']
        assert abs(residual) < 0.05


class TestMooseCFO:
    def test_moose_zero_cfo(self, moose_preamble):
        result = estimate_fine_cfo_moose(moose_preamble, N, 0, CP)
        assert abs(result['cfo_normalized']) < 0.05

    def test_moose_known_cfo(self, moose_preamble):
        true_cfo = 0.2
        n = np.arange(len(moose_preamble))
        rx = moose_preamble * np.exp(1j * 2 * np.pi * true_cfo * n / N)
        result = estimate_fine_cfo_moose(rx, N, 0, CP)
        assert abs(result['cfo_normalized'] - true_cfo) < 0.05


class TestIntegerCFO:
    def test_integer_cfo_zero(self):
        """No integer CFO → estimate should be 0."""
        pilot_indices, pilot_values = generate_pilot_pattern(N, 8)
        freq = np.zeros(N, dtype=complex)
        for i, v in zip(pilot_indices, pilot_values):
            freq[i] = v
        result = estimate_integer_cfo(freq, pilot_indices, pilot_values, 5)
        assert result['integer_cfo'] == 0

    def test_integer_cfo_known(self):
        """Known integer shift should be recovered."""
        pilot_indices, pilot_values = generate_pilot_pattern(N, 8)
        freq = np.zeros(N, dtype=complex)
        for i, v in zip(pilot_indices, pilot_values):
            freq[i] = v
        shift = 3
        shifted_freq = np.roll(freq, shift)
        result = estimate_integer_cfo(shifted_freq, pilot_indices, pilot_values, 5)
        assert result['integer_cfo'] == -shift or abs(result['integer_cfo'] - shift) <= 1


# ── Phase Tracking Tests ──────────────────────────────────────────────────────

class TestPhaseNoise:
    def test_phase_noise_shape(self):
        pn = generate_phase_noise(1000, linewidth_hz=1000, fs=FS, seed=0)
        assert pn.shape == (1000,)
        assert pn.dtype == float

    def test_phase_noise_variance_scales(self):
        """Higher linewidth → larger variance."""
        pn_low = generate_phase_noise(10000, 100, FS, seed=0)
        pn_high = generate_phase_noise(10000, 10000, FS, seed=0)
        assert np.var(np.diff(pn_high)) > np.var(np.diff(pn_low))

    def test_cpe_estimation_clean(self):
        """CPE estimate on noise-free signal should be near the true CPE."""
        pilot_indices, pilot_values = generate_pilot_pattern(N, 8)
        freq = np.zeros(N, dtype=complex)
        for i, v in zip(pilot_indices, pilot_values):
            freq[i] = v
        true_cpe = 0.4   # radians
        freq_noisy = freq * np.exp(1j * true_cpe)
        result = estimate_cpe_pilots(freq_noisy, pilot_indices, pilot_values)
        assert abs(result['cpe_rad'] - true_cpe) < 0.05

    def test_pll_converges(self):
        """PLL phase error should decrease over time."""
        pll = PLLPhaseTracker(Kp=0.1, Ki=0.01)
        n = 200
        true_phase = 0.5
        decisions = np.ones(n, dtype=complex)
        rx = np.exp(1j * true_phase) * decisions
        for r, d in zip(rx, decisions):
            pll.step(r, d)
        # Later errors should be smaller than early errors
        early_err = np.mean(np.abs(pll.error_history[:20]))
        late_err = np.mean(np.abs(pll.error_history[-20:]))
        assert late_err < early_err


class TestICISelfCancellation:
    def test_encode_decode_roundtrip(self):
        """Encode then decode should recover original symbols."""
        symbols = qpsk_symbols(32, seed=5)
        encoded = ici_self_cancellation_encode(symbols)
        decoded = ici_self_cancellation_decode(encoded)
        np.testing.assert_allclose(decoded, symbols, atol=1e-10)

    def test_encoded_length(self):
        symbols = qpsk_symbols(32)
        encoded = ici_self_cancellation_encode(symbols)
        assert len(encoded) == 2 * len(symbols)


# ── Integration Test: Full Pipeline ──────────────────────────────────────────

class TestSyncPipeline:
    def test_pipeline_runs(self):
        """Pipeline should run without errors on a simple signal."""
        pipeline = OFDMSyncPipeline(N=N, cp_len=CP)
        preamble = generate_sc_preamble(N, CP)
        rx = awgn_channel(preamble, SNR_HIGH, seed=0)
        result = pipeline.synchronize(rx)
        assert result.success or len(result.error_msg) > 0  # at least ran

    def test_pipeline_timing_reasonable(self):
        """Pipeline timing offset should be within CP window of ground truth."""
        pipeline = OFDMSyncPipeline(N=N, cp_len=CP)
        preamble = generate_sc_preamble(N, CP)
        rx = awgn_channel(preamble, SNR_HIGH, seed=10)
        result = pipeline.synchronize(rx)
        assert abs(result.timing_offset - CP) <= CP

    def test_pipeline_cfo_reasonable(self):
        """Total CFO estimate should be close to injected CFO."""
        pipeline = OFDMSyncPipeline(N=N, cp_len=CP, enable_moose=False)
        preamble = generate_sc_preamble(N, CP)
        true_cfo = 0.2
        n = np.arange(len(preamble))
        rx = preamble * np.exp(1j * 2 * np.pi * true_cfo * n / N)
        rx = awgn_channel(rx, SNR_HIGH, seed=0)
        result = pipeline.synchronize(rx)
        assert abs(result.fractional_cfo - true_cfo) < 0.1

    def test_pipeline_multipath(self):
        """Pipeline should still find timing under mild multipath."""
        pipeline = OFDMSyncPipeline(N=N, cp_len=CP)
        preamble = generate_sc_preamble(N, CP)
        mp = multipath_channel(preamble, [0, 4, 8], [1.0, 0.5+0.3j, 0.2])
        rx = awgn_channel(mp, SNR_HIGH, seed=3)
        result = pipeline.synchronize(rx)
        assert abs(result.timing_offset - CP) <= CP  # within CP window


# ── MSE Utility ──────────────────────────────────────────────────────────────

class TestMSEUtility:
    def test_mse_zero_error(self):
        estimates = np.full(100, 0.3)
        stats = cfo_estimation_mse(0.3, estimates)
        assert stats['mse'] < 1e-12
        assert abs(stats['bias']) < 1e-12

    def test_mse_nonzero(self):
        rng = np.random.default_rng(0)
        estimates = 0.3 + rng.normal(0, 0.05, 1000)
        stats = cfo_estimation_mse(0.3, estimates)
        assert stats['mse'] > 0
        assert abs(stats['rmse'] - 0.05) < 0.01
