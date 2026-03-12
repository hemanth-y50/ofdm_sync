"""
evaluate.py
===========
Monte-Carlo evaluation of OFDM synchronization performance.

Generates 4 SEPARATE plots saved to output_dir/:
  1. plot1_timing_error_vs_snr.png
  2. plot2_cfo_mse_vs_snr.png
  3. plot3_phase_noise_evm.png
  4. plot4_doppler_stability.png

Run:
    python evaluate.py --output results/
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ofdm_utils import (
    generate_sc_preamble, generate_moose_preamble,
    awgn_channel, generate_pilot_pattern, qpsk_symbols,
    ofdm_modulate, ofdm_demodulate, doppler_channel
)
from src.timing_sync import schmidl_cox_timing
from src.cfo_estimation import (
    estimate_fractional_cfo, estimate_fine_cfo_moose,
    apply_cfo_correction, cfo_estimation_mse
)
from src.phase_tracking import (
    generate_phase_noise, apply_phase_noise,
    estimate_cpe_pilots, compute_evm_phase_noise
)
from src.sync_pipeline import OFDMSyncPipeline

# ── Config ────────────────────────────────────────────────────────────────────
N         = 64
CP        = 16
FS        = 1e6
N_TRIALS  = 200
SNR_RANGE = np.arange(-5, 31, 3)
TRUE_CFO  = 0.25

# ── Style ─────────────────────────────────────────────────────────────────────
DARK    = '#0d1117'
ACCENT1 = '#58a6ff'
ACCENT2 = '#f78166'
ACCENT3 = '#3fb950'
ACCENT4 = '#d2a8ff'
GRID    = '#30363d'
TEXT    = '#e6edf3'

def _base_fig(title):
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=DARK)
    ax.set_facecolor(DARK)
    ax.set_title(title, color=TEXT, fontsize=14, fontweight='bold', pad=14)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    return fig, ax

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close(fig)
    print(f"  saved -> {path}")

# ── 1. Timing ─────────────────────────────────────────────────────────────────
def eval_timing_vs_snr(n_trials=N_TRIALS):
    print("  [1/4] Timing error vs SNR ...")
    preamble = generate_sc_preamble(N, CP, seed=42)
    mae = []
    for snr in SNR_RANGE:
        errors = [abs(schmidl_cox_timing(awgn_channel(preamble, snr, seed=t),
                                          N, CP)['timing_offset'] - CP)
                  for t in range(n_trials)]
        mae.append(np.mean(errors))
    return SNR_RANGE, np.array(mae)

def plot_timing(output_dir, snr, mae):
    fig, ax = _base_fig('Timing Error vs SNR  (Schmidl-Cox)')
    ax.semilogy(snr, mae, color=ACCENT1, lw=2.5, marker='o', markersize=6,
                label='SC Timing MAE', zorder=3)
    ax.axhline(1, color=ACCENT2, ls='--', lw=1.8, alpha=0.85,
               label='1-sample threshold')
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('Mean Absolute Timing Error (samples)', fontsize=11)
    ax.legend(fontsize=10, facecolor='#161b22', labelcolor=TEXT, edgecolor=GRID)
    _save(fig, os.path.join(output_dir, 'plot1_timing_error_vs_snr.png'))

# ── 2. CFO MSE ────────────────────────────────────────────────────────────────
# FIX: Moose called with timing_offset=0, cp_len=CP so each block is CP-stripped
def eval_cfo_vs_snr(n_trials=N_TRIALS):
    print("  [2/4] CFO MSE vs SNR ...")
    sc_pre    = generate_sc_preamble(N, CP, seed=42)
    moose_pre = generate_moose_preamble(N, CP, seed=42)
    mse_frac, mse_moose = [], []

    for snr in SNR_RANGE:
        ef_list, em_list = [], []
        for t in range(n_trials):
            n_arr = np.arange(len(sc_pre))
            rx_sc = awgn_channel(sc_pre * np.exp(1j*2*np.pi*TRUE_CFO*n_arr/N),
                                  snr, seed=t)
            n_arr2 = np.arange(len(moose_pre))
            rx_m  = awgn_channel(moose_pre * np.exp(1j*2*np.pi*TRUE_CFO*n_arr2/N),
                                  snr, seed=t)
            try:
                ef_list.append(estimate_fractional_cfo(rx_sc, N, CP)['cfo_normalized'])
            except Exception:
                ef_list.append(0.0)
            try:
                # FIXED: timing_offset=0 (signal starts at preamble), cp_len=CP
                em_list.append(estimate_fine_cfo_moose(rx_m, N,
                                                        timing_offset=0,
                                                        cp_len=CP)['cfo_normalized'])
            except Exception:
                em_list.append(0.0)

        mse_frac.append(cfo_estimation_mse(TRUE_CFO, np.array(ef_list))['mse'])
        mse_moose.append(cfo_estimation_mse(TRUE_CFO, np.array(em_list))['mse'])

    return SNR_RANGE, np.array(mse_frac), np.array(mse_moose)

def plot_cfo(output_dir, snr, mse_frac, mse_moose):
    fig, ax = _base_fig('CFO Estimation MSE vs SNR')
    ax.semilogy(snr, mse_frac,  color=ACCENT2, lw=2.5, marker='s', markersize=6,
                label='Schmidl-Cox  (Fractional)', zorder=3)
    ax.semilogy(snr, mse_moose, color=ACCENT3, lw=2.5, marker='^', markersize=6,
                label='Moose  (Fine)', zorder=3)
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('CFO MSE  (normalised^2)', fontsize=11)
    ax.legend(fontsize=10, facecolor='#161b22', labelcolor=TEXT, edgecolor=GRID)
    _save(fig, os.path.join(output_dir, 'plot2_cfo_mse_vs_snr.png'))

# ── 3. Phase Noise EVM ────────────────────────────────────────────────────────
# FIX: use ofdm_demodulate on full symbol; cap EVM at 100%
def eval_phase_noise_evm(n_trials=50):
    print("  [3/4] Phase noise EVM vs linewidth ...")
    linewidths = np.logspace(1, 5, 20)
    pilot_idx, pilot_val = generate_pilot_pattern(N, 8)
    evm_raw, evm_cpe = [], []

    for lw in linewidths:
        evms_r, evms_c = [], []
        for t in range(n_trials):
            symbols = qpsk_symbols(N, seed=t)
            tx      = ofdm_modulate(symbols, N, CP, pilot_idx, pilot_val)
            pn      = generate_phase_noise(len(tx), lw, FS, seed=t)
            rx_pn   = apply_phase_noise(tx, pn)

            # FIXED: demodulate full symbol from index 0
            ref_freq = ofdm_demodulate(tx,    N, CP)
            rx_freq  = ofdm_demodulate(rx_pn, N, CP)

            evms_r.append(min(compute_evm_phase_noise(ref_freq, rx_freq), 100.0))

            cpe_res = estimate_cpe_pilots(rx_freq, pilot_idx, pilot_val)
            evms_c.append(min(compute_evm_phase_noise(ref_freq,
                                                       cpe_res['corrected_rx']), 100.0))
        evm_raw.append(np.mean(evms_r))
        evm_cpe.append(np.mean(evms_c))

    return linewidths, np.array(evm_raw), np.array(evm_cpe)

def plot_evm(output_dir, linewidths, evm_raw, evm_cpe):
    fig, ax = _base_fig('Phase Noise EVM vs Oscillator Linewidth')
    ax.semilogx(linewidths, evm_raw, color=ACCENT4, lw=2.5,
                label='Without CPE Correction', zorder=3)
    ax.semilogx(linewidths, evm_cpe, color=ACCENT3, lw=2.5, ls='--',
                label='With CPE Correction', zorder=3)
    ax.fill_between(linewidths, evm_cpe, evm_raw, alpha=0.12, color=ACCENT3)
    ax.set_xlabel('Oscillator 3-dB Linewidth (Hz)', fontsize=11)
    ax.set_ylabel('EVM (%)', fontsize=11)
    ax.set_ylim(bottom=0, top=105)
    ax.legend(fontsize=10, facecolor='#161b22', labelcolor=TEXT, edgecolor=GRID)
    _save(fig, os.path.join(output_dir, 'plot3_phase_noise_evm.png'))

# ── 4. Doppler Stability ──────────────────────────────────────────────────────
def eval_doppler_stability(n_trials=100):
    print("  [4/4] Sync stability vs Doppler ...")
    doppler_range = np.linspace(0, 0.5, 15)
    preamble = generate_sc_preamble(N, CP, seed=42)
    success_rate = []

    for fd in doppler_range:
        success = sum(
            1 for t in range(n_trials)
            if abs(schmidl_cox_timing(
                awgn_channel(doppler_channel(preamble, fd), 20, seed=t),
                N, CP)['timing_offset'] - CP) <= CP
        )
        success_rate.append(success / n_trials * 100)

    return doppler_range, np.array(success_rate)

def plot_doppler(output_dir, doppler, stab):
    fig, ax = _base_fig('Sync Stability vs Doppler Shift  (SNR = 20 dB)')
    ax.plot(doppler, stab, color=ACCENT1, lw=2.5, marker='D', markersize=6,
            label='Timing Success Rate', zorder=3)
    ax.fill_between(doppler, stab, alpha=0.15, color=ACCENT1)
    ax.axhline(95, color=ACCENT2, ls='--', lw=1.8, alpha=0.85,
               label='95% threshold')
    ax.set_xlabel('Normalised Doppler  f_d * T', fontsize=11)
    ax.set_ylabel('Timing Success Rate (%)', fontsize=11)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=10, facecolor='#161b22', labelcolor=TEXT, edgecolor=GRID)
    _save(fig, os.path.join(output_dir, 'plot4_doppler_stability.png'))

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results')
    parser.add_argument('--trials', type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print("\n-- OFDM Sync Evaluation ------------------------------------------")

    snr, timing_mae          = eval_timing_vs_snr(args.trials)
    snr, mse_frac, mse_moose = eval_cfo_vs_snr(args.trials)
    lws, evm_raw, evm_cpe    = eval_phase_noise_evm()
    dop, stab                = eval_doppler_stability()

    print("\n  Saving plots ...")
    plot_timing(args.output, snr, timing_mae)
    plot_cfo(args.output, snr, mse_frac, mse_moose)
    plot_evm(args.output, lws, evm_raw, evm_cpe)
    plot_doppler(args.output, dop, stab)

    print("\n-- Done: 4 plots saved to:", args.output, "--\n")
