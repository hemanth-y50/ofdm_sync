# OFDM Synchronization Module

> **Timing · CFO Estimation · Phase Noise Mitigation · PLL Tracking**

A clean, well-documented Python implementation of a complete OFDM synchronization chain — built for reproducibility, easy extension, and resume/portfolio use.

---

## Features

| Stage | Algorithm | File |
|---|---|---|
| **Coarse Timing** | Schmidl–Cox plateau detection | `src/timing_sync.py` |
| **Fine Timing** | Cross-correlation with known preamble | `src/timing_sync.py` |
| **Fractional CFO** | Schmidl–Cox phase estimator | `src/cfo_estimation.py` |
| **Fine CFO** | Moose algorithm (repeated preamble) | `src/cfo_estimation.py` |
| **Integer CFO** | Pilot-based exhaustive search | `src/cfo_estimation.py` |
| **Phase Noise Model** | Wiener process (Lorentzian PSD) | `src/phase_tracking.py` |
| **CPE Estimation** | Pilot-aided Common Phase Error | `src/phase_tracking.py` |
| **ICI Mitigation** | Self-cancellation encode/decode | `src/phase_tracking.py` |
| **Phase Tracking** | Second-order PI PLL | `src/phase_tracking.py` |
| **Full Pipeline** | Integrated 9-stage sync chain | `src/sync_pipeline.py` |

---

## Project Structure

```
ofdm_sync/
├── src/
│   ├── __init__.py          # Public API exports
│   ├── timing_sync.py       # Schmidl–Cox + cross-correlation timing
│   ├── cfo_estimation.py    # Fractional / Moose / Integer CFO
│   ├── phase_tracking.py    # Phase noise, CPE, ICI, PLL
│   ├── ofdm_utils.py        # Modulator, channel models, utilities
│   └── sync_pipeline.py     # End-to-end OFDMSyncPipeline class
├── tests/
│   └── test_sync.py         # 25+ unit & integration tests (pytest)
├── notebooks/
│   └── demo.ipynb           # Interactive walkthrough
├── results/                 # Auto-generated evaluation plots
├── evaluate.py              # Monte-Carlo evaluation script
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone & install dependencies
git clone https://github.com/<your-username>/ofdm-sync.git
cd ofdm-sync
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run Monte-Carlo evaluation (generates plots in results/)
python evaluate.py --trials 200
```

---

## Usage

### One-liner synchronization

```python
from src.sync_pipeline import OFDMSyncPipeline
from src.ofdm_utils import generate_sc_preamble, awgn_channel

pipeline = OFDMSyncPipeline(N=64, cp_len=16)
preamble = generate_sc_preamble(N=64, cp_len=16)
rx = awgn_channel(preamble, snr_db=20)

result = pipeline.synchronize(rx)
print(f"Timing offset : {result.timing_offset}")
print(f"Fractional CFO: {result.fractional_cfo:.4f}")
print(f"Integer CFO   : {result.integer_cfo}")
print(f"CPE           : {result.cpe_rad:.4f} rad")
```

### Individual modules

```python
from src.timing_sync import schmidl_cox_timing
from src.cfo_estimation import estimate_fractional_cfo, apply_cfo_correction
from src.phase_tracking import PLLPhaseTracker, estimate_cpe_pilots

# Timing
timing = schmidl_cox_timing(rx_signal, N=64, cp_len=16)
d = timing['timing_offset']

# CFO correction
cfo = estimate_fractional_cfo(rx_signal, N=64, timing_offset=d)
corrected = apply_cfo_correction(rx_signal, cfo['cfo_normalized'], N=64, cp_len=16)

# Phase tracking
pll = PLLPhaseTracker(Kp=0.05, Ki=0.001)
for sample, decision in zip(rx_samples, decisions):
    corrected_sample = pll.step(sample, decision)
```

---

## Evaluation Results

Running `python evaluate.py` produces a 4-panel figure:

- **Timing MAE vs SNR** — sub-sample accuracy above 10 dB SNR
- **CFO MSE vs SNR** — Moose outperforms Schmidl–Cox by ~3 dB at low SNR
- **Phase Noise EVM vs Linewidth** — CPE correction reduces EVM by up to 40%
- **Sync Stability vs Doppler** — >95% success rate for fd·T < 0.2

---

## Algorithms Reference

### Schmidl–Cox Timing

Exploits the two-identical-half structure of the preamble:

```
M(d) = |P(d)|² / R(d)²

P(d) = Σ_{k=0}^{L-1} r*(d+k) · r(d+k+L)
R(d) = Σ_{k=0}^{L-1} |r(d+k+L)|²
```

The metric M(d) forms a plateau of width equal to the CP length, allowing robust timing detection even under multipath.

### Moose CFO Estimation

Two consecutive identical OFDM blocks allow fine CFO estimation:

```
ε_fine = angle( Σ_k r2[k] · r1*[k] ) / (2π)
```

Range: ±0.5 subcarrier spacings. Higher accuracy than Schmidl–Cox alone.

### PLL Phase Tracker

Second-order proportional-integral loop:

```
e[n]   = angle( r[n] · ŝ*[n] )
v[n]   = Kp·e[n] + Ki·Σe
θ[n+1] = θ[n] + v[n]
```

Tracks residual phase noise after CPE correction.

---

## References

1. T. M. Schmidl and D. C. Cox, *"Robust frequency and timing synchronization for OFDM,"* IEEE Trans. Commun., Dec. 1997.
2. P. H. Moose, *"A technique for orthogonal frequency division multiplexing frequency offset correction,"* IEEE Trans. Commun., Oct. 1994.
3. J. van de Beek et al., *"ML estimation of time and frequency offset in OFDM systems,"* IEEE Trans. Signal Process., Jul. 1997.
4. L. Tomba, *"On the effect of Wiener phase noise in OFDM systems,"* IEEE Trans. Commun., May 1998.
5. D. Petrovic et al., *"Effects of phase noise on OFDM systems with and without PLL,"* IEEE Trans. Commun., Aug. 2007.

---

## License

MIT License — free to use, modify, and distribute.
