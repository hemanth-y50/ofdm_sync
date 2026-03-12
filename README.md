# OFDM Synchronization Pipeline

A complete, modular **OFDM synchronization system** implemented from scratch in Python. Covers the full sync chain — coarse timing, fractional/fine/integer CFO estimation, PLL-based phase tracking, and CPE correction — with rigorous Monte-Carlo evaluation across SNR, Doppler, and oscillator linewidth conditions.

---

## Results

| Test | Result |
|------|--------|
| Timing MAE < 1 sample | SNR ≥ 5 dB |
| Schmidl-Cox CFO MSE | ~10⁻³ @ 10 dB SNR |
| Moose CFO MSE | ~10⁻⁴ @ 10 dB SNR (10× better than SC) |
| CPE correction gain | Significant EVM reduction up to ~10 kHz linewidth |
| Timing success rate | > 95% for normalized Doppler f_d·T ≤ 0.25 |

<p align="center">
  <img src="results/plot1_timing_error_vs_snr.png" width="48%"/>
  <img src="results/plot2_cfo_mse_vs_snr.png" width="48%"/>
</p>
<p align="center">
  <img src="results/plot3_phase_noise_evm.png" width="48%"/>
  <img src="results/plot4_doppler_stability.png" width="48%"/>
</p>

---

## Synchronization Pipeline

```
Received Signal  rx[n]
        │
        ▼
┌───────────────────┐
│  Stage 1: Timing  │  Schmidl-Cox metric, plateau detection
│  (Coarse)         │  → timing_offset (sample index)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stage 2: Frac.   │  Schmidl-Cox phase angle
│  CFO Estimation   │  ε_frac = angle(P) / π   (|ε| ≤ 0.5 Δf)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stage 3: Fine    │  Moose algorithm (two repeated preamble blocks)
│  CFO (Moose)      │  ε_fine = angle(Σ r2·r1*) / 2π
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stage 4: CFO     │  Counter-rotating phase ramp: exp(-j·2π·ε·n/N)
│  Correction       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stage 5: FFT     │  CP removal + N-point FFT
│  Demodulation     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stage 6: Integer │  Exhaustive pilot-correlation search ±10 subcarriers
│  CFO Search       │  → circular shift of frequency-domain output
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stage 7: CPE     │  Pilot-aided: φ_CPE = angle(Σ rx_p · tx_p*)
│  Estimation       │  Linear interpolation between pilot positions
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Stage 8: PLL     │  Second-order PI decision-directed PLL
│  Phase Tracking   │  e[n] = angle(r[n]·ŝ*[n])
│                   │  v[n] = Kp·e[n] + Ki·Σe
└────────┬──────────┘
         │
         ▼
   Synchronized rx_freq[k]
```

---

## Features

### Timing Synchronization (`src/timing_sync.py`)
- **Schmidl-Cox algorithm** — sliding-window correlator, plateau detection (threshold = 0.95 × peak)
- **Cross-correlation timing** — fine sync using known time-domain preamble
- Uses preamble with two identical halves (even subcarriers only → natural half-symbol repetition)

### CFO Estimation (`src/cfo_estimation.py`)
- **Fractional CFO (Schmidl-Cox)** — phase of cross-correlator P; range ±0.5 subcarrier spacings
- **Fine CFO (Moose)** — two repeated N-point blocks; 10× lower MSE than SC at same SNR
- **Integer CFO** — exhaustive pilot correlation search over ±10 subcarrier range
- **CFO correction** — counter-rotating time-domain exponential applied after timing

### Phase Tracking (`src/phase_tracking.py`)
- **Wiener-process phase noise model** — cumulative sum of Gaussian increments, σ² = 2π·Δν/fs
- **Pilot-aided CPE estimation** — MLE: φ̂ = angle(Σ r_p · s_p*)
- **PI decision-directed PLL** — proportional (Kp) + integral (Ki) loop filter; continuous phase tracking
- **ICI self-cancellation** — encode [S, −S] on adjacent subcarriers; decode via difference combining

### Full Pipeline (`src/sync_pipeline.py`)
- `OFDMSyncPipeline` class — configurable N, CP, pilot spacing, PLL gains
- `SyncResult` dataclass — returns all intermediate estimates for analysis
- Modular: each stage can be enabled/disabled independently

### Evaluation (`evaluate.py`)
- 200-trial Monte-Carlo simulation per SNR/Doppler point
- 4 plots: timing MAE, CFO MSE (SC vs. Moose), phase noise EVM vs. linewidth, Doppler stability

---

## Project Structure

```
ofdm_sync/
├── evaluate.py                 # Monte-Carlo evaluation, generates all 4 plots
├── src/
│   ├── timing_sync.py          # Schmidl-Cox + cross-correlation timing
│   ├── cfo_estimation.py       # Fractional / Moose / integer CFO + correction
│   ├── phase_tracking.py       # Phase noise model, PLL, CPE, ICI cancellation
│   ├── sync_pipeline.py        # End-to-end OFDMSyncPipeline class
│   └── ofdm_utils.py           # OFDM mod/demod, preamble gen, channel models
├── tests/
│   └── test_sync.py            # Unit tests
└── results/
    ├── plot1_timing_error_vs_snr.png
    ├── plot2_cfo_mse_vs_snr.png
    ├── plot3_phase_noise_evm.png
    └── plot4_doppler_stability.png
```

---

## Quick Start

### Requirements
```bash
pip install numpy scipy matplotlib
```

### Run Full Evaluation
```bash
cd ofdm_sync
python evaluate.py --output results/ --trials 200
```

Generates all 4 performance plots to `results/`.

### Use the Pipeline in Your Code
```python
from src.sync_pipeline import OFDMSyncPipeline

pipeline = OFDMSyncPipeline(
    N=64,            # FFT size
    cp_len=16,       # cyclic prefix length
    pilot_spacing=8, # 1 pilot every 8 subcarriers
    pll_kp=0.05,     # PLL proportional gain
    pll_ki=0.001,    # PLL integral gain
    enable_moose=True,
    enable_pll=True,
)

result = pipeline.synchronize(rx_signal)
print(f"Timing offset : {result.timing_offset}")
print(f"Total CFO     : {result.total_cfo:.4f} (normalized)")
print(f"CPE           : {result.cpe_rad:.4f} rad")
```

### Run Unit Tests
```bash
python -m pytest tests/test_sync.py -v
```

---

## Theory References

| Algorithm | Reference |
|-----------|-----------|
| Schmidl-Cox timing & fractional CFO | T. M. Schmidl and D. C. Cox, IEEE Trans. Commun., vol. 45, no. 12, 1997 |
| Moose fine CFO | P. H. Moose, IEEE Trans. Commun., vol. 42, no. 10, 1994 |
| Phase noise model (Wiener process) | L. Tomba, IEEE Trans. Commun., vol. 46, no. 5, 1998 |
| ICI self-cancellation | Y. Zhao and S.-G. Haggman, IEEE Trans. Commun., vol. 49, no. 12, 2001 |

---

## Author

**Yenuganti Hemanth Kumar**
B.Tech ECE, RGUKT Nuzvid | [GitHub](https://github.com/hemanth-y50) | [Email](mailto:yenugantihemanthkumar@gmail.com)
