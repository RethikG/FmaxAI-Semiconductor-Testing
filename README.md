<<<<<<< HEAD
# FmaxAI: AI-Optimized Semiconductor Fmax Search for Test Time Reduction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![Physics-Guided](https://img.shields.io/badge/Semiconductor-ATE%20Speed%20Binning-green.svg)](#)
[![Code Style: Modular](https://img.shields.io/badge/Architecture-Clean%20%26%20Modular-blueviolet.svg)](#)

**FmaxAI** is an advanced, physics-guided machine learning framework engineered to dramatically reduce Automatic Test Equipment (ATE) and System-Level Test (SLT) execution time during semiconductor speed binning.

By utilizing fast, pre-test on-chip telemetry (ring oscillators, leakage currents, $V_{min}$, and wafer spatial coordinates) with **Split Conformal Prediction**, FmaxAI restricts the physical tester's frequency search to a calibrated, narrow interval. 

> **Core Safety Guarantee:** Unlike pure predictive binning, FmaxAI's default windowed search **always physically measures and confirms the die's $F_{max}$ on silicon**. Machine learning tells the tester *where to look*, not *what to decide*, ensuring zero yield compromise and zero escape risk.

---

## 📑 Table of Contents
- [The Semiconductor Speed Binning Problem](#-the-semiconductor-speed-binning-problem)
- [Conventional Search vs. AI-Optimized Approach](#-conventional-search-vs-ai-optimized-approach)
- [Comprehensive Search Strategy Benchmark](#-comprehensive-search-strategy-benchmark)
- [Algorithms Implemented](#-algorithms-implemented)
- [Machine Learning & Zero Data Leakage Pipeline](#-machine-learning--zero-data-leakage-pipeline)
- [Quality, Risk Management & Production Controls](#-quality-risk-management--production-controls)
- [Interactive Streamlit Dashboard](#-interactive-streamlit-dashboard)
- [Repository Architecture](#-repository-architecture)
- [Installation & Local Execution](#-installation--local-execution)
- [Automated Test Suite](#-automated-test-suite)
- [Limitations & Scientific Disclaimer](#-limitations--scientific-disclaimer)
- [References](#-references)

---

## ⚡ The Semiconductor Speed Binning Problem

In modern semiconductor fabrication (CPUs, GPUs, mobile SoCs, automotive microcontrollers), microscopic variations in transistor gate lengths, oxide thicknesses, and chemical doping cause chips cut from the same silicon wafer to operate at different maximum frequencies ($F_{max}$). 

To optimize revenue and product positioning, manufacturers test every die to classify it into speed tiers (speed bins) or DVFS (Dynamic Voltage and Frequency Scaling) tables.

### The Test-Time Cost Model
The time required to find $F_{max}$ at a single voltage-temperature corner is governed by:
$$T_{Fmax} = N_{steps} \times (t_{switch} + t_{apply})$$

- $N_{steps}$: Number of clock frequencies evaluated on the tester.
- $t_{switch} \approx 5\text{ ms}$: Hardware overhead to switch clock dividers, relock PLLs, and settle power supplies.
- $t_{apply} \approx 25\text{ ms}$: Time to apply test patterns (at ATE) or hundreds of milliseconds to seconds to boot code (at System-Level Test, SLT).
- In multi-corner DVFS binning (e.g., 4 voltages $\times$ 2 temperatures = 8 corners), test time multiplies linearly.

In high-volume manufacturing (20+ million units/year), tester cost rates ($C_{test} \approx \$0.004/\text{second}$) turn test seconds into **millions of dollars of operational expense**. Reducing $N_{steps}$ with predictive AI delivers the largest, most reliable financial return.

---

## 🔍 Conventional Search vs. AI-Optimized Approach

```
1. Conventional Binary Search (Blind, full grid 1600-2400 MHz):
   [1600 MHz] ------------------------ [2000 MHz?] ------------------------ [2400 MHz]
                                      (6 to 7 blind steps per corner)

2. AI-Optimized Windowed Binary Search (Informed, narrow window ~75 MHz):
   [1600 MHz] ------------ [1925 MHz === [Window: ~3.8 steps] === 2000 MHz] ------------ [2400 MHz]
                           (Starts directly where ML predicts the die operates)
```

### Conventional Methods & Their Weaknesses
1. **Linear Sweep**: Steps upward from 1600 MHz in 10 MHz increments until failure. Robust, but extremely slow ($40\text{--}60$ steps per corner).
2. **Binary Search**: Logarithmically halves the search window. Fast on average ($6\text{--}7$ steps), but **completely blind** (treats every chip identically) and **fragile to test noise**: a single spurious failure early in the search permanently steers the tester into the wrong frequency range.
3. **Naive ML Seeding**: Starting a linear walk at the predicted frequency spends most of its steps traversing residual error ($6.2$ tests on average), barely improving on binary search.

### The FmaxAI Windowed Approach
1. Fast on-chip sensors (ring oscillators, leakage, $V_{min}$) are read at wafer sort or initial test setup.
2. A Gradient Boosted model infers the die's expected frequency $\hat{y}$ in $<1\text{ ms}$.
3. Split Conformal Prediction generates a high-confidence search window $[L, U] = [\hat{y} - 1.5\sigma, \hat{y} + 1.5\sigma]$.
4. The tester executes binary search **only inside the narrow $[L, U]$ window**.
5. **Fail-Safe Edge Recovery:** If the true $F_{max}$ lies outside the window, dynamic exponential striding extends the search outward until the true boundary is resolved.

---

## 📊 Comprehensive Search Strategy Benchmark

Results below reflect a standardized simulation study on held-out test silicon under realistic tester repeatability noise ($\sigma_{test} = 4.0\text{ MHz}$, $10\text{ MHz}$ grid, $8\text{ MHz}$ measurement guard band):

| Search Strategy | Mean Tests | p95 Tests | Time (ms) | Saving vs. Binary | Saving vs. Linear | Final $F_{max}$ | Quality Risk (Escapes) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear sweep** | 40.49 | 58 | 1215 ms | $-527\%$ | $0\%$ | Measured | Baseline |
| **Binary search** | 6.45 | 7 | 194 ms | $0\%$ | $+84\%$ | Measured | Noise-sensitive |
| **Bayesian (population prior)** | 6.52 | 9 | 195 ms | $-1\%$ | $+84\%$ | Posterior estimate | Noise-robust |
| **ML-seeded walk (naive)** | 6.17 | 10 | 185 ms | $+4\%$ | $+85\%$ | Measured | Inefficient tail |
| **ML-windowed binary (Alg 1)** | **3.89** | **8** | **117 ms** | **+40%** | **+90%** | **Measured** | **0 ppm (Fails safe)** |
| **ML + Bayesian (Alg 2)** | 4.35 | 7 | 131 ms | $+33\%$ | $+89\%$ | Posterior estimate | Noise-robust |
| **ML confirm-bin (Alg 3)** | **1.77** | **2** | **53 ms** | **+73%** | **+96%** | Bin Boundary Check | Binning only |

---

## 🛠️ Algorithms Implemented

### Algorithm 1: Prediction-Windowed Binary Search (Recommended Default)
- **Inputs:** Unit features $x$, frequency grid $G$, interval multiplier $k$.
- **Step 1:** Verify $x$ against Out-of-Distribution (OOD) criteria. If OOD, route to full binary search.
- **Step 2:** Predict $(\hat{y}, L, U)$ using conformal calibration.
- **Step 3:** Restrict binary search to grid slice $G[\text{index}(L) \dots \text{index}(U)]$.
- **Step 4 (Dynamic Edge Recovery):**
  - If upper window edge passes ($best == G[hi]$): stride upward exponentially ($stride \times 2$) to capture unexpected fast dies.
  - If lower edge fails ($best \text{ is None}$): binary search below window $G[0 \dots lo-1]$.
- **Return:** Measured $F_{max}$ (100% physically verified).

### Algorithm 2: Probabilistic Bisection with Learned Prior (Bayesian Adaptive Search)
- Initializes posterior distribution $p(F) = \mathcal{N}(\mu, s^2)$ from the ML prediction.
- Iteratively tests at the posterior median and updates likelihood:
  $$P(\text{pass} \mid f, F_{max}) = \Phi\left(\frac{F_{max} - f}{\sigma_{test}}\right)$$
- Terminates when the 90% credible interval $\le 15\text{ MHz}$; returns the conservative 5th percentile.

### Algorithm 3: Confirm-Only Binning
- Designed for products requiring only the discrete speed bin (e.g. 1900 MHz), not continuous frequency.
- Evaluates bin edges starting at $\hat{y} - k\sigma$ with guard band; resolves bin assignment in **1.77 tests per part**.

### Algorithm 4: Cross-Corner DVFS Window Reuse
- Models the physical $V_{dd}\text{--}F_{max}$ curve. 
- The measured $F_{max}$ at initial voltage $V_0$ narrows the search windows for $V_1, V_2, \dots$, reducing multi-corner test counts by **54.8%**.

---

## 🧠 Machine Learning & Zero Data Leakage Pipeline

### 1. Strictly Isolated Pre-Test Features
Only features available *prior* to ATE frequency search are exposed to the model:
- `ro_01` to `ro_10`: 10 Ring Oscillator frequencies tracking local gate delay.
- `iddq_norm`: Quiescent standby leakage current (inversely tracks threshold voltage $V_t$).
- `vmin_norm`: Minimum functional operating voltage from scan or memory BIST.
- `die_x`, `die_y`, `radius_sq`: Normalized wafer Cartesian and radial coordinates ($x^2 + y^2$).

### 2. Quarantined Downstream Columns (Guaranteed Zero Leakage)
The following simulation outcome columns are strictly excluded from model features:
`fmax_pred_mhz`, `window_low_mhz`, `window_high_mhz`, `tests_binary`, `tests_windowed`, `fmax_measured_windowed_mhz`, and `true_bin_mhz`.

### 3. Model Evaluation on Held-Out Test Set (4,000 Units)

| Model Family | RMSE (MHz) | MAE (MHz) | $R^2$ Score | NRMSE (%) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ridge Regression Baseline** | **25.38** | **20.31** | **0.9406** | **3.56%** | Fast, interpretable, embedded coefficients |
| **Gradient Boosting (GBM)** | **27.05** | **21.56** | **0.9325** | **3.79%** | Matches Technical Document benchmark (26.9 MHz) |
| **Random Forest** | 27.20 | 21.66 | 0.9317 | 3.81% | Robust ensemble baseline |
| **LightGBM** | 27.99 | 22.31 | 0.9277 | 3.92% | Fast inference for automated test cells |

---

## 🛡️ Quality, Risk Management & Production Controls

1. **Out-of-Distribution (OOD) Detector (`src/quality/ood_detector.py`)**:
   - Calculates multivariate Mahalanobis distance on incoming monitor vectors:
     $$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$
   - Any unit exceeding the 99th percentile threshold or failing individual feature bounds is routed to the reference full binary search.
2. **Audit Sampling & Drift Monitoring (`src/quality/drift_monitor.py`)**:
   - A random sample of units (~few percent, $\approx 60$ dies/lot) is routed to full search to generate unbiased labels.
   - EWMA (Exponentially Weighted Moving Average) and CUSUM control charts track prediction residuals to detect fab process drift.
3. **Guard-Band Sensitivity & Escapes Analysis**:
   - Demonstrates that pure prediction binning requires a large $3\sigma$ guard band to prevent over-bin escapes (<100 ppm), sacrificing 74% of dies to lower bins.
   - Windowed binary search eliminates this trade-off because the final grade is physically measured.

---

## 🖥️ Interactive Streamlit Dashboard

The project includes an interactive web application built with Streamlit and Plotly:

1. **Executive Summary & KPIs**: High-level test-time reduction metrics, Table 6 strategy benchmark, and an interactive factory ROI calculator.
2. **Wafer Spatial Map & EDA**: Interactive 2D circular wafer maps showing radial variation and sensor correlations.
3. **ML Models & Conformal Bounds**: Regression performance comparisons, feature importance plots, and real-time $k\sigma$ calibration sliders.
4. **Live Search Trajectory Simulator**: Step-by-step physical tester trajectories (Figure 3) comparing Linear Sweep vs. Binary Search vs. Windowed Search for any chip.
5. **Quality, Risk & OOD Fallback**: Real-time sensor anomaly injection to test Mahalanobis OOD triggers, and EWMA/CUSUM drift charts.
6. **Multi-Corner DVFS Simulator**: Multi-voltage scaling and cross-corner window reuse (Algorithm 4).
7. **Dataset & Benchmark Explorer**: Complete table viewer with search, filtering, and CSV export.

---

## 📁 Repository Architecture

```
FmaxAI_Project/
├── .gitignore                    # Git exclusions (.venv, caches, temporary logs)
├── conftest.py                   # Pytest configuration and path resolution
├── requirements.txt              # Pinned production dependencies
├── README.md                     # Comprehensive project documentation
├── synthetic_fmax_dataset.csv    # 6,000-unit semiconductor dataset
├── AI_Optimized_Fmax_Search_Technical_Document.pdf # Technical specification
│
├── src/                          # Core Python framework
│   ├── __init__.py
│   ├── config.py                 # Tester specs, timing constants, bins, and guard bands
│   ├── data_loader.py            # Leak-free data pipeline (train/cal/test splits)
│   ├── models/                   # ML & uncertainty quantification
│   │   ├── __init__.py
│   │   ├── train.py              # Model training, comparison, and evaluation
│   │   └── conformal.py          # Split Conformal Prediction calibration
│   ├── simulation/               # Silicon testing simulation engine
│   │   ├── __init__.py
│   │   ├── tester_env.py         # Virtual ATE tester with Gaussian noise
│   │   ├── search_algorithms.py  # Algorithms 1, 2, 3, 4 + Linear & Binary baselines
│   │   └── benchmark.py          # Batch simulation runner
│   └── quality/                  # Safety, risk & statistical process control
│       ├── __init__.py
│       ├── ood_detector.py       # Mahalanobis distance OOD excursion detector
│       └── drift_monitor.py      # EWMA/CUSUM residual drift and lot PSI monitor
│
├── app/                          # Interactive Streamlit application
│   ├── streamlit_app.py          # Multi-tab dashboard entry point
│   └── components/
│       └── plots.py              # Reusable Plotly charts (trajectories, wafer maps)
│
└── tests/                        # Automated unit tests
    ├── test_search_algorithms.py # Search algorithm correctness & edge recovery
    └── test_conformal.py         # Pipeline integrity & interval coverage
```

---

## 💻 Installation & Local Execution

### 1. Prerequisites
- Python 3.11 or 3.12
- Operating System: Windows, macOS, or Linux

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/FmaxAI.git
cd FmaxAI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Test Suite

Run the full automated test suite with pytest:
```bash
python -m pytest -v tests/
```
All 8 unit tests verify:
- Zero data leakage and split validation.
- Conformal interval coverage on held-out data.
- Correctness of Linear Sweep, Binary Search, and Confirm-Only Binning.
- Windowed Binary Search edge-recovery mechanics.
- Out-of-Distribution Mahalanobis distance triggers.

---

## ⚠️ Limitations & Scientific Disclaimer

1. **Synthetic Data Simulation**: All results in this repository are evaluated on synthetic semiconductor data designed to mirror real silicon physical characteristics (ring oscillators, leakage, and Gaussian tester repeatability noise). Numerical values are illustrative and should be re-established on target silicon products before high-volume manufacturing deployment.
2. **Thermal & Supply Settle Times**: Physical tester settling times ($t_{switch}$, $t_{apply}$) depend on specific ATE hardware architectures and device power delivery networks.
3. **High-Reliability Products**: For mission-critical applications (automotive, aerospace), the final frequency grade must remain backed by physical measurements (Approaches B through E). Pure predictive binning (Approach A) is strictly not recommended for safety-critical silicon.

---

## 📚 References

1. S.-P. Mu, M. C.-T. Chao, S.-H. Chen, Y.-M. Wang, *"Statistical framework and built-in self-speed-binning system for speed binning using on-chip ring oscillators,"* IEEE Trans. VLSI Systems, 2016.
2. K.-W. Chang, M. C.-T. Chao, *"DVFS binning using machine-learning techniques,"* M.S. thesis, National Chiao Tung University, 2017.
3. N. Bellarmino et al., *"Semi-supervised deep learning for microcontroller performance screening,"* Proc. IEEE European Test Symposium (ETS), 2023.
4. A. N. Angelopoulos, S. Bates, *"A gentle introduction to conformal prediction and distribution-free uncertainty quantification,"* arXiv:2107.07511, 2021.
5. R. Waeber, P. I. Frazier, S. G. Henderson, *"Bisection search with noisy responses,"* SIAM J. Control and Optimization, vol. 51, no. 3, 2013.
=======
# FmaxAI-Semiconductor-Testing
AI-powered semiconductor Fmax prediction and test-time optimization
>>>>>>> 59fa6c190456d09391a6ee56d5009cff8dabb7a1
