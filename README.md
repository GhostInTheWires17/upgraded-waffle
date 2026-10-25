# Quant Finance Lab

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 34 Passed](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](tests/)

> *"My journey learning mathematics, programming, and quantitative finance."*  
> — **Manish Sai**, Original Author & Quantitative Researcher

---

## 1. Project Identity & Philosophy

**Quant Finance Lab** is an open, modular quantitative research laboratory built to explore stochastic modelling, Monte Carlo simulation, financial mathematics, and risk dynamics under uncertainty.

The project originated as an educational investigation into how win probability, risk-reward ratios, and multiplicative compounding govern final wealth distributions. This version completely rebuilds the codebase from monolithic scripts into an extensible, mathematically rigorous research platform while maintaining the author's original vision.

Quant Finance Lab is **not** a commercial black-box trading bot or financial signal vendor. Instead, it is a transparent laboratory where researchers, students, and engineers can:
* Formulate stochastic hypotheses.
* Run reproducible Monte Carlo experiments.
* Compare theoretical analytical expectations against empirical simulation outcomes.
* Investigate path drawdowns, ruin probabilities, and fat-tailed tail risk.
* Understand the mathematical foundations underlying capital growth and risk management.

---

## 2. Features

### Core Simulation Engine
* **High-Performance Vectorization**: Built on NumPy array operations and chunked memory processing, capable of simulating over **150,000 paths per second** on a standard laptop.
* **Deterministic Reproducibility**: Seeded random streams using NumPy `SeedSequence` and `default_rng`, ensuring that experiments reproduce bit-for-bit.
* **Preserved Bernoulli Fixed-Return Model**: The author's original binary trading model is preserved as a first-class citizen, fully parameterized and tested.

### Advanced Mathematical & Risk Analytics
* **Analytical Benchmarks**: Closed-form theoretical expected balance $\mathbb{E}[B_N]$ and geometric median calculations with automated z-score hypothesis testing against simulation means.
* **Statistical Reliability**: Wilson score confidence intervals for all discrete event probabilities (target reach, ruin risk, profit rates) and Student's t intervals for expected balances.
* **Path Drawdowns & Ruin**: Continuous tracking of running peaks, Maximum Drawdown (MDD) distributions, and first-passage ruin vs terminal horizon ruin.
* **Tail-Risk Metrics**: Value-at-Risk (95% and 99% VaR) and Expected Shortfall (CVaR) with explicit, standardized sign conventions.
* **Higher Moments**: Sample mean, median, standard deviation, standardized Fisher-Pearson skewness, and excess kurtosis.

### Modeling Extensions & Friction
* **Continuous & Fat-Tailed Models**: Gaussian (Normal), Student-t (fat-tailed leptokurtic), Geometric Brownian Motion (GBM), and Markov Regime-Switching models.
* **Trading Friction**: Configurable fixed transaction fees and percentage slippage / broker commissions.
* **Position Sizing**: Full compounding, fixed-fractional allocation ($f \in (0, 1]$), fixed dollar staking, and theoretical Kelly Criterion calculations.

### Analysis, Visualizations & Interface
* **Sensitivity & Scenario Sweeps**: 1D parameter sweeps and 2D parameter grids powered by Common Random Numbers (CRN) to isolate true parameter sensitivity from Monte Carlo noise.
* **Publication-Grade Visualizations**: Clean, academic Matplotlib charts for balance histograms, equity curves, drawdown distributions, and error-bar target success curves.
* **Command-Line Interface (CLI)**: Full CLI (`simulate`, `sensitivity`, `scenarios`, `dashboard`).
* **Interactive Streamlit Dashboard**: Optional graphical user interface for interactive parameter exploration.
* **Historical Data Subsystem**: Isolated CSV price loader with strict schema validation and empirical return distribution calibration.

---

## 3. Mathematical Model

### Compounding Recurrence
Let $B_0 > 0$ denote initial capital. In each trade step $t \in \{1, 2, \dots, N\}$, a fractional return $R_t$ occurs:
$$B_{t+1} = B_t (1 + R_t)$$

Over $N$ sequential trades, final capital is:
$$B_N = B_0 \prod_{t=1}^N (1 + R_t)$$

### Core Fixed-Return Bernoulli Model
For independent and identically distributed (i.i.d.) trades where each trade wins with probability $p$ yielding $R_{\text{win}}$ and loses with probability $1-p$ yielding $R_{\text{loss}}$:
$$\mathbb{E}[1 + R_t] = p(1 + R_{\text{win}}) + (1 - p)(1 + R_{\text{loss}})$$

The exact theoretical expected final balance is:
$$\mathbb{E}[B_N] = B_0 \left[ p(1 + R_{\text{win}}) + (1 - p)(1 + R_{\text{loss}}) \right]^N$$

The geometric growth rate (Kelly exponent) governing the median path is:
$$g = \mathbb{E}[\ln(1 + R_t)] = p \ln(1 + R_{\text{win}}) + (1 - p) \ln(1 + R_{\text{loss}})$$
$$\text{Median}[B_N] \approx B_0 e^{N g}$$

By Jensen's Inequality, since $e^x$ is strictly convex:
$$\mathbb{E}[B_N] > \text{Median}[B_N]$$
This mathematical property explains why simulated multiplicative trading paths are inherently **right-skewed**: most paths cluster near the median, while rare favorable runs create a long right tail.

---

## 4. Installation

Quant Finance Lab requires **Python 3.10 or higher**.

### Windows (PowerShell)
```powershell
# 1. Clone repository
git clone https://github.com/manishsai-rv/quant-finance-lab.git
cd quant-finance-lab

# 2. Create and activate virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install package
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

### macOS / Linux
```bash
# 1. Clone repository
git clone https://github.com/manishsai-rv/quant-finance-lab.git
cd quant-finance-lab

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install package
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

---

## 5. Quick Start

Run your first reproducible Monte Carlo simulation directly from the command line:

```bash
python -m quant_finance_lab simulate --initial-capital 1000 --win-prob 0.60 --win-return 0.02 --loss-return -0.01 --simulations 10000 --trades 100 --seed 42
```

### Example Terminal Output:
```text
========================================================================
 QUANT FINANCE LAB — STOCHASTIC ANALYTICS SIMULATION
========================================================================
Model: Fixed Bernoulli Return Model
Simulations (Paths): 10,000  |  Trades per Path: 100
Initial Capital: $1,000.00  |  Random Seed: 42
Execution Time: 0.053 seconds
------------------------------------------------------------------------

[1] THEORETICAL BENCHMARK VS MONTE CARLO ESTIMATION
  Theoretical Expected Balance E[B_N] : $2,218.47
  Theoretical Geometric Median B_N   : $2,194.92
  Monte Carlo Sample Mean            : $2,219.64
  Monte Carlo Mean 95% CI            : [$2,213.27, $2,226.00]
  Absolute Difference                : $+1.17 (+0.05%)
  Standard Error of the Mean         : $3.25
  Z-Score of Difference              : +0.36
  Statistical Consistency Check      : PASS (Theoretical value inside 95% CI)

[2] FINAL BALANCE DISTRIBUTION & MOMENTS
  Lowest Final Balance Observed      : $1,580.53 (1st pct) / min $1,172.61
  Highest Final Balance Observed     : $3,048.13 (99th pct) / max $4,493.43
  Median Final Balance               : $2,194.92
  Average (Mean) Final Balance       : $2,219.64
  Sample Standard Deviation          : $324.66
  Standardized Skewness              : +0.406 -> Right-skewed (long positive tail)
  Excess Kurtosis                    : +0.380 (leptokurtic)
  Probability of Finishing in Profit : 100.0% (95% CI: [100.0%, 100.0%])
  Probability Below Initial Capital  : 0.0% (95% CI: [0.0%, 0.0%])

[3] FINANCIAL TARGET SUCCESS RATES
 Target ($) | Success Rate | 95% Wilson CI     | Path-Hit Rate
------------|--------------|-------------------|--------------
      1,000 |      100.00% | [99.96%, 100.00%] | 100.00%
      1,500 |       99.46% | [99.30%, 99.59%] |  99.67%
      2,000 |       76.57% | [75.73%, 77.39%] |  77.29%
      2,500 |       17.87% | [17.13%, 18.63%] |  19.03%
      3,000 |        1.43% | [ 1.22%,  1.68%] |   1.56%
      3,500 |        0.02% | [ 0.01%,  0.07%] |   0.02%
      4,000 |        0.01% | [ 0.00%,  0.06%] |   0.01%

[4] RISK AND DRAWDOWN ANALYSIS
  Median Maximum Drawdown            : 4.9%
  95th Percentile Maximum Drawdown   : 7.7%
  Worst Drawdown Observed            : 14.0%
  95% Value at Risk (Loss Magnitude) : $-728.61 (-72.9%)
  95% Expected Shortfall (CVaR)      : $-626.65 (-62.7%)
  Terminal Ruin (<= $500)     : 0.00% (95% CI: [0.00%, 0.04%])
  Path Ruin (First Passage)          : 0.00% (95% CI: [0.00%, 0.04%])
========================================================================
```

---

## 6. Python API Examples

### Seeded Basic Simulation with Plot Export
```python
from quant_finance_lab import run_simulation, print_console_report
from quant_finance_lab.reporting.export import export_all_plots

result = run_simulation(
    initial_capital=1000.0,
    win_probability=0.60,
    win_return=0.02,
    loss_return=-0.01,
    num_simulations=10000,
    num_trades=100,
    seed=42,
)

print_console_report(result)
export_all_plots(result, "results/run_plots")
```

### Parameter Sensitivity Sweep (Common Random Numbers)
```python
from quant_finance_lab.analysis.sensitivity import sweep_parameter_1d
from quant_finance_lab import SimulationConfig

res_sens = sweep_parameter_1d(
    parameter_name="win_probability",
    parameter_values=[0.45, 0.50, 0.55, 0.60, 0.65],
    base_config=SimulationConfig(num_simulations=5000, seed=42),
    use_crn=True,
)
print(res_sens.to_dataframe())
```

---

## 7. Interactive Research Dashboard

Quant Finance Lab includes a full-featured Streamlit research dashboard for interactive exploration:

```bash
python -m quant_finance_lab dashboard
```

The dashboard allows researchers to interactively adjust model parameters, inspect sample equity curves, view drawdown histograms, evaluate financial targets with Wilson error bars, and execute sensitivity sweeps.

---

## 8. Project Structure

```text
quant-finance-lab/
├── README.md                           # Master project guide and documentation
├── pyproject.toml                      # Build metadata and dependency specifications
├── LICENSE                             # MIT License (preserving author attribution)
├── CHANGELOG.md                        # Version release history
├── CONTRIBUTING.md                     # Contribution guidelines
├── .gitignore                          # Clean repository exclusions
│
├── src/quant_finance_lab/              # Core library source code
│   ├── __init__.py                     # Clean top-level API exports
│   ├── __main__.py                     # python -m quant_finance_lab entrypoint
│   ├── cli.py                          # Full Command-Line Interface
│   │
│   ├── core/                           # System foundation and validation
│   │   ├── config.py                   # Configuration dataclasses (SimulationConfig)
│   │   ├── validation.py               # Parameter boundary validation
│   │   ├── random_state.py             # Reproducible SeedSequence RNG manager
│   │   └── exceptions.py               # Domain exception hierarchy
│   │
│   ├── models/                         # Stochastic return generating models
│   │   ├── base.py                     # ReturnModel abstract base class
│   │   ├── fixed_return.py             # Bernoulli binary fixed-return model
│   │   ├── variable_return.py          # Normal, Student-t fat-tailed models
│   │   ├── stochastic_processes.py     # Geometric Brownian Motion (GBM)
│   │   ├── market_regimes.py           # 2-state Markov regime-switching model
│   │   ├── costs.py                    # Transaction fee and slippage models
│   │   └── position_sizing.py          # Fixed fractional & Kelly allocation
│   │
│   ├── simulation/                     # Execution engine & analytical validation
│   │   ├── engine.py                   # Vectorized chunk-based simulation runner
│   │   ├── monte_carlo.py              # MonteCarloSimulator & SimulationResult
│   │   └── analytical_benchmarks.py    # Theoretical vs MC z-score benchmark
│   │
│   ├── metrics/                        # Statistical & financial metrics layer
│   │   ├── confidence.py               # Wilson score & Student's t intervals
│   │   ├── drawdown.py                 # Peak-to-trough series & MDD distribution
│   │   ├── risk.py                     # First-passage ruin, VaR & CVaR
│   │   ├── returns.py                  # Geometric growth & risk-adjusted metrics
│   │   └── distributions.py            # Moments, skewness, kurtosis, targets
│   │
│   ├── analysis/                       # Experimentation & comparative tools
│   │   ├── sensitivity.py              # 1D/2D parameter sweeps with CRN
│   │   ├── strategy_comparison.py      # Multi-model comparative framework
│   │   └── scenarios.py                # Pessimistic/Base/Optimistic stress tests
│   │
│   ├── data/                           # Historical price subsystem
│   │   ├── loaders.py                  # Validated CSV price bar loader
│   │   ├── validation.py               # OHLCV chronological schema integrity
│   │   └── returns.py                  # Empirical distribution calibration
│   │
│   ├── visualization/                  # Academic plotting suite (Matplotlib)
│   │   ├── style.py                    # Research styling & headless Agg backend
│   │   └── plots.py                    # Equity curves, histograms, heatmaps
│   │
│   ├── reporting/                      # Structured output generators
│   │   ├── console.py                  # Formatted ASCII terminal report
│   │   ├── tables.py                   # Formatted statistical tables
│   │   ├── export.py                   # JSON metadata, CSV paths & plot exports
│   │   └── metadata.py                 # Runtime environment capture
│   │
│   └── dashboard/                      # Interactive web interface
│       └── app.py                      # Streamlit dashboard application
│
├── tests/                              # Automated test suite (34 tests)
│   ├── unit/                           # Validation, model, metric unit tests
│   ├── deterministic/                  # Edge cases (p=0, p=1, zero trades, ruin)
│   ├── statistical/                    # LLN convergence & confidence coverage
│   ├── integration/                    # End-to-end simulation & CLI tests
│   └── regression/                     # Continuity with legacy v0.0.1 script
│
├── examples/                           # Runnable research scripts
│   ├── basic_simulation.py             # Basic reproducible simulation
│   ├── analytical_validation.py        # Convergence against theoretical E[B_N]
│   ├── drawdown_analysis.py            # Peak drawdowns & ruin threshold tests
│   ├── sensitivity_analysis.py         # 1D and 2D parameter sweeps
│   ├── strategy_comparison.py          # Comparative evaluation of 4 strategies
│   └── benchmark_performance.py        # High-throughput benchmark measurement
│
├── docs/                               # Detailed technical documentation
│   ├── mathematical_foundations.md     # Equations, derivations & worked examples
│   ├── statistical_methods.md          # Monte Carlo uncertainty vs model risk
│   ├── architecture.md                 # Design principles & extension guides
│   ├── assumptions_and_limitations.md  # Explicit boundaries & constraints
│   ├── user_guide.md                   # CLI, Python API, and dashboard guide
│   └── roadmap.md                      # Release milestones and future research
│
├── legacy/                             # Preserved original author implementations
│   ├── stochastic_analytics_simulation_v0.0.1.py
│   ├── stochastic_analytics_simulation_v0.0.2.py
│   └── README.original.md
│
└── results/                            # Generated simulation charts & JSON exports
    └── .gitkeep
```

---

## 9. Running Tests

The test suite contains **34 automated tests** covering unit, deterministic, statistical, integration, and regression specifications:

```bash
# Run all tests
python -m pytest

# Run specific test suites
python -m pytest tests/unit
python -m pytest tests/deterministic
python -m pytest tests/statistical
python -m pytest tests/integration
python -m pytest tests/regression
```

---

## 10. Performance Benchmarks

Measured on Windows 11 with Python 3.14.6 and NumPy 2.5.4:

| Simulations | Trades | Mode | Time (s) | Throughput (paths/sec) |
|---|---|---|---|---|
| 1,000 | 100 | Full Paths | 0.0069 | 144,209 |
| 10,000 | 100 | Full Paths | 0.0594 | 168,314 |
| 50,000 | 100 | Full Paths | 0.2645 | 189,052 |
| 100,000 | 100 | Memory-Safe | 0.2196 | **455,384** |

---

## 11. Assumptions & Limitations

* **Model Assumptions**: The core model assumes stationary parameters, independent trade outcomes, infinite liquidity, and frictionless fills unless transaction cost parameters are configured.
* **Sampling vs Parameter Uncertainty**: Monte Carlo confidence intervals quantify only the random sampling variation of the simulation; they do not eliminate uncertainty in estimated market parameters.
* **Disclaimer**: This software is intended strictly for quantitative research and educational exploration. Simulated outcomes do not guarantee real-world returns and do not constitute financial advice.

---

## 12. Attribution & License

* **Original Concept & Vision**: Created by **Manish Sai** to explore programming, mathematics, and quantitative finance.
* **License**: Released under the [MIT License](LICENSE).
