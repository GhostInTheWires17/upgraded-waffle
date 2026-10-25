# Changelog

All notable changes to Quant Finance Lab are documented in this file.

## [1.0.0] - 2026-10-20

### Major Engineering Remake
- **Core Architecture**: Refactored monolithic scripts (`v0.0.1`, `v0.0.2`) into a modular, professional quantitative research library `quant_finance_lab`.
- **Vectorized Engine**: Replaced serial loops with NumPy-vectorized simulation engine and chunked memory processing.
- **Analytical Benchmark**: Implemented exact mathematical expectation $\mathbb{E}[B_N] = B_0 [p(1+R_{\text{win}}) + (1-p)(1+R_{\text{loss}})]^N$, log-space geometric median, and statistical z-score validation.
- **Statistical Reliability**: Added Wilson score confidence intervals for all target, ruin, and profit probabilities; Student's t confidence intervals for expected balances.
- **Risk Layer**: Added peak-to-trough path drawdown tracking, Maximum Drawdown (MDD) distributions, Value-at-Risk (95%, 99%), and Expected Shortfall (CVaR).
- **Alternative Stochastic Models**: Added continuous Normal, fat-tailed Student-t, Geometric Brownian Motion (GBM), and Markov Regime-Switching models.
- **Trading Friction & Sizing**: Added transaction cost models (fixed fees, proportional slippage) and position sizing (fractional, fixed dollar, Kelly criterion).
- **Sensitivity & Scenarios**: Added 1D and 2D parameter sweeps with Common Random Numbers (CRN) and scenario stress tests.
- **Data Subsystem**: Added historical OHLCV CSV loader, integrity validation, and empirical parameter calibration.
- **Visualization**: Added publication-grade Matplotlib plots with clean academic styling and automated export.
- **Command-Line Interface**: Built full CLI (`simulate`, `sensitivity`, `scenarios`, `dashboard`).
- **Interactive Dashboard**: Built interactive Streamlit research dashboard.
- **Test Suite**: Added 34 tests covering unit, deterministic edge cases, statistical convergence, integration, and regression continuity.
- **Documentation**: Rewrote complete README, mathematical foundations, architecture, and user guides.
