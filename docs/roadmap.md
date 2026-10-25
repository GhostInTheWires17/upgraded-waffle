# Quant Finance Lab: Research Roadmap

This roadmap documents the completed milestones in Version 1.0.0 and outlines future exploration directions for the quantitative laboratory.

---

## Completed Milestones (Version 1.0.0)

- [x] **Core Vectorized Monte Carlo Engine**: 100x speedup over sequential Python loops, processing >150,000 paths/second.
- [x] **Analytical Benchmarks**: Closed-form mathematical expected balance $\mathbb{E}[B_N]$ and geometric median derivation with statistical hypothesis testing against Monte Carlo outputs.
- [x] **Statistical Reliability**: Wilson score confidence intervals for Bernoulli event probabilities; Student's t intervals for expected balances.
- [x] **Path Drawdown & Ruin Analytics**: True running peak drawdown calculation, MDD distributions, and first-passage ruin vs terminal ruin comparison.
- [x] **Tail Risk Metrics**: Value-at-Risk (VaR) and Expected Shortfall (CVaR) with explicit sign conventions.
- [x] **Continuous & Alternative Stochastic Models**: Gaussian, fat-tailed Student-t, Geometric Brownian Motion (GBM), and Markov Regime-Switching models.
- [x] **Friction & Position Sizing**: Transaction fees, basis-point slippage, fixed-fractional sizing, and Kelly Criterion calculators.
- [x] **Sensitivity & Scenario Engine**: 1D parameter sweeps, 2D grid sweeps with Common Random Numbers (CRN), and pessimistic/base/optimistic scenarios.
- [x] **Historical CSV Subsystem**: Isolated price loader with strict chronological validation and empirical return calibration.
- [x] **Publication-Grade Visualization**: Clean Matplotlib visualizers for equity curves, distributions, drawdowns, and error-bar target charts.
- [x] **Comprehensive Test Suite**: 34 automated unit, deterministic, statistical, integration, and regression tests.
- [x] **Interactive Streamlit Dashboard**: Web UI for intuitive visual parameter exploration.

---

## Future Research Directions

### 1. Advanced Time-Series Dynamics
* **GARCH(1,1) and EGARCH Volatility Clustering**: Incorporate conditional heteroskedasticity into trade return sequences.
* **Jump-Diffusion Models (Merton Jump Diffusion)**: Simulate discontinuous market shocks and flash crashes.
* **Fractional Brownian Motion (Hurst Exponent)**: Model long-memory and mean-reverting asset dynamics ($H \neq 0.5$).

### 2. Multi-Asset Portfolio Simulation
* **Correlation Matrices & Cholesky Decomposition**: Simulate correlated multi-strategy and multi-asset portfolios.
* **Dynamic Rebalancing & Portfolio Sizing**: Compare risk-parity, equal-weight, and mean-variance optimal allocations.

### 3. Execution & Market Microstructure
* **Limit Order Book (LOB) Queue Dynamics**: Model adverse selection and queue position risk.
* **Almgren-Chriss Optimal Execution**: Non-linear market impact under volume-weighted time decay.
