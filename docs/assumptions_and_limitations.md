# Assumptions and Limitations

Quant Finance Lab is an educational and mathematical research platform. All simulation results must be interpreted within the context of the underlying assumptions and statistical boundaries.

---

## 1. Mathematical Assumptions

### 1.1 Independence of Trade Outcomes (i.i.d.)
The baseline `FixedReturnModel` assumes that every trade's outcome is strictly independent of prior trades:
$$\Pr(R_t \mid R_1, \dots, R_{t-1}) = \Pr(R_t)$$
* **Real-World Reality**: Real financial assets exhibit volatility clustering (Mandlebrot, 1963; Engle, 1982), momentum, mean-reversion, and autocorrelation. Consecutive losses occur in clusters during market sell-offs.

### 1.2 Constant Parameter Stationarity
The win probability $p$, winning return $R_{\text{win}}$, and losing return $R_{\text{loss}}$ are treated as fixed constants over the entire horizon:
* **Real-World Reality**: Market efficiency, macro regimes, and algorithmic competition cause quantitative edges to decay or fluctuate across time.

### 1.3 Infinite Liquidity and Zero Slippage
Unless explicitly configured in `TransactionCostModel`, simulations assume instantaneous fills at exact target prices without size-dependent market impact:
* **Real-World Reality**: Large positions suffer slippage, execution delay, and liquidity shortages during market crashes.

---

## 2. Statistical Limitations

### 2.1 Sample Size and Tail Instability
Estimating tail events such as 99% Value-at-Risk or 1-in-1,000 extreme drawdowns requires substantial simulation volumes:
* With $M = 1,000$ simulations, the 99th percentile represents only 10 empirical paths, making the estimate susceptible to high variance.
* For stable tail-risk analysis, use $M \ge 20,000$ simulations.

### 2.2 Point Estimates vs Realized Outcomes
An estimated $75\%$ probability of reaching a target is a conditional mathematical estimate based on the model's assumed parameters. It does **not** imply that an investor in financial markets has a $75\%$ chance of profit.

---

## 3. Disclaimers

Quant Finance Lab does not provide personalized investment advice, trade recommendations, or financial guarantees. Simulated performance does not reflect live trading results and does not account for execution risks, regulatory constraints, tax implications, or systemic market events.
