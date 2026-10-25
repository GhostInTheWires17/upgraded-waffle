# Mathematical Foundations of Quant Finance Lab

This document details the mathematical formulations, derivations, statistical assumptions, and worked analytical examples underpinning Quant Finance Lab.

---

## 1. Multiplicative Capital Compounding

### 1.1 Capital State Transition
Let $B_0 > 0$ denote the initial capital at time $t = 0$. Each trade $t \in \{1, 2, \dots, N\}$ generates a fractional return rate $R_t \in [-1, \infty)$, where:
* $R_t > 0$ represents a winning trade (e.g. $+0.02$ for a 2% gain).
* $R_t < 0$ represents a losing trade (e.g. $-0.01$ for a 1% loss).
* $R_t = -1.0$ represents total capital ruin (100% loss).

Under full reinvestment (frictionless compounding), capital evolves sequentially according to the recurrence relation:
$$B_t = B_{t-1} (1 + R_t)$$

Expanding this product over $N$ trades yields the closed-form path representation:
$$B_N = B_0 \prod_{t=1}^N (1 + R_t)$$

In logarithmic space, this corresponds to an additive random walk:
$$\ln B_N = \ln B_0 + \sum_{t=1}^N \ln(1 + R_t)$$

---

## 2. The Core Bernoulli Fixed-Return Model

### 2.1 Formulation and Assumptions
The model assumes that trades are independent and identically distributed (i.i.d.) Bernoulli trials with stationary win probability $p \in [0, 1]$:

$$1 + R_t = \begin{cases} 1 + R_{\text{win}} & \text{with probability } p \\ 1 + R_{\text{loss}} & \text{with probability } 1 - p \end{cases}$$

where $R_{\text{win}} \ge 0$ and $-1 \le R_{\text{loss}} \le 0$.

### 2.2 Analytical Expected Single-Trade Return
The expected single-trade arithmetic return is:
$$\mathbb{E}[R_t] = p R_{\text{win}} + (1 - p) R_{\text{loss}}$$

The single-trade multiplier expectation is:
$$\mathbb{E}[1 + R_t] = 1 + \mathbb{E}[R_t] = p (1 + R_{\text{win}}) + (1 - p)(1 + R_{\text{loss}})$$

The single-trade variance is:
$$\text{Var}[R_t] = \mathbb{E}[R_t^2] - (\mathbb{E}[R_t])^2 = p(1 - p)(R_{\text{win}} - R_{\text{loss}})^2$$

### 2.3 Analytical Expected Final Balance $\mathbb{E}[B_N]$
Because the individual trade outcomes $R_1, R_2, \dots, R_N$ are mutually independent:
$$\mathbb{E}[B_N] = \mathbb{E}\left[ B_0 \prod_{t=1}^N (1 + R_t) \right] = B_0 \prod_{t=1}^N \mathbb{E}[1 + R_t]$$

Substituting the single-trade multiplier yields the exact theoretical formula:
$$\mathbb{E}[B_N] = B_0 \left[ p(1 + R_{\text{win}}) + (1 - p)(1 + R_{\text{loss}}) \right]^N$$

#### Worked Example:
Given parameters:
* $B_0 = \$1,000$
* $p = 0.60$
* $R_{\text{win}} = +0.02$
* $R_{\text{loss}} = -0.01$
* $N = 100$

Step 1: Calculate expected multiplier per trade:
$$\mathbb{E}[1 + R_t] = 0.60(1 + 0.02) + 0.40(1 - 0.01) = 0.612 + 0.396 = 1.008$$

Step 2: Compound over 100 trades:
$$\mathbb{E}[B_{100}] = 1,000 \times (1.008)^{100} \approx 1,000 \times 2.218468 = \$2,218.47$$

---

## 3. The Geometric Growth Rate and Skewness

### 3.1 Geometric Growth Rate (Kelly Median)
While $\mathbb{E}[B_N]$ describes the ensemble mean across millions of parallel universes, an individual trajectory's typical growth rate is governed by the expected logarithmic return:
$$g = \mathbb{E}[\ln(1 + R_t)] = p \ln(1 + R_{\text{win}}) + (1 - p) \ln(1 + R_{\text{loss}})$$

By the Law of Large Numbers, the sample fraction of wins in $N$ trades approaches $p$. Therefore, the median terminal capital satisfies:
$$\text{Median}[B_N] \approx B_0 \exp(N g) = B_0 (1 + R_{\text{win}})^{N p} (1 + R_{\text{loss}})^{N(1 - p)}$$

#### Numerical Comparison:
Using the same parameters ($B_0 = 1000, p = 0.60, R_w = 0.02, R_l = -0.01, N = 100$):
$$g = 0.60 \ln(1.02) + 0.40 \ln(0.99) \approx 0.60(0.019803) + 0.40(-0.010050) = 0.011882 - 0.004020 = 0.0078615$$
$$\text{Median}[B_{100}] \approx 1,000 \times e^{100 \times 0.0078615} = 1,000 \times e^{0.78615} \approx \$2,194.92$$

### 3.2 Right-Skewness (Jensen's Inequality)
By Jensen's Inequality, because the exponential function $f(x) = e^x$ is strictly convex:
$$\mathbb{E}[e^X] > e^{\mathbb{E}[X]}$$
Setting $X = \sum_{t=1}^N \ln(1 + R_t)$, we have:
$$\mathbb{E}[B_N] > \text{Median}[B_N]$$
$$\$2,218.47 > \$2,194.92$$

This mathematically explains why the simulated final balance distribution is invariably **right-skewed**: a large mass of outcomes clusters around the median, while a small fraction of exceptionally favorable sequence paths stretches a long tail to the right.

---

## 4. Drawdown Mathematics

### 4.1 Running Peak and Drawdown Process
For a simulated balance trajectory $B = \{B_0, B_1, \dots, B_N\}$:
$$\text{Peak}_t = \max_{0 \le s \le t} B_s, \quad t \in \{0, 1, \dots, N\}$$

The relative peak-to-trough drawdown at step $t$ is:
$$\text{Drawdown}_t = \frac{B_t}{\text{Peak}_t} - 1 \in [-1, 0]$$

### 4.2 Maximum Drawdown (MDD)
The Maximum Drawdown magnitude over the entire horizon is:
$$\text{MDD} = \max_{0 \le t \le N} \left( 1 - \frac{B_t}{\text{Peak}_t} \right) = -\min_{0 \le t \le N} \text{Drawdown}_t \in [0, 1]$$

---

## 5. Statistical Reliability: The Wilson Score Interval

### 5.1 Estimating Event Probabilities
Let $K$ be the number of simulations out of $M$ that satisfy an event (e.g., reaching target $T$ or ruin threshold).
The naive point estimate is:
$$\hat{p} = \frac{K}{M}$$

The traditional Wald confidence interval $\hat{p} \pm z \sqrt{\hat{p}(1-\hat{p})/M}$ performs poorly for extreme proportions ($\hat{p} \approx 0$ or $1$) and can produce bounds outside $[0, 1]$.

### 5.2 Wilson Score Interval Formulation
Quant Finance Lab uses the Wilson Score Interval (Wilson, 1927). Inverting the score test yields the confidence limits:
$$\tilde{p} = \frac{\hat{p} + \frac{z^2}{2M}}{1 + \frac{z^2}{M}}$$
$$w = \frac{z}{1 + \frac{z^2}{M}} \sqrt{\frac{\hat{p}(1 - \hat{p})}{M} + \frac{z^2}{4M^2}}$$
$$\text{CI}_{1-\alpha} = [\max(0, \tilde{p} - w), \min(1, \tilde{p} + w)]$$

where $z = \Phi^{-1}(1 - \alpha/2)$ is the standard normal critical value.

---

## 6. Tail Risk: Value-at-Risk and Expected Shortfall

### 6.1 Dollar Loss Convention
Define total loss over the horizon as $L = B_0 - B_N$. Positive $L$ represents capital destruction.

### 6.2 Value at Risk ($\text{VaR}_\alpha$)
At confidence level $\alpha \in (0, 1)$ (typically 0.95 or 0.99):
$$\text{VaR}_\alpha = \inf \{ l \in \mathbb{R} : \Pr(L \le l) \ge \alpha \}$$

### 6.3 Expected Shortfall ($\text{CVaR}_\alpha$)
Expected Shortfall measures the conditional expectation of loss given that loss exceeds $\text{VaR}_\alpha$:
$$\text{ES}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha]$$
CVaR is a coherent risk measure satisfying sub-additivity: $\text{CVaR}(X + Y) \le \text{CVaR}(X) + \text{CVaR}(Y)$.
