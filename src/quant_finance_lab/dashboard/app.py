"""Interactive Streamlit research dashboard for Quant Finance Lab."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.models.variable_return import NormalReturnModel, StudentTReturnModel
from quant_finance_lab.models.stochastic_processes import GeometricBrownianMotion
from quant_finance_lab.models.costs import TransactionCostModel
from quant_finance_lab.models.position_sizing import PositionSizer
from quant_finance_lab.simulation.monte_carlo import MonteCarloSimulator
from quant_finance_lab.visualization.plots import (
    plot_equity_curves,
    plot_final_balance_distribution,
    plot_drawdown_distribution,
    plot_target_probabilities,
)
from quant_finance_lab.analysis.sensitivity import sweep_parameter_1d

st.set_page_config(page_title="Quant Finance Lab", layout="wide", page_icon="📈")

st.title("📈 Quant Finance Lab")
st.markdown(
    "**An educational quantitative finance and stochastic simulation laboratory.**  \n"
    "Explore how trading edge, win probability, compounding, and risk-reward dynamics shape final wealth under uncertainty."
)

# Sidebar: Controls
st.sidebar.header("🔬 Experiment Configuration")
model_type = st.sidebar.selectbox(
    "Return Generating Model",
    ["Fixed Return (Bernoulli)", "Normal Returns", "Student-t (Fat-Tailed)", "Geometric Brownian Motion (GBM)"],
)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    initial_capital = st.number_input("Initial Capital ($)", min_value=10.0, value=1000.0, step=100.0)
    num_trades = st.number_input("Trades per Path", min_value=1, value=100, step=10)
with col_s2:
    num_simulations = st.number_input("Simulations (Paths)", min_value=100, value=10000, step=1000)
    seed = st.number_input("Seed (0 = Random)", min_value=0, value=42, step=1)

actual_seed = seed if seed > 0 else None

# Model-specific parameters
if model_type == "Fixed Return (Bernoulli)":
    st.sidebar.subheader("Bernoulli Parameters")
    win_prob = st.sidebar.slider("Win Probability (%)", min_value=0.0, max_value=100.0, value=60.0) / 100.0
    win_ret = st.sidebar.number_input("Win Return (%)", min_value=0.0, value=2.0, step=0.5) / 100.0
    loss_ret = -st.sidebar.number_input("Loss Return (%) [Magnitude]", min_value=0.0, max_value=100.0, value=1.0, step=0.5) / 100.0
    model = FixedReturnModel(win_probability=win_prob, win_return=win_ret, loss_return=loss_ret)
elif model_type == "Normal Returns":
    st.sidebar.subheader("Normal Distribution Parameters")
    mean_ret = st.sidebar.number_input("Mean Return per Trade (%)", value=0.10, step=0.05) / 100.0
    vol = st.sidebar.number_input("Volatility per Trade (%)", min_value=0.01, value=2.0, step=0.5) / 100.0
    model = NormalReturnModel(mean_return=mean_ret, volatility=vol)
elif model_type == "Student-t (Fat-Tailed)":
    st.sidebar.subheader("Student-t Parameters")
    df = st.sidebar.number_input("Degrees of Freedom (nu > 2)", min_value=2.1, value=5.0, step=0.5)
    loc = st.sidebar.number_input("Location / Mean (%)", value=0.10, step=0.05) / 100.0
    scale = st.sidebar.number_input("Scale parameter (%)", min_value=0.01, value=1.5, step=0.5) / 100.0
    model = StudentTReturnModel(degrees_of_freedom=df, loc=loc, scale=scale)
else:
    st.sidebar.subheader("GBM Parameters")
    drift = st.sidebar.number_input("Annual Drift (%)", value=8.0, step=1.0) / 100.0
    vol = st.sidebar.number_input("Annual Volatility (%)", min_value=1.0, value=20.0, step=1.0) / 100.0
    model = GeometricBrownianMotion(annual_drift=drift, annual_volatility=vol)

# Risk Thresholds
st.sidebar.subheader("Risk & Friction")
ruin_thresh = st.sidebar.number_input("Ruin Threshold ($)", min_value=0.0, value=initial_capital * 0.5, step=50.0)
enable_costs = st.sidebar.checkbox("Enable Transaction Costs", value=False)
fixed_fee = st.sidebar.number_input("Fixed Cost per Trade ($)", min_value=0.0, value=0.50, step=0.10) if enable_costs else 0.0
pct_fee = st.sidebar.number_input("Proportional Fee (%)", min_value=0.0, value=0.05, step=0.01) / 100.0 if enable_costs else 0.0

cost_model = TransactionCostModel(fixed_cost=fixed_fee, proportional_cost=pct_fee) if enable_costs else None

# Targets
target_str = st.sidebar.text_input("Target Balances ($)", "1000, 1500, 2000, 2500, 3000, 3500, 4000")
targets = [float(x.strip()) for x in target_str.split(",") if x.strip()]

cfg = SimulationConfig(
    initial_capital=initial_capital,
    num_simulations=num_simulations,
    num_trades=num_trades,
    seed=actual_seed,
    targets=targets,
    ruin_threshold=ruin_thresh if ruin_thresh < initial_capital else None,
    store_paths=True,
)

# Run Simulation
sim = MonteCarloSimulator(config=cfg, model=model, cost_model=cost_model)
res = sim.run()

# High-Level Metric KPIs
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("MC Mean Final Balance", f"${res.distribution_metrics.mean:,.2f}")
with kpi2:
    st.metric("MC Median Final Balance", f"${res.distribution_metrics.median:,.2f}")
with kpi3:
    st.metric("Profit Probability", f"{res.distribution_metrics.probability_profit.point_estimate:.1%}")
with kpi4:
    st.metric("Median Max Drawdown", f"{res.drawdown_metrics.median_max_drawdown:.1%}")
with kpi5:
    ruin_val = (
        f"{res.risk_metrics.terminal_ruin_probability.point_estimate:.1%}"
        if res.risk_metrics.terminal_ruin_probability
        else "N/A"
    )
    st.metric("Ruin Probability", ruin_val)

if res.analytical_benchmark:
    st.info(
        f"**Theoretical Expected Balance E[B_N]:** ${res.analytical_benchmark.theoretical_expected_balance:,.2f}  |  "
        f"**Monte Carlo Error:** ${res.analytical_benchmark.absolute_error:+,.2f} ({res.analytical_benchmark.relative_error_pct:+.2f}%)  |  "
        f"**95% Confidence Interval:** [${res.distribution_metrics.mean_ci.lower:,.2f}, ${res.distribution_metrics.mean_ci.upper:,.2f}]"
    )

# Visualizations Tabs
tab_paths, tab_dist, tab_drawdown, tab_targets, tab_sens, tab_math = st.tabs(
    ["Equity Curves", "Balance Distribution", "Drawdowns & Risk", "Target Success Rates", "Sensitivity Sweep", "Mathematical Foundations"]
)

with tab_paths:
    fig_paths = plot_equity_curves(res, num_sample_paths=50)
    st.pyplot(fig_paths)

with tab_dist:
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        fig_dist = plot_final_balance_distribution(res)
        st.pyplot(fig_dist)
    with col_d2:
        st.subheader("Statistical Moments")
        st.write(f"- **Sample Mean:** ${res.distribution_metrics.mean:,.2f}")
        st.write(f"- **Sample Median:** ${res.distribution_metrics.median:,.2f}")
        st.write(f"- **Standard Deviation:** ${res.distribution_metrics.std_dev:,.2f}")
        st.write(f"- **Skewness:** {res.distribution_metrics.skewness:+.3f}")
        st.write(f"- **Excess Kurtosis:** {res.distribution_metrics.excess_kurtosis:+.3f}")
        st.write(f"- **Lowest Observed:** ${res.return_metrics.min_final_balance:,.2f}")
        st.write(f"- **Highest Observed:** ${res.return_metrics.max_final_balance:,.2f}")

with tab_drawdown:
    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        fig_dd = plot_drawdown_distribution(res)
        st.pyplot(fig_dd)
    with col_w2:
        st.subheader("Risk & Tail Metrics")
        st.write(f"- **Median Max Drawdown:** {res.drawdown_metrics.median_max_drawdown:.1%}")
        st.write(f"- **95th Percentile MDD:** {res.drawdown_metrics.percentile_95_max_drawdown:.1%}")
        st.write(f"- **Worst Drawdown Observed:** {res.drawdown_metrics.worst_drawdown_observed:.1%}")
        st.write(f"- **95% VaR (Loss):** ${res.risk_metrics.var_95_dollar:,.2f} ({res.risk_metrics.var_95_pct:.1%})")
        st.write(f"- **95% Expected Shortfall:** ${res.risk_metrics.cvar_95_dollar:,.2f} ({res.risk_metrics.cvar_95_pct:.1%})")

with tab_targets:
    fig_t = plot_target_probabilities(res)
    st.pyplot(fig_t)

with tab_sens:
    st.subheader("1D Parameter Sensitivity Sweep")
    sweep_param = st.selectbox("Select Parameter to Sweep", ["win_probability", "win_return", "loss_return", "num_trades"])
    if st.button("Run Parameter Sweep"):
        with st.spinner("Simulating parameter sweep..."):
            vals = np.linspace(0.40, 0.70, 7) if sweep_param == "win_probability" else np.linspace(10, 200, 10)
            res_sens = sweep_parameter_1d(sweep_param, vals, base_config=cfg)
            st.dataframe(res_sens.to_dataframe(), use_container_width=True)

with tab_math:
    st.markdown(
        """
        ### Multiplicative Compounding Equation
        $$B_{t+1} = B_t \\times (1 + R_t)$$
        
        ### Theoretical Expected Final Balance
        For independent binary trials with win probability $p$, win return $R_{win}$, and loss return $R_{loss}$:
        $$\\mathbb{E}[B_N] = B_0 \\left[ p(1 + R_{win}) + (1 - p)(1 + R_{loss}) \\right]^N$$
        
        ### Theoretical Geometric Growth Rate (Kelly Median)
        $$g = \\mathbb{E}[\\ln(1 + R_t)] = p \\ln(1 + R_{win}) + (1 - p) \\ln(1 + R_{loss})$$
        $$\\text{Median}[B_N] \\approx B_0 e^{N g}$$
        
        Multiplicative dynamics naturally induce right-skewed lognormal-like distributions, explaining why $\\mathbb{E}[B_N] > \\text{Median}[B_N]$.
        """
    )
