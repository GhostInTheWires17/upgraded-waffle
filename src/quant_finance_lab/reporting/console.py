"""Clean, research-grade terminal console reporting."""

from __future__ import annotations

from quant_finance_lab.simulation.monte_carlo import SimulationResult
from quant_finance_lab.reporting.tables import format_target_table, format_drawdown_table


def format_console_report(result: SimulationResult) -> str:
    """Format complete research simulation results into a comprehensive console report."""
    cfg = result.config
    m = result.model
    dm = result.distribution_metrics
    rm = result.risk_metrics
    dd = result.drawdown_metrics
    bench = result.analytical_benchmark

    lines = []
    lines.append("=" * 72)
    lines.append(" QUANT FINANCE LAB — STOCHASTIC ANALYTICS SIMULATION")
    lines.append("=" * 72)
    lines.append(f"Model: {m.name}")
    lines.append(f"Simulations (Paths): {cfg.num_simulations:,}  |  Trades per Path: {cfg.num_trades:,}")
    lines.append(f"Initial Capital: ${cfg.initial_capital:,.2f}  |  Random Seed: {cfg.seed}")
    lines.append(f"Execution Time: {result.execution_time_seconds:.3f} seconds")
    lines.append("-" * 72)

    # Analytical Benchmark Comparison
    if bench:
        lines.append("\n[1] THEORETICAL BENCHMARK VS MONTE CARLO ESTIMATION")
        lines.append(f"  Theoretical Expected Balance E[B_N] : ${bench.theoretical_expected_balance:,.2f}")
        if bench.theoretical_median_balance is not None:
            lines.append(f"  Theoretical Geometric Median B_N   : ${bench.theoretical_median_balance:,.2f}")
        lines.append(f"  Monte Carlo Sample Mean            : ${bench.mc_estimated_mean:,.2f}")
        lines.append(f"  Monte Carlo Mean 95% CI            : [${dm.mean_ci.lower:,.2f}, ${dm.mean_ci.upper:,.2f}]")
        lines.append(f"  Absolute Difference                : ${bench.absolute_error:+,.2f} ({bench.relative_error_pct:+.2f}%)")
        lines.append(f"  Standard Error of the Mean         : ${bench.standard_error:,.2f}")
        lines.append(f"  Z-Score of Difference              : {bench.z_score:+.2f}")
        ci_status = "PASS (Theoretical value inside 95% CI)" if bench.is_within_confidence_interval else "MARGINAL"
        lines.append(f"  Statistical Consistency Check      : {ci_status}")

    # Final Balance Distribution
    lines.append("\n[2] FINAL BALANCE DISTRIBUTION & MOMENTS")
    lines.append(f"  Lowest Final Balance Observed      : ${dm.percentiles['p1']:,.2f} (1st pct) / min ${result.return_metrics.min_final_balance:,.2f}")
    lines.append(f"  Highest Final Balance Observed     : ${dm.percentiles['p99']:,.2f} (99th pct) / max ${result.return_metrics.max_final_balance:,.2f}")
    lines.append(f"  Median Final Balance               : ${dm.median:,.2f}")
    lines.append(f"  Average (Mean) Final Balance       : ${dm.mean:,.2f}")
    lines.append(f"  Sample Standard Deviation          : ${dm.std_dev:,.2f}")
    
    skew_msg = "Right-skewed (long positive tail)" if dm.skewness > 0.05 else ("Left-skewed" if dm.skewness < -0.05 else "Symmetric")
    lines.append(f"  Standardized Skewness              : {dm.skewness:+.3f} -> {skew_msg}")
    lines.append(f"  Excess Kurtosis                    : {dm.excess_kurtosis:+.3f} (leptokurtic)" if dm.excess_kurtosis > 0 else f"  Excess Kurtosis: {dm.excess_kurtosis:+.3f}")
    lines.append(f"  Probability of Finishing in Profit : {dm.probability_profit.point_estimate:.1%} (95% CI: [{dm.probability_profit.lower:.1%}, {dm.probability_profit.upper:.1%}])")
    lines.append(f"  Probability Below Initial Capital  : {dm.probability_loss.point_estimate:.1%} (95% CI: [{dm.probability_loss.lower:.1%}, {dm.probability_loss.upper:.1%}])")

    # Target Probability Table
    if dm.target_probabilities:
        lines.append("\n[3] FINANCIAL TARGET SUCCESS RATES")
        lines.append(format_target_table(dm.target_probabilities))

    # Drawdown & Ruin Risk
    lines.append("\n[4] RISK AND DRAWDOWN ANALYSIS")
    lines.append(f"  Median Maximum Drawdown            : {dd.median_max_drawdown:.1%}")
    lines.append(f"  95th Percentile Maximum Drawdown   : {dd.percentile_95_max_drawdown:.1%}")
    lines.append(f"  Worst Drawdown Observed            : {dd.worst_drawdown_observed:.1%}")
    lines.append(f"  95% Value at Risk (Loss Magnitude) : ${rm.var_95_dollar:,.2f} ({rm.var_95_pct:.1%})")
    lines.append(f"  95% Expected Shortfall (CVaR)      : ${rm.cvar_95_dollar:,.2f} ({rm.cvar_95_pct:.1%})")
    if rm.terminal_ruin_probability:
        lines.append(f"  Terminal Ruin (<= ${cfg.ruin_threshold:,.0f})     : {rm.terminal_ruin_probability.point_estimate:.2%} (95% CI: [{rm.terminal_ruin_probability.lower:.2%}, {rm.terminal_ruin_probability.upper:.2%}])")
    if rm.path_ruin_probability:
        lines.append(f"  Path Ruin (First Passage)          : {rm.path_ruin_probability.point_estimate:.2%} (95% CI: [{rm.path_ruin_probability.lower:.2%}, {rm.path_ruin_probability.upper:.2%}])")

    lines.append("=" * 72)
    return "\n".join(lines)


def print_console_report(result: SimulationResult) -> None:
    """Print the formatted console report to standard output."""
    print(format_console_report(result))
