"""Example 3: Path Drawdowns, Running Peaks, and Ruin Probability.

Investigates sequential path behavior, peak-to-trough drawdowns, and ruin-crossing probabilities.
"""

from quant_finance_lab.simulation.monte_carlo import run_simulation


def main() -> None:
    print("=" * 72)
    print(" QUANT FINANCE LAB — DRAWDOWN AND RUIN ANALYSIS")
    print("=" * 72)

    # Moderate edge strategy with 10,000 paths
    res = run_simulation(
        initial_capital=1000.0,
        win_probability=0.55,
        win_return=0.02,
        loss_return=-0.015,
        num_simulations=10000,
        num_trades=150,
        seed=101,
        ruin_threshold=500.0,  # 50% capital loss
        store_paths=True,
    )

    dd = res.drawdown_metrics
    rm = res.risk_metrics

    print(f"Strategy Edge: 55% Win Rate | +2.0% Win / -1.5% Loss | 150 Trades")
    print(f"Mean Final Balance   : ${res.distribution_metrics.mean:,.2f}")
    print(f"Median Final Balance : ${res.distribution_metrics.median:,.2f}")
    print("-" * 72)
    print("DRAWDOWN METRICS:")
    print(f"  Median Maximum Drawdown          : {dd.median_max_drawdown:.1%}")
    print(f"  90th Percentile Maximum Drawdown : {dd.percentile_90_max_drawdown:.1%}")
    print(f"  95th Percentile Maximum Drawdown : {dd.percentile_95_max_drawdown:.1%}")
    print(f"  Worst Drawdown Observed          : {dd.worst_drawdown_observed:.1%}")
    print("-" * 72)
    print("RUIN RISK COMPARISON:")
    term_ruin = rm.terminal_ruin_probability.point_estimate
    path_ruin = rm.path_ruin_probability.point_estimate
    print(f"  Terminal Ruin (Ending below $500)      : {term_ruin:.2%}")
    print(f"  Path Ruin (Crossing $500 at any point) : {path_ruin:.2%}")
    print("\nInsight: Notice that Path Ruin probability is significantly higher than Terminal Ruin,")
    print("because paths may dip below $500 during severe drawdowns and later recover.")
    print("=" * 72)


if __name__ == "__main__":
    main()
