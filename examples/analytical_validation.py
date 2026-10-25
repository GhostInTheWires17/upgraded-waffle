"""Example 2: Analytical Validation and Monte Carlo Convergence.

Demonstrates that Monte Carlo sample mean converges to the closed-form
theoretical expectation E[B_N] as the number of simulated paths increases,
illustrating the Law of Large Numbers and Monte Carlo standard error decay (1 / sqrt(M)).
"""

from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.simulation.monte_carlo import MonteCarloSimulator


def main() -> None:
    print("=" * 72)
    print(" QUANT FINANCE LAB — THEORETICAL VALIDATION & CONVERGENCE")
    print("=" * 72)

    initial_capital = 1000.0
    win_prob = 0.60
    win_ret = 0.02
    loss_ret = -0.01
    trades = 100

    model = FixedReturnModel(win_probability=win_prob, win_return=win_ret, loss_return=loss_ret)
    theo_mean = model.analytical_expected_balance(initial_capital, trades)
    theo_median = model.analytical_median_balance(initial_capital, trades)

    print(f"Mathematical Formulation: E[B_N] = B_0 * [ p*(1+R_w) + (1-p)*(1+R_l) ]^N")
    print(f"Exact Theoretical Expected Balance : ${theo_mean:,.2f}")
    print(f"Theoretical Geometric Median       : ${theo_median:,.2f}")
    print("-" * 72)
    print(" Simulations | MC Sample Mean | Abs Error | Rel Error (%) | 95% Confidence Interval")
    print("-------------|----------------|-----------|---------------|------------------------")

    sample_sizes = [500, 2000, 10000, 50000]
    for m in sample_sizes:
        cfg = SimulationConfig(
            initial_capital=initial_capital,
            num_simulations=m,
            num_trades=trades,
            seed=123,
            store_paths=False,
        )
        res = MonteCarloSimulator(config=cfg, model=model).run()
        bench = res.analytical_benchmark
        dm = res.distribution_metrics
        ci = dm.mean_ci

        print(
            f" {m:11,d} | ${dm.mean:13,.2f} | ${bench.absolute_error:+8.2f} | {bench.relative_error_pct:+12.2f}% | [${ci.lower:,.2f}, ${ci.upper:,.2f}]"
        )

    print("-" * 72)
    print("Notice: Agreement between MC mean and theoretical expectation improves with sample size,")
    print("with theoretical expectation consistently falling within the 95% confidence interval.")
    print("=" * 72)


if __name__ == "__main__":
    main()
