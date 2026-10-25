"""Regression tests validating continuity with legacy v0.0.1 and v0.0.2 model behavior."""

import pytest
from quant_finance_lab.simulation.monte_carlo import run_simulation


def test_legacy_v001_equivalence() -> None:
    """
    Direct regression test comparing new simulation against the baseline
    parameters of original v0.0.1 script:
      - Initial capital: $1,000
      - Win probability: 0.60
      - Win return: +2% (0.02)
      - Loss return: -1% (-0.01)
      - Number of trades: 100
      - Number of simulations: 10,000
    """
    res = run_simulation(
        initial_capital=1000.0,
        win_probability=0.60,
        win_return=0.02,
        loss_return=-0.01,
        num_simulations=10000,
        num_trades=100,
        seed=42,
        targets=[1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
    )

    # 1. Theoretical benchmark check:
    # E[B_N] = 1000 * (0.60 * 1.02 + 0.40 * 0.99)^100 = 1000 * (1.008)^100 = 2219.6402...
    assert res.analytical_benchmark is not None
    theo_mean = res.analytical_benchmark.theoretical_expected_balance
    assert theo_mean == pytest.approx(1000.0 * (1.008 ** 100))

    # 2. Monte Carlo estimated mean should be close to theoretical expectation
    assert abs(res.distribution_metrics.mean - theo_mean) < 3.0 * res.analytical_benchmark.standard_error

    # 3. Confirm author's original right-skewness observation:
    # In compounding processes, the mean is strictly greater than the median!
    assert res.distribution_metrics.mean > res.distribution_metrics.median
    assert res.distribution_metrics.skewness > 0.0

    # 4. Target success rate monotonicity:
    # Higher balance targets must have non-increasing success rates
    target_probs = [t.terminal_probability.point_estimate for t in res.distribution_metrics.target_probabilities]
    for i in range(len(target_probs) - 1):
        assert target_probs[i] >= target_probs[i + 1]
