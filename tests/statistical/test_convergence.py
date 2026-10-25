"""Statistical convergence and asymptotic accuracy tests."""

import math
import pytest
from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.simulation.monte_carlo import MonteCarloSimulator


def test_monte_carlo_expectation_convergence() -> None:
    """
    Test that Monte Carlo sample mean is within 3 standard errors of the
    exact theoretical expected final balance (99.7% confidence region).
    """
    initial_capital = 1000.0
    win_prob = 0.60
    win_ret = 0.02
    loss_ret = -0.01
    trades = 100
    num_sims = 20000

    cfg = SimulationConfig(
        initial_capital=initial_capital,
        num_simulations=num_sims,
        num_trades=trades,
        seed=2026,
        store_paths=False,
    )
    model = FixedReturnModel(win_probability=win_prob, win_return=win_ret, loss_return=loss_ret)
    res = MonteCarloSimulator(config=cfg, model=model).run()

    bench = res.analytical_benchmark
    assert bench is not None
    # Check z-score is within 3 standard deviations
    assert abs(bench.z_score) < 3.0
    assert bench.is_within_confidence_interval


def test_empirical_bernoulli_trade_win_rate() -> None:
    """Check that generated return elements match expected Bernoulli win probability."""
    model = FixedReturnModel(win_probability=0.60, win_return=0.02, loss_return=-0.01)
    import numpy as np
    rng = np.random.default_rng(999)
    rets = model.generate_returns(5000, 200, rng)
    empirical_p = np.mean(rets == 0.02)
    se = math.sqrt(0.60 * 0.40 / (5000 * 200))
    assert abs(empirical_p - 0.60) < 3.5 * se
