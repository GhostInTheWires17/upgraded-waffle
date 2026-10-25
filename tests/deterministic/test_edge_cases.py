"""Deterministic edge case tests with exact mathematical expectations."""

import numpy as np
import pytest
from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.core.exceptions import ValidationError
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.simulation.monte_carlo import MonteCarloSimulator


def test_deterministic_100_percent_win() -> None:
    # If win probability is 1.0, every single trade must win with return 0.02
    # Final balance must be exactly 1000 * (1.02)^10
    cfg = SimulationConfig(initial_capital=1000.0, num_simulations=50, num_trades=10, seed=1)
    model = FixedReturnModel(win_probability=1.0, win_return=0.02, loss_return=-0.01)
    res = MonteCarloSimulator(config=cfg, model=model).run()

    expected_final = 1000.0 * (1.02 ** 10)
    assert np.allclose(res.final_balances, expected_final)
    assert res.drawdown_metrics.worst_drawdown_observed == 0.0
    assert res.distribution_metrics.probability_profit.point_estimate == 1.0


def test_deterministic_zero_percent_win() -> None:
    # If win probability is 0.0, every single trade must lose with return -0.01
    # Final balance must be exactly 1000 * (0.99)^10
    cfg = SimulationConfig(initial_capital=1000.0, num_simulations=50, num_trades=10, seed=1)
    model = FixedReturnModel(win_probability=0.0, win_return=0.02, loss_return=-0.01)
    res = MonteCarloSimulator(config=cfg, model=model).run()

    expected_final = 1000.0 * (0.99 ** 10)
    assert np.allclose(res.final_balances, expected_final)
    assert res.distribution_metrics.probability_profit.point_estimate == 0.0
    assert res.distribution_metrics.probability_loss.point_estimate == 1.0


def test_deterministic_zero_trades() -> None:
    # 0 trades: final balance must equal initial capital exactly
    cfg = SimulationConfig(initial_capital=1000.0, num_simulations=20, num_trades=0, seed=1)
    model = FixedReturnModel(win_probability=0.60, win_return=0.02, loss_return=-0.01)
    res = MonteCarloSimulator(config=cfg, model=model).run()

    assert np.all(res.final_balances == 1000.0)
    assert res.drawdown_metrics.worst_drawdown_observed == 0.0


def test_deterministic_total_loss_ruin() -> None:
    # 100% loss (loss_return = -1.0) with win_probability = 0.0 -> balance becomes 0.0
    cfg = SimulationConfig(initial_capital=1000.0, num_simulations=10, num_trades=5, seed=1)
    model = FixedReturnModel(win_probability=0.0, win_return=0.02, loss_return=-1.0)
    res = MonteCarloSimulator(config=cfg, model=model).run()

    assert np.all(res.final_balances == 0.0)


def test_invalid_configurations() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(initial_capital=0.0)  # Initial capital must be > 0

    with pytest.raises(ValidationError):
        SimulationConfig(num_simulations=0)  # Must have >= 1 simulations

    with pytest.raises(ValidationError):
        SimulationConfig(num_trades=-5)
