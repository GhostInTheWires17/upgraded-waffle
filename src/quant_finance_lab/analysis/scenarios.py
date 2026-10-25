"""Pessimistic, Base-Case, and Optimistic scenario definitions and execution."""

from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd

from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.analysis.strategy_comparison import StrategySpec, compare_strategies
from quant_finance_lab.simulation.monte_carlo import SimulationResult


def run_scenario_analysis(
    initial_capital: float = 1000.0,
    num_simulations: int = 5000,
    num_trades: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, SimulationResult]]:
    """
    Evaluate standardized hypothetical scenarios:
      - Pessimistic: Win Rate 45%, Win 1.8%, Loss -1.2% (Negative / Adverse edge)
      - Base-Case: Win Rate 55%, Win 2.0%, Loss -1.0% (Moderate edge)
      - Optimistic: Win Rate 65%, Win 2.5%, Loss -1.0% (Strong edge)
      
    Note: These are exploratory stress-testing scenarios, not real-world market predictions.
    """
    scenarios = {
        "Pessimistic Scenario": StrategySpec(
            config=SimulationConfig(
                initial_capital=initial_capital,
                num_simulations=num_simulations,
                num_trades=num_trades,
                seed=seed,
            ),
            model=FixedReturnModel(
                win_probability=0.45,
                win_return=0.018,
                loss_return=-0.012,
            ),
        ),
        "Base-Case Scenario": StrategySpec(
            config=SimulationConfig(
                initial_capital=initial_capital,
                num_simulations=num_simulations,
                num_trades=num_trades,
                seed=seed,
            ),
            model=FixedReturnModel(
                win_probability=0.55,
                win_return=0.020,
                loss_return=-0.010,
            ),
        ),
        "Optimistic Scenario": StrategySpec(
            config=SimulationConfig(
                initial_capital=initial_capital,
                num_simulations=num_simulations,
                num_trades=num_trades,
                seed=seed,
            ),
            model=FixedReturnModel(
                win_probability=0.65,
                win_return=0.025,
                loss_return=-0.010,
            ),
        ),
    }

    return compare_strategies(scenarios, shared_seed=seed)
