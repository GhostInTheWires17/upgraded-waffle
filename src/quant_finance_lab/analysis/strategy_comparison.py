"""Multi-strategy comparative evaluation under consistent random streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd

from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.base import ReturnModel
from quant_finance_lab.models.costs import TransactionCostModel
from quant_finance_lab.models.position_sizing import PositionSizer
from quant_finance_lab.simulation.monte_carlo import MonteCarloSimulator, SimulationResult


@dataclass
class StrategySpec:
    """Strategy configuration specification for comparison."""

    config: SimulationConfig
    model: ReturnModel
    cost_model: Optional[TransactionCostModel] = None
    position_sizer: Optional[PositionSizer] = None


def compare_strategies(
    strategies: Dict[str, StrategySpec],
    shared_seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, Dict[str, SimulationResult]]:
    """
    Run and compare multiple strategies under consistent seeds and horizons.
    
    Returns:
        (summary_dataframe, dict_of_simulation_results)
    """
    results: Dict[str, SimulationResult] = {}
    rows = []

    for name, spec in strategies.items():
        # Inject shared seed if requested to minimize random sampling variance
        cfg = SimulationConfig(
            initial_capital=spec.config.initial_capital,
            num_simulations=spec.config.num_simulations,
            num_trades=spec.config.num_trades,
            seed=shared_seed if shared_seed is not None else spec.config.seed,
            targets=spec.config.targets,
            ruin_threshold=spec.config.ruin_threshold,
            confidence_level=spec.config.confidence_level,
            chunk_size=spec.config.chunk_size,
            store_paths=spec.config.store_paths,
        )

        sim = MonteCarloSimulator(
            config=cfg,
            model=spec.model,
            cost_model=spec.cost_model,
            position_sizer=spec.position_sizer,
        )
        res = sim.run()
        results[name] = res

        bench_str = (
            f"${res.analytical_benchmark.theoretical_expected_balance:,.2f}"
            if res.analytical_benchmark
            else "N/A"
        )
        ruin_p = (
            f"{res.risk_metrics.terminal_ruin_probability.point_estimate:.1%}"
            if res.risk_metrics.terminal_ruin_probability
            else "N/A"
        )

        rows.append(
            {
                "Strategy": name,
                "Model": res.model.name,
                "Initial Capital": f"${cfg.initial_capital:,.2f}",
                "Theoretical Expectation": bench_str,
                "MC Mean Balance": f"${res.distribution_metrics.mean:,.2f}",
                "MC Median Balance": f"${res.distribution_metrics.median:,.2f}",
                "Profit Probability": f"{res.distribution_metrics.probability_profit.point_estimate:.1%}",
                "Ruin Probability": ruin_p,
                "Median Max Drawdown": f"{res.drawdown_metrics.median_max_drawdown:.1%}",
                "95th Pct Drawdown": f"{res.drawdown_metrics.percentile_95_max_drawdown:.1%}",
                "95% VaR ($)": f"${res.risk_metrics.var_95_dollar:,.2f}",
                "95% CVaR ($)": f"${res.risk_metrics.cvar_95_dollar:,.2f}",
                "Geometric Growth Rate": f"{res.return_metrics.geometric_growth_rate:.2%}",
            }
        )

    df_summary = pd.DataFrame(rows)
    return df_summary, results
