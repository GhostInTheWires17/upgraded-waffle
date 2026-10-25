"""Vectorized simulation engine with chunked memory management."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.base import ReturnModel
from quant_finance_lab.models.costs import TransactionCostModel
from quant_finance_lab.models.position_sizing import PositionSizer
from quant_finance_lab.metrics.drawdown import compute_max_drawdowns


def simulate_chunk(
    initial_capital: float,
    chunk_sims: int,
    num_trades: int,
    model: ReturnModel,
    rng: np.random.Generator,
    cost_model: Optional[TransactionCostModel] = None,
    position_sizer: Optional[PositionSizer] = None,
    store_paths: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Simulate a batch/chunk of trading paths.
    
    Returns:
        (final_balances, balance_paths_or_none, max_drawdown_magnitudes, min_balances)
    """
    if num_trades == 0:
        finals = np.full(chunk_sims, initial_capital, dtype=np.float64)
        mins = finals.copy()
        mdds = np.zeros(chunk_sims, dtype=np.float64)
        paths = np.full((chunk_sims, 1), initial_capital, dtype=np.float64) if store_paths else None
        return finals, paths, mdds, mins

    returns = model.generate_returns(chunk_sims, num_trades, rng)

    has_costs = cost_model is not None and cost_model.has_costs
    has_custom_sizing = position_sizer is not None and position_sizer.sizing_type != "full"

    if not has_costs and not has_custom_sizing:
        # High-performance pure multiplicative vectorization
        multipliers = 1.0 + returns
        # Handle absorption at 0 if total ruin occurs (-1.0 return)
        multipliers = np.maximum(multipliers, 0.0)

        if store_paths:
            paths = np.empty((chunk_sims, num_trades + 1), dtype=np.float64)
            paths[:, 0] = initial_capital
            paths[:, 1:] = initial_capital * np.cumprod(multipliers, axis=1)
            finals = paths[:, -1]
            mins = np.min(paths, axis=1)
            mdds = compute_max_drawdowns(paths)
            return finals, paths, mdds, mins
        else:
            # Memory-light execution: accumulate running statistics without holding full path matrix
            running_balances = np.full(chunk_sims, initial_capital, dtype=np.float64)
            running_peaks = np.full(chunk_sims, initial_capital, dtype=np.float64)
            running_mins = np.full(chunk_sims, initial_capital, dtype=np.float64)
            max_drawdowns = np.zeros(chunk_sims, dtype=np.float64)

            for t in range(num_trades):
                running_balances *= multipliers[:, t]
                running_peaks = np.maximum(running_peaks, running_balances)
                running_mins = np.minimum(running_mins, running_balances)
                current_dd = np.where(
                    running_peaks > 0.0,
                    1.0 - (running_balances / running_peaks),
                    0.0,
                )
                max_drawdowns = np.maximum(max_drawdowns, current_dd)

            return running_balances, None, max_drawdowns, running_mins

    # Simulation with transaction friction and/or position sizing
    if store_paths:
        paths = np.empty((chunk_sims, num_trades + 1), dtype=np.float64)
        paths[:, 0] = initial_capital
        current = np.full(chunk_sims, initial_capital, dtype=np.float64)

        for t in range(num_trades):
            r_t = returns[:, t]
            if has_custom_sizing and position_sizer is not None:
                if position_sizer.sizing_type == "fixed_fractional":
                    allocated = current * position_sizer.fraction
                elif position_sizer.sizing_type == "fixed_dollar":
                    allocated = np.minimum(current, position_sizer.dollar_amount)
                else:
                    allocated = current
            else:
                allocated = current

            gross_pnl = allocated * r_t

            if has_costs and cost_model is not None:
                friction = cost_model.fixed_cost + (allocated * cost_model.proportional_cost)
                current = np.maximum(0.0, current + gross_pnl - friction)
            else:
                current = np.maximum(0.0, current + gross_pnl)

            paths[:, t + 1] = current

        finals = paths[:, -1]
        mins = np.min(paths, axis=1)
        mdds = compute_max_drawdowns(paths)
        return finals, paths, mdds, mins
    else:
        current = np.full(chunk_sims, initial_capital, dtype=np.float64)
        running_peaks = np.full(chunk_sims, initial_capital, dtype=np.float64)
        running_mins = np.full(chunk_sims, initial_capital, dtype=np.float64)
        max_drawdowns = np.zeros(chunk_sims, dtype=np.float64)

        for t in range(num_trades):
            r_t = returns[:, t]
            if has_custom_sizing and position_sizer is not None:
                if position_sizer.sizing_type == "fixed_fractional":
                    allocated = current * position_sizer.fraction
                elif position_sizer.sizing_type == "fixed_dollar":
                    allocated = np.minimum(current, position_sizer.dollar_amount)
                else:
                    allocated = current
            else:
                allocated = current

            gross_pnl = allocated * r_t

            if has_costs and cost_model is not None:
                friction = cost_model.fixed_cost + (allocated * cost_model.proportional_cost)
                current = np.maximum(0.0, current + gross_pnl - friction)
            else:
                current = np.maximum(0.0, current + gross_pnl)

            running_peaks = np.maximum(running_peaks, current)
            running_mins = np.minimum(running_mins, current)
            current_dd = np.where(running_peaks > 0.0, 1.0 - (current / running_peaks), 0.0)
            max_drawdowns = np.maximum(max_drawdowns, current_dd)

        return current, None, max_drawdowns, running_mins
