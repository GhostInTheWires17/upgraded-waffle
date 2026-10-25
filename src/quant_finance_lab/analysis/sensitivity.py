"""1D and 2D sensitivity analysis with Common Random Numbers (CRN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Callable, Any
import numpy as np
import pandas as pd

from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.simulation.monte_carlo import MonteCarloSimulator


@dataclass
class SensitivityResult1D:
    """Results from a 1-dimensional parameter sweep."""

    parameter_name: str
    values: list[float]
    mean_balances: list[float]
    median_balances: list[float]
    profit_probabilities: list[float]
    ruin_probabilities: list[float]
    median_max_drawdowns: list[float]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.parameter_name: self.values,
                "mean_final_balance": self.mean_balances,
                "median_final_balance": self.median_balances,
                "probability_of_profit": self.profit_probabilities,
                "probability_of_ruin": self.ruin_probabilities,
                "median_max_drawdown": self.median_max_drawdowns,
            }
        )


@dataclass
class SensitivityResult2D:
    """Results from a 2-dimensional grid sweep."""

    param_x_name: str
    param_x_values: list[float]
    param_y_name: str
    param_y_values: list[float]
    grid_mean_balance: np.ndarray
    grid_median_balance: np.ndarray
    grid_profit_prob: np.ndarray
    grid_ruin_prob: np.ndarray
    grid_median_mdd: np.ndarray

    def to_dataframe(self, metric: str = "mean_balance") -> pd.DataFrame:
        """Return 2D pivot table DataFrame for heatmaps."""
        data_map = {
            "mean_balance": self.grid_mean_balance,
            "median_balance": self.grid_median_balance,
            "profit_prob": self.grid_profit_prob,
            "ruin_prob": self.grid_ruin_prob,
            "median_mdd": self.grid_median_mdd,
        }
        grid = data_map.get(metric, self.grid_mean_balance)
        df = pd.DataFrame(
            grid,
            index=[f"{v:.4g}" for v in self.param_y_values],
            columns=[f"{v:.4g}" for v in self.param_x_values],
        )
        df.index.name = self.param_y_name
        df.columns.name = self.param_x_name
        return df


def sweep_parameter_1d(
    parameter_name: str,
    parameter_values: Sequence[float],
    base_config: Optional[SimulationConfig] = None,
    base_model_kwargs: Optional[dict[str, Any]] = None,
    use_crn: bool = True,
) -> SensitivityResult1D:
    """
    Execute 1-dimensional sensitivity parameter sweep.
    
    Args:
        parameter_name: Parameter to sweep ('win_probability', 'win_return', 'loss_return', 'num_trades').
        parameter_values: Values to evaluate.
        base_config: Baseline simulation configuration.
        base_model_kwargs: Baseline parameters for FixedReturnModel.
        use_crn: If True, uses Common Random Numbers (identical seed for each sweep step)
                 to isolate parameter sensitivity from Monte Carlo noise.
    """
    cfg = base_config or SimulationConfig(num_simulations=5000, num_trades=100, seed=42)
    m_kwargs = base_model_kwargs or {"win_probability": 0.60, "win_return": 0.02, "loss_return": -0.01}

    means, medians, profits, ruins, mdds = [], [], [], [], []

    for val in parameter_values:
        step_kwargs = m_kwargs.copy()
        step_cfg = SimulationConfig(
            initial_capital=cfg.initial_capital,
            num_simulations=cfg.num_simulations,
            num_trades=int(val) if parameter_name == "num_trades" else cfg.num_trades,
            seed=cfg.seed if use_crn else None,
            ruin_threshold=cfg.ruin_threshold,
            store_paths=False,
        )

        if parameter_name in step_kwargs:
            step_kwargs[parameter_name] = val

        model = FixedReturnModel(**step_kwargs)
        sim = MonteCarloSimulator(config=step_cfg, model=model)
        res = sim.run()

        means.append(res.distribution_metrics.mean)
        medians.append(res.distribution_metrics.median)
        profits.append(res.distribution_metrics.probability_profit.point_estimate)
        ruin_p = (
            res.risk_metrics.terminal_ruin_probability.point_estimate
            if res.risk_metrics.terminal_ruin_probability
            else 0.0
        )
        ruins.append(ruin_p)
        mdds.append(res.drawdown_metrics.median_max_drawdown)

    return SensitivityResult1D(
        parameter_name=parameter_name,
        values=list(parameter_values),
        mean_balances=means,
        median_balances=medians,
        profit_probabilities=profits,
        ruin_probabilities=ruins,
        median_max_drawdowns=mdds,
    )


def sweep_grid_2d(
    param_x_name: str,
    param_x_values: Sequence[float],
    param_y_name: str,
    param_y_values: Sequence[float],
    base_config: Optional[SimulationConfig] = None,
    base_model_kwargs: Optional[dict[str, Any]] = None,
    use_crn: bool = True,
) -> SensitivityResult2D:
    """Execute 2-dimensional grid sweep."""
    cfg = base_config or SimulationConfig(num_simulations=2000, num_trades=100, seed=42)
    m_kwargs = base_model_kwargs or {"win_probability": 0.60, "win_return": 0.02, "loss_return": -0.01}

    nx = len(param_x_values)
    ny = len(param_y_values)

    grid_mean = np.zeros((ny, nx), dtype=np.float64)
    grid_med = np.zeros((ny, nx), dtype=np.float64)
    grid_prof = np.zeros((ny, nx), dtype=np.float64)
    grid_ruin = np.zeros((ny, nx), dtype=np.float64)
    grid_mdd = np.zeros((ny, nx), dtype=np.float64)

    for i, y_val in enumerate(param_y_values):
        for j, x_val in enumerate(param_x_values):
            step_kwargs = m_kwargs.copy()
            step_cfg = SimulationConfig(
                initial_capital=cfg.initial_capital,
                num_simulations=cfg.num_simulations,
                num_trades=cfg.num_trades,
                seed=cfg.seed if use_crn else None,
                ruin_threshold=cfg.ruin_threshold,
                store_paths=False,
            )

            if param_x_name in step_kwargs:
                step_kwargs[param_x_name] = x_val
            elif param_x_name == "num_trades":
                step_cfg.num_trades = int(x_val)

            if param_y_name in step_kwargs:
                step_kwargs[param_y_name] = y_val
            elif param_y_name == "num_trades":
                step_cfg.num_trades = int(y_val)

            model = FixedReturnModel(**step_kwargs)
            res = MonteCarloSimulator(config=step_cfg, model=model).run()

            grid_mean[i, j] = res.distribution_metrics.mean
            grid_med[i, j] = res.distribution_metrics.median
            grid_prof[i, j] = res.distribution_metrics.probability_profit.point_estimate
            grid_ruin[i, j] = (
                res.risk_metrics.terminal_ruin_probability.point_estimate
                if res.risk_metrics.terminal_ruin_probability
                else 0.0
            )
            grid_mdd[i, j] = res.drawdown_metrics.median_max_drawdown

    return SensitivityResult2D(
        param_x_name=param_x_name,
        param_x_values=list(param_x_values),
        param_y_name=param_y_name,
        param_y_values=list(param_y_values),
        grid_mean_balance=grid_mean,
        grid_median_balance=grid_med,
        grid_profit_prob=grid_prof,
        grid_ruin_prob=grid_ruin,
        grid_median_mdd=grid_mdd,
    )
