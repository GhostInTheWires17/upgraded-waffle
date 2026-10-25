"""Monte Carlo simulation orchestrator and result container."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
import pandas as pd

from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.core.random_state import RandomStateManager
from quant_finance_lab.models.base import ReturnModel
from quant_finance_lab.models.costs import TransactionCostModel
from quant_finance_lab.models.position_sizing import PositionSizer
from quant_finance_lab.simulation.engine import simulate_chunk
from quant_finance_lab.simulation.analytical_benchmarks import (
    compare_analytical_vs_mc,
    AnalyticalBenchmark,
)
from quant_finance_lab.metrics.drawdown import (
    DrawdownMetrics,
    analyze_drawdowns,
)
from quant_finance_lab.metrics.risk import (
    RiskMetrics,
    analyze_risk,
)
from quant_finance_lab.metrics.returns import (
    ReturnMetrics,
    compute_return_metrics,
)
from quant_finance_lab.metrics.distributions import (
    DistributionMetrics,
    analyze_distribution,
)


@dataclass
class SimulationResult:
    """Complete simulation result and statistical metrics container."""

    config: SimulationConfig
    model: ReturnModel
    final_balances: np.ndarray
    max_drawdowns: np.ndarray
    min_balances: np.ndarray
    balance_paths: Optional[np.ndarray]
    drawdown_metrics: DrawdownMetrics
    risk_metrics: RiskMetrics
    return_metrics: ReturnMetrics
    distribution_metrics: DistributionMetrics
    analytical_benchmark: Optional[AnalyticalBenchmark]
    execution_time_seconds: float
    cost_model: Optional[TransactionCostModel] = None
    position_sizer: Optional[PositionSizer] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert path summary metrics to a pandas DataFrame."""
        df = pd.DataFrame(
            {
                "final_balance": self.final_balances,
                "max_drawdown": self.max_drawdowns,
                "min_balance": self.min_balances,
                "total_return_pct": (self.final_balances / self.config.initial_capital) - 1.0,
            }
        )
        return df

    def to_dict(self) -> dict[str, Any]:
        """Convert entire result and metrics hierarchy to JSON-serializable dictionary."""
        return {
            "metadata": self.metadata,
            "config": self.config.to_dict(),
            "model": self.model.to_dict(),
            "cost_model": self.cost_model.to_dict() if self.cost_model else None,
            "position_sizer": self.position_sizer.to_dict() if self.position_sizer else None,
            "execution_time_seconds": self.execution_time_seconds,
            "distribution_metrics": self.distribution_metrics.to_dict(),
            "return_metrics": self.return_metrics.to_dict(),
            "drawdown_metrics": self.drawdown_metrics.to_dict(),
            "risk_metrics": self.risk_metrics.to_dict(),
            "analytical_benchmark": (
                self.analytical_benchmark.to_dict() if self.analytical_benchmark else None
            ),
        }


class MonteCarloSimulator:
    """High-level simulation coordinator."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        model: Optional[ReturnModel] = None,
        cost_model: Optional[TransactionCostModel] = None,
        position_sizer: Optional[PositionSizer] = None,
    ) -> None:
        self.config = config or SimulationConfig()
        if model is None:
            from quant_finance_lab.models.fixed_return import FixedReturnModel
            self.model = FixedReturnModel()
        else:
            self.model = model
        self.cost_model = cost_model
        self.position_sizer = position_sizer

    def run(self) -> SimulationResult:
        """Execute the Monte Carlo simulation across chunks and compute all metrics."""
        t_start = time.perf_counter()

        rng_mgr = RandomStateManager(self.config.seed)
        total_sims = self.config.num_simulations
        chunk_size = self.config.chunk_size
        n_trades = self.config.num_trades
        b0 = self.config.initial_capital

        all_finals = []
        all_mdds = []
        all_mins = []
        all_paths = [] if self.config.store_paths else None

        remaining = total_sims
        while remaining > 0:
            current_chunk = min(remaining, chunk_size)
            finals, paths, mdds, mins = simulate_chunk(
                initial_capital=b0,
                chunk_sims=current_chunk,
                num_trades=n_trades,
                model=self.model,
                rng=rng_mgr.rng,
                cost_model=self.cost_model,
                position_sizer=self.position_sizer,
                store_paths=self.config.store_paths,
            )
            all_finals.append(finals)
            all_mdds.append(mdds)
            all_mins.append(mins)
            if self.config.store_paths and paths is not None:
                all_paths.append(paths)

            remaining -= current_chunk

        final_balances = np.concatenate(all_finals)
        max_drawdowns = np.concatenate(all_mdds)
        min_balances = np.concatenate(all_mins)
        balance_paths = np.vstack(all_paths) if self.config.store_paths and all_paths else None

        # Calculate metrics
        dist_metrics = analyze_distribution(
            initial_capital=b0,
            final_balances=final_balances,
            targets=self.config.targets,
            balance_paths=balance_paths,
            confidence_level=self.config.confidence_level,
        )

        ret_metrics = compute_return_metrics(
            initial_capital=b0,
            final_balances=final_balances,
            num_trades=n_trades,
        )

        drawdown_metrics = (
            analyze_drawdowns(
                balance_paths, confidence_level=self.config.confidence_level
            )
            if balance_paths is not None
            else DrawdownMetrics(
                mean_max_drawdown=float(np.mean(max_drawdowns)),
                median_max_drawdown=float(np.median(max_drawdowns)),
                std_max_drawdown=float(np.std(max_drawdowns, ddof=1)) if len(max_drawdowns) > 1 else 0.0,
                percentile_90_max_drawdown=float(np.percentile(max_drawdowns, 90)),
                percentile_95_max_drawdown=float(np.percentile(max_drawdowns, 95)),
                percentile_99_max_drawdown=float(np.percentile(max_drawdowns, 99)),
                worst_drawdown_observed=float(np.max(max_drawdowns)),
                threshold_exceedance={},
            )
        )

        risk_metrics = analyze_risk(
            initial_capital=b0,
            final_balances=final_balances,
            balance_paths=balance_paths,
            ruin_threshold=self.config.ruin_threshold,
            confidence_level=self.config.confidence_level,
        )

        # Analytical benchmark comparison if supported by model
        benchmark: Optional[AnalyticalBenchmark] = None
        has_costs = self.cost_model is not None and self.cost_model.has_costs
        has_custom_sizing = self.position_sizer is not None and self.position_sizer.sizing_type != "full"

        if not has_costs and not has_custom_sizing:
            theo_mean = self.model.analytical_expected_balance(b0, n_trades)
            theo_med = self.model.analytical_median_balance(b0, n_trades)
            if theo_mean is not None:
                benchmark = compare_analytical_vs_mc(
                    theoretical_expected_balance=theo_mean,
                    theoretical_median_balance=theo_med,
                    mc_mean=dist_metrics.mean,
                    mc_std=dist_metrics.std_dev,
                    sample_size=total_sims,
                    mean_ci=dist_metrics.mean_ci,
                )

        t_elapsed = time.perf_counter() - t_start

        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rng": rng_mgr.to_metadata(),
            "num_simulations": total_sims,
            "num_trades": n_trades,
            "stored_paths": self.config.store_paths,
        }

        return SimulationResult(
            config=self.config,
            model=self.model,
            final_balances=final_balances,
            max_drawdowns=max_drawdowns,
            min_balances=min_balances,
            balance_paths=balance_paths,
            drawdown_metrics=drawdown_metrics,
            risk_metrics=risk_metrics,
            return_metrics=ret_metrics,
            distribution_metrics=dist_metrics,
            analytical_benchmark=benchmark,
            execution_time_seconds=t_elapsed,
            cost_model=self.cost_model,
            position_sizer=self.position_sizer,
            metadata=metadata,
        )


def run_simulation(
    initial_capital: float = 1000.0,
    win_probability: float = 0.60,
    win_return: float = 0.02,
    loss_return: float = -0.01,
    num_simulations: int = 10000,
    num_trades: int = 100,
    seed: Optional[int] = None,
    targets: Optional[list[float]] = None,
    ruin_threshold: Optional[float] = 500.0,
    store_paths: bool = True,
) -> SimulationResult:
    """Convenience helper to run a basic simulation with default FixedReturnModel."""
    from quant_finance_lab.models.fixed_return import FixedReturnModel

    cfg = SimulationConfig(
        initial_capital=initial_capital,
        num_simulations=num_simulations,
        num_trades=num_trades,
        seed=seed,
        targets=targets if targets is not None else [1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
        ruin_threshold=ruin_threshold,
        store_paths=store_paths,
    )
    model = FixedReturnModel(
        win_probability=win_probability,
        win_return=win_return,
        loss_return=loss_return,
    )
    simulator = MonteCarloSimulator(config=cfg, model=model)
    return simulator.run()
