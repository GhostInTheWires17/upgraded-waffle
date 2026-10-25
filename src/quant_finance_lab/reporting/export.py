"""Export utilities for simulation results, metadata, and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict
from quant_finance_lab.simulation.monte_carlo import SimulationResult
from quant_finance_lab.reporting.metadata import generate_environment_metadata
from quant_finance_lab.visualization.plots import (
    plot_final_balance_distribution,
    plot_equity_curves,
    plot_drawdown_distribution,
    plot_target_probabilities,
    plot_analytical_vs_mc,
)


def export_results_to_json(result: SimulationResult, file_path: str | Path) -> Path:
    """Save full simulation results, metrics, and reproducibility metadata to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = result.to_dict()
    data["environment"] = generate_environment_metadata()

    def _numpy_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_numpy_serializer)

    return path


def export_results_to_csv(result: SimulationResult, file_path: str | Path) -> Path:
    """Save path-level simulation statistics to a CSV file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = result.to_dataframe()
    df.to_csv(path, index=False)
    return path


def export_all_plots(result: SimulationResult, output_dir: str | Path) -> Dict[str, Path]:
    """Generate and save all applicable research plots to the target directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved_plots: Dict[str, Path] = {}

    p_hist = out / "final_balance_distribution.png"
    plot_final_balance_distribution(result, save_path=p_hist)
    saved_plots["distribution"] = p_hist

    if result.balance_paths is not None:
        p_equity = out / "equity_curves.png"
        plot_equity_curves(result, save_path=p_equity)
        saved_plots["equity_curves"] = p_equity

    p_mdd = out / "drawdown_distribution.png"
    plot_drawdown_distribution(result, save_path=p_mdd)
    saved_plots["drawdowns"] = p_mdd

    if result.distribution_metrics.target_probabilities:
        p_targets = out / "target_probabilities.png"
        plot_target_probabilities(result, save_path=p_targets)
        saved_plots["targets"] = p_targets

    if result.analytical_benchmark is not None:
        p_bench = out / "analytical_vs_monte_carlo.png"
        plot_analytical_vs_mc(result, save_path=p_bench)
        saved_plots["analytical_benchmark"] = p_bench

    return saved_plots
