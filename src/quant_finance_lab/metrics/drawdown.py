"""Drawdown metrics and maximum drawdown distribution analysis."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence
import numpy as np

from quant_finance_lab.metrics.confidence import wilson_score_interval, ConfidenceInterval


@dataclass
class DrawdownMetrics:
    """Comprehensive summary of path-level drawdown statistics."""

    mean_max_drawdown: float
    median_max_drawdown: float
    std_max_drawdown: float
    percentile_90_max_drawdown: float
    percentile_95_max_drawdown: float
    percentile_99_max_drawdown: float
    worst_drawdown_observed: float
    threshold_exceedance: dict[float, ConfidenceInterval]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["threshold_exceedance"] = {
            f"{k:.0%}": v.to_dict() for k, v in self.threshold_exceedance.items()
        }
        return d


def compute_drawdown_series(balance_paths: np.ndarray) -> np.ndarray:
    """
    Compute peak-to-trough relative drawdown time series for balance paths.
    
    Args:
        balance_paths: 2D array of shape (num_simulations, num_steps)
        
    Returns:
        drawdowns: 2D array of same shape where DD[i, t] = B[i, t] / Peak[i, t] - 1.0 <= 0.0
    """
    if balance_paths.ndim != 2:
        raise ValueError(f"balance_paths must be 2D, got shape {balance_paths.shape}")
    running_peaks = np.maximum.accumulate(balance_paths, axis=1)
    # Safe division: if running_peak is zero, drawdown is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_peaks > 0.0, (balance_paths / running_peaks) - 1.0, 0.0)
    return drawdowns


def compute_max_drawdowns(balance_paths: np.ndarray) -> np.ndarray:
    """
    Compute Maximum Drawdown (MDD) magnitude for each simulation path.
    
    Returns 1D array of positive magnitudes in [0.0, 1.0].
    E.g., 0.25 indicates a 25% peak-to-trough drop.
    """
    drawdowns = compute_drawdown_series(balance_paths)
    min_dd = np.min(drawdowns, axis=1)  # Most negative drawdown
    return -min_dd  # Return positive magnitude


def analyze_drawdowns(
    balance_paths: np.ndarray,
    thresholds: Sequence[float] = (0.10, 0.20, 0.30, 0.50),
    confidence_level: float = 0.95,
) -> DrawdownMetrics:
    """Compute complete drawdown statistics across all simulated paths."""
    mdd_magnitudes = compute_max_drawdowns(balance_paths)
    num_sims = len(mdd_magnitudes)

    exceedances: dict[float, ConfidenceInterval] = {}
    for thresh in thresholds:
        count = int(np.sum(mdd_magnitudes >= thresh))
        ci = wilson_score_interval(count, num_sims, confidence_level=confidence_level)
        exceedances[thresh] = ci

    return DrawdownMetrics(
        mean_max_drawdown=float(np.mean(mdd_magnitudes)),
        median_max_drawdown=float(np.median(mdd_magnitudes)),
        std_max_drawdown=float(np.std(mdd_magnitudes, ddof=1)) if num_sims > 1 else 0.0,
        percentile_90_max_drawdown=float(np.percentile(mdd_magnitudes, 90)),
        percentile_95_max_drawdown=float(np.percentile(mdd_magnitudes, 95)),
        percentile_99_max_drawdown=float(np.percentile(mdd_magnitudes, 99)),
        worst_drawdown_observed=float(np.max(mdd_magnitudes)),
        threshold_exceedance=exceedances,
    )
