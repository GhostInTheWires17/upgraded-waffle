"""Distributional metrics, skewness, kurtosis, and target reach analysis."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence, Optional
import numpy as np
from scipy import stats

from quant_finance_lab.metrics.confidence import (
    wilson_score_interval,
    mean_confidence_interval,
    ConfidenceInterval,
)


@dataclass
class TargetMetrics:
    """Probabilities of reaching financial targets."""

    target: float
    terminal_probability: ConfidenceInterval
    path_probability: Optional[ConfidenceInterval]

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "terminal_probability": self.terminal_probability.to_dict(),
            "path_probability": (
                self.path_probability.to_dict() if self.path_probability is not None else None
            ),
        }


@dataclass
class DistributionMetrics:
    """Statistical distribution properties of simulated final balances."""

    mean: float
    mean_ci: ConfidenceInterval
    median: float
    std_dev: float
    variance: float
    skewness: float
    excess_kurtosis: float
    percentiles: dict[str, float]
    probability_profit: ConfidenceInterval
    probability_loss: ConfidenceInterval
    target_probabilities: list[TargetMetrics]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean_ci"] = self.mean_ci.to_dict()
        d["probability_profit"] = self.probability_profit.to_dict()
        d["probability_loss"] = self.probability_loss.to_dict()
        d["target_probabilities"] = [t.to_dict() for t in self.target_probabilities]
        return d


def analyze_distribution(
    initial_capital: float,
    final_balances: np.ndarray,
    targets: Sequence[float] = (),
    balance_paths: Optional[np.ndarray] = None,
    confidence_level: float = 0.95,
) -> DistributionMetrics:
    """Compute complete statistical distribution parameters and target probabilities."""
    n = len(final_balances)
    mean_val = float(np.mean(final_balances))
    std_val = float(np.std(final_balances, ddof=1)) if n > 1 else 0.0
    var_val = float(np.var(final_balances, ddof=1)) if n > 1 else 0.0
    median_val = float(np.median(final_balances))

    if std_val < 1e-12:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = float(stats.skew(final_balances, bias=False)) if n > 2 else 0.0
        kurt_val = float(stats.kurtosis(final_balances, bias=False)) if n > 3 else 0.0

    mean_ci = mean_confidence_interval(
        mean_val, std_val, n, confidence_level=confidence_level
    ) if n > 1 else ConfidenceInterval(mean_val, mean_val, mean_val, confidence_level, 0.0, "deterministic")

    pct_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(final_balances, pct_keys)
    percentiles_dict = {f"p{k}": float(v) for k, v in zip(pct_keys, pct_values)}

    # Profit / Loss events
    profit_count = int(np.sum(final_balances > initial_capital))
    loss_count = int(np.sum(final_balances < initial_capital))
    prob_profit = wilson_score_interval(profit_count, n, confidence_level=confidence_level)
    prob_loss = wilson_score_interval(loss_count, n, confidence_level=confidence_level)

    # Target threshold analysis
    target_metrics_list: list[TargetMetrics] = []
    max_paths = np.max(balance_paths, axis=1) if balance_paths is not None else None

    for target in targets:
        term_count = int(np.sum(final_balances >= target))
        term_ci = wilson_score_interval(term_count, n, confidence_level=confidence_level)

        path_ci: Optional[ConfidenceInterval] = None
        if max_paths is not None:
            path_count = int(np.sum(max_paths >= target))
            path_ci = wilson_score_interval(path_count, n, confidence_level=confidence_level)

        target_metrics_list.append(
            TargetMetrics(
                target=target,
                terminal_probability=term_ci,
                path_probability=path_ci,
            )
        )

    return DistributionMetrics(
        mean=mean_val,
        mean_ci=mean_ci,
        median=median_val,
        std_dev=std_val,
        variance=var_val,
        skewness=skew_val,
        excess_kurtosis=kurt_val,
        percentiles=percentiles_dict,
        probability_profit=prob_profit,
        probability_loss=prob_loss,
        target_probabilities=target_metrics_list,
    )
