"""Analytical benchmarks and theoretical validation against Monte Carlo estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional
from quant_finance_lab.metrics.confidence import ConfidenceInterval


@dataclass
class AnalyticalBenchmark:
    """
    Comparison between analytical theoretical expectation and empirical Monte Carlo estimate.
    
    Attributes:
        theoretical_expected_balance: Exact mathematical expectation E[B_N].
        theoretical_median_balance: Theoretical / geometric median balance if tractable.
        mc_estimated_mean: Sample mean from Monte Carlo simulation.
        absolute_error: mc_estimated_mean - theoretical_expected_balance.
        relative_error_pct: (mc_estimated_mean / theoretical_expected_balance - 1) * 100.
        standard_error: Standard error of the Monte Carlo mean estimator.
        z_score: Number of standard errors MC estimate deviates from theoretical value.
        is_within_confidence_interval: Whether theoretical value falls inside the MC CI.
    """

    theoretical_expected_balance: float
    theoretical_median_balance: Optional[float]
    mc_estimated_mean: float
    absolute_error: float
    relative_error_pct: float
    standard_error: float
    z_score: float
    is_within_confidence_interval: bool

    def to_dict(self) -> dict:
        return asdict(self)


def compare_analytical_vs_mc(
    theoretical_expected_balance: float,
    theoretical_median_balance: Optional[float],
    mc_mean: float,
    mc_std: float,
    sample_size: int,
    mean_ci: ConfidenceInterval,
) -> AnalyticalBenchmark:
    """Compare theoretical expectations with empirical Monte Carlo simulation outcomes."""
    abs_err = mc_mean - theoretical_expected_balance
    rel_err_pct = (
        (abs_err / theoretical_expected_balance) * 100.0
        if theoretical_expected_balance != 0.0
        else 0.0
    )
    se = mc_std / math.sqrt(sample_size) if sample_size > 0 else 0.0
    z = abs_err / se if se > 1e-12 else 0.0
    within_ci = bool(mean_ci.lower <= theoretical_expected_balance <= mean_ci.upper)

    return AnalyticalBenchmark(
        theoretical_expected_balance=theoretical_expected_balance,
        theoretical_median_balance=theoretical_median_balance,
        mc_estimated_mean=mc_mean,
        absolute_error=abs_err,
        relative_error_pct=rel_err_pct,
        standard_error=se,
        z_score=z,
        is_within_confidence_interval=within_ci,
    )
