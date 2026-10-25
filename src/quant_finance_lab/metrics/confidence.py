"""Statistical reliability and confidence interval calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from scipy import stats
from quant_finance_lab.core.validation import validate_confidence_level, validate_integer


@dataclass(frozen=True)
class ConfidenceInterval:
    """
    Container for statistical point estimate and confidence interval.
    
    Attributes:
        point_estimate: Sample point estimate (e.g. sample mean, empirical proportion).
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
        confidence_level: Confidence level (e.g. 0.95 for 95% confidence).
        standard_error: Standard error of the estimator.
        method: Statistical method utilized (e.g. 'wilson_score', 'student_t').
    """

    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    standard_error: float
    method: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "point_estimate": self.point_estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "standard_error": self.standard_error,
            "method": self.method,
        }


def wilson_score_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Calculate the Wilson score confidence interval for a binomial proportion.
    
    The Wilson score interval provides superior coverage probability compared
    to the normal approximation (Wald interval), particularly for extreme probabilities
    near 0.0 or 1.0, or small sample sizes.
    
    Mathematical definition:
      p_hat = successes / trials
      z = Phi^{-1}(1 - (1 - confidence_level)/2)
      center = (p_hat + z^2 / (2n)) / (1 + z^2 / n)
      half_width = (z / (1 + z^2 / n)) * sqrt(p_hat*(1 - p_hat)/n + z^2 / (4n^2))
      CI = [center - half_width, center + half_width]
    """
    n = validate_integer(trials, "trials", min_val=1)
    k = validate_integer(successes, "successes", min_val=0)
    if k > n:
        raise ValueError(f"successes ({k}) cannot exceed trials ({n})")
    
    cl = validate_confidence_level(confidence_level)
    p_hat = k / n
    alpha = 1.0 - cl
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    denominator = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denominator
    spread = (z / denominator) * math.sqrt(
        (p_hat * (1.0 - p_hat) / n) + ((z ** 2) / (4.0 * (n ** 2)))
    )

    if k == 0:
        lower = 0.0
    else:
        lower = max(0.0, float(center - spread))

    if k == n:
        upper = 1.0
    else:
        upper = min(1.0, float(center + spread))

    se = math.sqrt(p_hat * (1.0 - p_hat) / n) if n > 0 else 0.0

    return ConfidenceInterval(
        point_estimate=p_hat,
        lower=lower,
        upper=upper,
        confidence_level=cl,
        standard_error=se,
        method="wilson_score",
    )


def mean_confidence_interval(
    sample_mean: float,
    sample_std: float,
    sample_size: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """
    Calculate confidence interval for sample mean using Student's t distribution.
    
    Mathematical definition:
      SE = sample_std / sqrt(sample_size)
      t_crit = t_{1 - alpha/2, df=n-1}
      CI = [mean - t_crit * SE, mean + t_crit * SE]
    """
    n = validate_integer(sample_size, "sample_size", min_val=2)
    cl = validate_confidence_level(confidence_level)
    alpha = 1.0 - cl

    se = sample_std / math.sqrt(n)
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)

    lower = sample_mean - t_crit * se
    upper = sample_mean + t_crit * se

    return ConfidenceInterval(
        point_estimate=sample_mean,
        lower=lower,
        upper=upper,
        confidence_level=cl,
        standard_error=se,
        method="student_t",
    )
