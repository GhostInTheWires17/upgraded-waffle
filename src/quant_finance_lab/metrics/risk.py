"""Risk metrics including Ruin Probability, Value-at-Risk (VaR), and Expected Shortfall (CVaR)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from quant_finance_lab.core.validation import validate_confidence_level
from quant_finance_lab.metrics.confidence import wilson_score_interval, ConfidenceInterval


@dataclass
class RiskMetrics:
    """Risk summary including tail loss and ruin probabilities."""

    var_95_dollar: float
    var_95_pct: float
    cvar_95_dollar: float
    cvar_95_pct: float
    var_99_dollar: float
    var_99_pct: float
    cvar_99_dollar: float
    cvar_99_pct: float
    path_ruin_probability: Optional[ConfidenceInterval]
    terminal_ruin_probability: Optional[ConfidenceInterval]

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.path_ruin_probability is not None:
            d["path_ruin_probability"] = self.path_ruin_probability.to_dict()
        if self.terminal_ruin_probability is not None:
            d["terminal_ruin_probability"] = self.terminal_ruin_probability.to_dict()
        return d


def compute_var_cvar(
    initial_capital: float,
    final_balances: np.ndarray,
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float]:
    """
    Compute Value-at-Risk (VaR) and Expected Shortfall (CVaR).
    
    Sign convention:
      Loss L = initial_capital - final_balance.
      Positive values represent financial losses.
      A 95% VaR of $150 means there is a 5% chance of losing $150 or more.
      
    Returns:
        (var_dollar, var_pct, cvar_dollar, cvar_pct)
    """
    cl = validate_confidence_level(confidence_level)
    losses = initial_capital - final_balances
    
    # Empirical percentile for loss
    alpha_pct = 100.0 * cl
    var_dollar = float(np.percentile(losses, alpha_pct))
    var_pct = var_dollar / initial_capital

    # Conditional expectation beyond VaR
    tail_losses = losses[losses >= var_dollar]
    cvar_dollar = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_dollar
    cvar_pct = cvar_dollar / initial_capital

    return var_dollar, var_pct, cvar_dollar, cvar_pct


def analyze_risk(
    initial_capital: float,
    final_balances: np.ndarray,
    balance_paths: Optional[np.ndarray] = None,
    ruin_threshold: Optional[float] = None,
    confidence_level: float = 0.95,
) -> RiskMetrics:
    """Analyze tail risk and ruin probabilities."""
    var_95_d, var_95_p, cvar_95_d, cvar_95_p = compute_var_cvar(
        initial_capital, final_balances, 0.95
    )
    var_99_d, var_99_p, cvar_99_d, cvar_99_p = compute_var_cvar(
        initial_capital, final_balances, 0.99
    )

    path_ruin_ci: Optional[ConfidenceInterval] = None
    terminal_ruin_ci: Optional[ConfidenceInterval] = None
    num_sims = len(final_balances)

    if ruin_threshold is not None:
        term_ruin_count = int(np.sum(final_balances <= ruin_threshold))
        terminal_ruin_ci = wilson_score_interval(
            term_ruin_count, num_sims, confidence_level=confidence_level
        )

        if balance_paths is not None:
            min_balances = np.min(balance_paths, axis=1)
            path_ruin_count = int(np.sum(min_balances <= ruin_threshold))
            path_ruin_ci = wilson_score_interval(
                path_ruin_count, num_sims, confidence_level=confidence_level
            )

    return RiskMetrics(
        var_95_dollar=var_95_d,
        var_95_pct=var_95_p,
        cvar_95_dollar=cvar_95_d,
        cvar_95_pct=cvar_95_p,
        var_99_dollar=var_99_d,
        var_99_pct=var_99_p,
        cvar_99_dollar=cvar_99_d,
        cvar_99_pct=cvar_99_p,
        path_ruin_probability=path_ruin_ci,
        terminal_ruin_probability=terminal_ruin_ci,
    )
