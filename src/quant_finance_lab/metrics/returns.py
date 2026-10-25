"""Return metrics, geometric growth, and risk-adjusted ratios."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from quant_finance_lab.core.validation import validate_positive


@dataclass
class ReturnMetrics:
    """Statistical summary of path and final returns."""

    mean_final_balance: float
    median_final_balance: float
    mean_total_return_pct: float
    median_total_return_pct: float
    min_final_balance: float
    max_final_balance: float
    geometric_growth_rate: float
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def compute_return_metrics(
    initial_capital: float,
    final_balances: np.ndarray,
    num_trades: int,
    risk_free_rate: float = 0.0,
    annualization_factor: Optional[float] = None,
) -> ReturnMetrics:
    """
    Compute comprehensive return statistics.
    
    Notice:
      Sharpe and Sortino ratios require an explicit annualization factor (e.g. 252 for daily trades)
      and operate on periodic returns. If annualization_factor is None, ratios are not calculated
      to prevent misleading ungrounded numbers.
    """
    b0 = validate_positive(initial_capital, "initial_capital")
    returns_pct = (final_balances / b0) - 1.0

    mean_balance = float(np.mean(final_balances))
    median_balance = float(np.median(final_balances))

    # Geometric mean growth rate per trade g = exp( (1/N) * sum(ln(B_N / B_0)) ) - 1
    # Only defined if final balances > 0
    positive_balances = final_balances[final_balances > 0.0]
    if len(positive_balances) == len(final_balances) and num_trades > 0:
        mean_log_ratio = float(np.mean(np.log(final_balances / b0)))
        geom_growth = math.exp(mean_log_ratio / num_trades) - 1.0
    else:
        geom_growth = -1.0 if len(positive_balances) == 0 else 0.0

    sharpe: Optional[float] = None
    sortino: Optional[float] = None

    if annualization_factor is not None and annualization_factor > 0 and num_trades > 0:
        # Effective per-trade returns across final balances
        std_ret = float(np.std(returns_pct, ddof=1))
        excess_mean = float(np.mean(returns_pct)) - risk_free_rate
        if std_ret > 1e-12:
            sharpe = (excess_mean / std_ret) * math.sqrt(annualization_factor / num_trades)
        
        # Downside standard deviation for Sortino
        downside = returns_pct[returns_pct < 0.0]
        if len(downside) > 1:
            downside_std = math.sqrt(float(np.mean(downside ** 2)))
            if downside_std > 1e-12:
                sortino = (excess_mean / downside_std) * math.sqrt(annualization_factor / num_trades)

    return ReturnMetrics(
        mean_final_balance=mean_balance,
        median_final_balance=median_balance,
        mean_total_return_pct=float(np.mean(returns_pct)),
        median_total_return_pct=float(np.median(returns_pct)),
        min_final_balance=float(np.min(final_balances)),
        max_final_balance=float(np.max(final_balances)),
        geometric_growth_rate=geom_growth,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
    )
