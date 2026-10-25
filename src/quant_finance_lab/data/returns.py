"""Return calculation and empirical statistics calibration."""

from __future__ import annotations

import math
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

from quant_finance_lab.core.exceptions import DataError


def compute_price_returns(
    df: pd.DataFrame, price_col: str = "close", log_returns: bool = False
) -> pd.Series:
    """Compute 1-period sequential simple or log returns from price series."""
    if price_col not in df.columns:
        raise DataError(f"Column '{price_col}' not present in DataFrame.")

    prices = df[price_col].astype(float)
    if log_returns:
        rets = np.log(prices / prices.shift(1)).dropna()
    else:
        rets = (prices / prices.shift(1) - 1.0).dropna()
    return rets


def summarize_empirical_returns(returns: pd.Series | np.ndarray) -> Dict[str, Any]:
    """
    Extract empirical distribution statistics from observed returns.
    
    Can be used to calibrate stochastic models (e.g., FixedReturnModel, StudentTReturnModel)
    to real historical asset behavior.
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        raise DataError("Need at least 2 return observations.")

    wins = arr[arr > 0.0]
    losses = arr[arr < 0.0]

    win_prob = len(wins) / n
    mean_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    return {
        "num_observations": n,
        "win_probability": win_prob,
        "mean_win_return": mean_win,
        "mean_loss_return": mean_loss,
        "overall_mean_return": float(np.mean(arr)),
        "volatility": float(np.std(arr, ddof=1)),
        "skewness": float(stats.skew(arr, bias=False)) if n > 2 else 0.0,
        "excess_kurtosis": float(stats.kurtosis(arr, bias=False)) if n > 3 else 0.0,
    }
