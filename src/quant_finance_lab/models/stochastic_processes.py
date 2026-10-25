"""Stochastic process return models including Geometric Brownian Motion (GBM)."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from quant_finance_lab.core.validation import (
    validate_finite,
    validate_positive,
    validate_integer,
)
from quant_finance_lab.models.base import ReturnModel


class GeometricBrownianMotion(ReturnModel):
    """
    Geometric Brownian Motion (GBM) Continuous-Time Asset Price Model.
    
    Under the Black-Scholes-Merton framework:
      dS_t = mu * S_t * dt + sigma * S_t * dW_t
      
    Exact discrete log-return transition over interval dt:
      ln(S_{t+dt} / S_t) = (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z,  Z ~ N(0, 1)
      
    Returns R_t = S_{t+dt} / S_t - 1 = exp( (mu - 0.5 * sigma^2)*dt + sigma*sqrt(dt)*Z ) - 1.
    """

    def __init__(
        self,
        annual_drift: float = 0.08,
        annual_volatility: float = 0.20,
        dt: float = 1.0 / 252.0,  # 1 trading day default
    ) -> None:
        self.annual_drift = validate_finite(annual_drift, "annual_drift")
        self.annual_volatility = validate_positive(annual_volatility, "annual_volatility")
        self.dt = validate_positive(dt, "dt")

    @property
    def name(self) -> str:
        return f"Geometric Brownian Motion (mu={self.annual_drift:.2%}, sigma={self.annual_volatility:.2%}, dt={self.dt:.5f})"

    @property
    def assumptions(self) -> list[str]:
        return [
            "Log returns are normally distributed with constant drift and diffusion.",
            "Continuous paths without jump discontinuities or tail fatness.",
            "Asset prices remain strictly positive: S_t > 0 for all t.",
            "Independent increments across non-overlapping time steps.",
        ]

    def generate_returns(
        self, num_simulations: int, num_trades: int, rng: np.random.Generator
    ) -> np.ndarray:
        num_sims = validate_integer(num_simulations, "num_simulations", min_val=1)
        n_trades = validate_integer(num_trades, "num_trades", min_val=0)
        if n_trades == 0:
            return np.empty((num_sims, 0), dtype=np.float64)

        drift_term = (self.annual_drift - 0.5 * (self.annual_volatility ** 2)) * self.dt
        diffusion_term = self.annual_volatility * math.sqrt(self.dt)

        z = rng.standard_normal(size=(num_sims, n_trades))
        log_increments = drift_term + diffusion_term * z
        return np.exp(log_increments) - 1.0

    def analytical_expected_balance(
        self, initial_capital: float, num_trades: int
    ) -> float:
        b0 = validate_positive(initial_capital, "initial_capital")
        n = validate_integer(num_trades, "num_trades", min_val=0)
        t = n * self.dt
        return b0 * math.exp(self.annual_drift * t)

    def analytical_median_balance(
        self, initial_capital: float, num_trades: int
    ) -> float:
        b0 = validate_positive(initial_capital, "initial_capital")
        n = validate_integer(num_trades, "num_trades", min_val=0)
        t = n * self.dt
        return b0 * math.exp((self.annual_drift - 0.5 * (self.annual_volatility ** 2)) * t)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "GeometricBrownianMotion",
            "name": self.name,
            "annual_drift": self.annual_drift,
            "annual_volatility": self.annual_volatility,
            "dt": self.dt,
            "assumptions": self.assumptions,
        }
