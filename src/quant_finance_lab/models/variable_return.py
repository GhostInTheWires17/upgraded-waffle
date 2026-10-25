"""Continuous and fat-tailed variable return models."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from quant_finance_lab.core.exceptions import ValidationError
from quant_finance_lab.core.validation import (
    validate_finite,
    validate_positive,
    validate_probability,
    validate_integer,
)
from quant_finance_lab.models.base import ReturnModel


class NormalReturnModel(ReturnModel):
    """
    Gaussian Continuous Return Model.
    
    Trade returns are sampled from a normal distribution R_t ~ N(mu, sigma^2).
    """

    def __init__(self, mean_return: float = 0.001, volatility: float = 0.02) -> None:
        self.mean_return = validate_finite(mean_return, "mean_return")
        self.volatility = validate_positive(volatility, "volatility")

    @property
    def name(self) -> str:
        return f"Normal Return Model (mu={self.mean_return:.4f}, sigma={self.volatility:.4f})"

    @property
    def assumptions(self) -> list[str]:
        return [
            "Trade returns are i.i.d. Gaussian random variables.",
            "Thin tails: extreme events follow normal distribution probabilities.",
            "Stationary mean return and volatility.",
            "Continuous return distribution bounded below at -1.0 (via truncation or clamping).",
        ]

    def generate_returns(
        self, num_simulations: int, num_trades: int, rng: np.random.Generator
    ) -> np.ndarray:
        num_sims = validate_integer(num_simulations, "num_simulations", min_val=1)
        n_trades = validate_integer(num_trades, "num_trades", min_val=0)
        if n_trades == 0:
            return np.empty((num_sims, 0), dtype=np.float64)

        raw = rng.normal(loc=self.mean_return, scale=self.volatility, size=(num_sims, n_trades))
        # Unleveled capital cannot lose more than 100% (-1.0)
        return np.maximum(raw, -1.0)

    def analytical_expected_trade_return(self) -> float:
        return self.mean_return

    def analytical_variance_trade_return(self) -> float:
        return self.volatility ** 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "NormalReturnModel",
            "name": self.name,
            "mean_return": self.mean_return,
            "volatility": self.volatility,
            "assumptions": self.assumptions,
        }


class StudentTReturnModel(ReturnModel):
    """
    Fat-Tailed Student-t Return Model.
    
    Generates heavy-tailed trade returns with configurable degrees of freedom nu.
    Captures financial stylised facts such as leptokurtosis and tail risk.
    """

    def __init__(
        self,
        degrees_of_freedom: float = 5.0,
        loc: float = 0.001,
        scale: float = 0.015,
    ) -> None:
        self.df = validate_positive(degrees_of_freedom, "degrees_of_freedom")
        if self.df <= 2.0:
            raise ValidationError(
                f"degrees_of_freedom must be > 2.0 for finite variance, got {self.df}"
            )
        self.loc = validate_finite(loc, "loc")
        self.scale = validate_positive(scale, "scale")

    @property
    def name(self) -> str:
        return f"Student-t Return Model (df={self.df:.1f}, loc={self.loc:.4f}, scale={self.scale:.4f})"

    @property
    def assumptions(self) -> list[str]:
        return [
            "Trade returns are i.i.d. Student-t distributed.",
            "Heavy/fat tails: higher probability of extreme drawdowns and windfalls than normal.",
            "Stationary parameters across all simulated trades.",
            "Returns bounded below at -1.0 to prevent negative asset balance.",
        ]

    def generate_returns(
        self, num_simulations: int, num_trades: int, rng: np.random.Generator
    ) -> np.ndarray:
        num_sims = validate_integer(num_simulations, "num_simulations", min_val=1)
        n_trades = validate_integer(num_trades, "num_trades", min_val=0)
        if n_trades == 0:
            return np.empty((num_sims, 0), dtype=np.float64)

        raw = rng.standard_t(df=self.df, size=(num_sims, n_trades)) * self.scale + self.loc
        return np.maximum(raw, -1.0)

    def analytical_expected_trade_return(self) -> float:
        return self.loc

    def analytical_variance_trade_return(self) -> float:
        # Variance of Student-t is scale^2 * (nu / (nu - 2)) for nu > 2
        return (self.scale ** 2) * (self.df / (self.df - 2.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "StudentTReturnModel",
            "name": self.name,
            "degrees_of_freedom": self.df,
            "loc": self.loc,
            "scale": self.scale,
            "assumptions": self.assumptions,
        }


class TwoStateVariableReturnModel(ReturnModel):
    """
    Win/Loss Bernoulli trial with variable return magnitudes.
    
    When a trade wins (prob p), return is drawn from N(mu_win, sigma_win^2).
    When a trade loses (prob 1-p), return is drawn from -|N(mu_loss, sigma_loss^2)|.
    """

    def __init__(
        self,
        win_probability: float = 0.55,
        win_mean: float = 0.025,
        win_std: float = 0.005,
        loss_mean: float = 0.015,
        loss_std: float = 0.005,
    ) -> None:
        self.win_probability = validate_probability(win_probability, "win_probability")
        self.win_mean = validate_positive(win_mean, "win_mean")
        self.win_std = validate_positive(win_std, "win_std", allow_zero=True)
        self.loss_mean = validate_positive(loss_mean, "loss_mean")
        self.loss_std = validate_positive(loss_std, "loss_std", allow_zero=True)

    @property
    def name(self) -> str:
        return "Two-State Variable Return Model"

    @property
    def assumptions(self) -> list[str]:
        return [
            "Trades are split between win and loss states with fixed probability p.",
            "Win and loss returns follow distinct normal distributions.",
            "Loss returns are strictly clamped to negative rates.",
        ]

    def generate_returns(
        self, num_simulations: int, num_trades: int, rng: np.random.Generator
    ) -> np.ndarray:
        num_sims = validate_integer(num_simulations, "num_simulations", min_val=1)
        n_trades = validate_integer(num_trades, "num_trades", min_val=0)
        if n_trades == 0:
            return np.empty((num_sims, 0), dtype=np.float64)

        wins = rng.random(size=(num_sims, n_trades)) < self.win_probability
        win_rets = np.maximum(
            rng.normal(self.win_mean, self.win_std, size=(num_sims, n_trades)), 0.0
        )
        loss_rets = -np.abs(
            rng.normal(self.loss_mean, self.loss_std, size=(num_sims, n_trades))
        )
        loss_rets = np.maximum(loss_rets, -1.0)
        return np.where(wins, win_rets, loss_rets)

    def analytical_expected_trade_return(self) -> float:
        return self.win_probability * self.win_mean - (1.0 - self.win_probability) * self.loss_mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "TwoStateVariableReturnModel",
            "name": self.name,
            "win_probability": self.win_probability,
            "win_mean": self.win_mean,
            "win_std": self.win_std,
            "loss_mean": self.loss_mean,
            "loss_std": self.loss_std,
            "assumptions": self.assumptions,
        }
