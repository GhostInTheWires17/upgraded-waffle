"""Fixed binary-outcome (win/loss) stochastic return model."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from quant_finance_lab.core.exceptions import ValidationError
from quant_finance_lab.core.validation import (
    validate_probability,
    validate_return,
    validate_positive,
    validate_integer,
)
from quant_finance_lab.models.base import ReturnModel


class FixedReturnModel(ReturnModel):
    """
    Core Stochastic Return Model: Independent Fixed-Return Trades.
    
    This preserves and rigorously elevates the original simulation logic from v0.0.1 and v0.0.2.
    Each trade outcome is modelled as a Bernoulli trial:
      - With probability p: trade produces positive return R_win (e.g., +0.02)
      - With probability (1 - p): trade produces negative return R_loss (e.g., -0.01)
    
    Balance update:
      B_{t+1} = B_t * (1 + R_t)
      
    Attributes:
        win_probability: Probability of a winning trade in [0.0, 1.0].
        win_return: Return rate on a winning trade (>= 0.0).
        loss_return: Return rate on a losing trade (in [-1.0, 0.0]).
    """

    def __init__(
        self,
        win_probability: float = 0.60,
        win_return: float = 0.02,
        loss_return: float = -0.01,
    ) -> None:
        self.win_probability = validate_probability(win_probability, "win_probability")
        self.win_return = validate_return(win_return, "win_return")
        if self.win_return < 0.0:
            raise ValidationError(
                f"win_return must be non-negative (>= 0.0), got {win_return}"
            )
        
        # Ensure loss_return is non-positive
        self.loss_return = validate_return(loss_return, "loss_return")
        if self.loss_return > 0.0:
            raise ValidationError(
                f"loss_return must be non-positive (<= 0.0), got {loss_return}. "
                f"For a 1% loss, enter -0.01, or use FixedReturnModel.from_magnitudes()."
            )

    @classmethod
    def from_magnitudes(
        cls,
        win_probability_pct: float = 60.0,
        win_return_pct: float = 2.0,
        loss_return_pct: float = 1.0,
    ) -> FixedReturnModel:
        """
        Convenience factory matching the user-friendly percentage input format
        of the original v0.0.2 interactive prompts.
        
        Example:
            model = FixedReturnModel.from_magnitudes(60.0, 2.0, 1.0)
            # Yields win_probability=0.60, win_return=0.02, loss_return=-0.01
        """
        p = win_probability_pct / 100.0
        w = win_return_pct / 100.0
        l = -(loss_return_pct / 100.0)
        return cls(win_probability=p, win_return=w, loss_return=l)

    @property
    def name(self) -> str:
        return "Fixed Bernoulli Return Model"

    @property
    def assumptions(self) -> list[str]:
        return [
            "Independent and identically distributed (i.i.d.) trades across time.",
            "Binary discrete outcomes: exactly one of two return levels occurs each trade.",
            "Stationary probability distribution: win probability p does not change over time.",
            "Frictionless trading unless an explicit cost model is attached.",
            "Instantaneous trade settlement with continuous compounding B_{t+1} = B_t * (1 + R_t).",
            "Zero market impact and infinite liquidity at the specified return rates.",
        ]

    def generate_returns(
        self, num_simulations: int, num_trades: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Vectorized return generation.
        
        Generates uniform random variates U ~ Uniform(0, 1) and applies threshold:
        If U < win_probability -> win_return, else loss_return.
        """
        num_sims = validate_integer(num_simulations, "num_simulations", min_val=1)
        n_trades = validate_integer(num_trades, "num_trades", min_val=0)

        if n_trades == 0:
            return np.empty((num_sims, 0), dtype=np.float64)

        if self.win_probability == 1.0:
            return np.full((num_sims, n_trades), self.win_return, dtype=np.float64)
        if self.win_probability == 0.0:
            return np.full((num_sims, n_trades), self.loss_return, dtype=np.float64)

        # Vectorized Bernoulli trials
        wins = rng.random(size=(num_sims, n_trades)) < self.win_probability
        returns = np.where(wins, self.win_return, self.loss_return)
        return returns

    def analytical_expected_trade_return(self) -> float:
        """E[R_t] = p * R_win + (1 - p) * R_loss"""
        return self.win_probability * self.win_return + (1.0 - self.win_probability) * self.loss_return

    def analytical_variance_trade_return(self) -> float:
        """Var[R_t] = p * (1 - p) * (R_win - R_loss)^2"""
        p = self.win_probability
        return p * (1.0 - p) * ((self.win_return - self.loss_return) ** 2)

    def analytical_expected_balance(
        self, initial_capital: float, num_trades: int
    ) -> float:
        """
        Exact theoretical expected balance:
        E[B_N] = B_0 * (p * (1 + R_win) + (1 - p) * (1 + R_loss))^N
        """
        b0 = validate_positive(initial_capital, "initial_capital")
        n = validate_integer(num_trades, "num_trades", min_val=0)
        multiplier = (
            self.win_probability * (1.0 + self.win_return)
            + (1.0 - self.win_probability) * (1.0 + self.loss_return)
        )
        return b0 * (multiplier ** n)

    def analytical_balance_variance(
        self, initial_capital: float, num_trades: int
    ) -> float:
        """
        Exact theoretical variance of final balance:
        Var[B_N] = B_0^2 * ( (E[(1+R)^2])^N - (E[1+R])^(2N) )
        """
        b0 = validate_positive(initial_capital, "initial_capital")
        n = validate_integer(num_trades, "num_trades", min_val=0)
        e_first = (
            self.win_probability * (1.0 + self.win_return)
            + (1.0 - self.win_probability) * (1.0 + self.loss_return)
        )
        e_second = (
            self.win_probability * ((1.0 + self.win_return) ** 2)
            + (1.0 - self.win_probability) * ((1.0 + self.loss_return) ** 2)
        )
        return (b0 ** 2) * ((e_second ** n) - (e_first ** (2 * n)))

    def analytical_median_balance(
        self, initial_capital: float, num_trades: int
    ) -> float | None:
        """
        Theoretical geometric growth approximation for the median terminal balance:
        g = E[ln(1 + R_t)] = p * ln(1 + R_win) + (1 - p) * ln(1 + R_loss)
        Median[B_N] ~ B_0 * exp(N * g)
        """
        b0 = validate_positive(initial_capital, "initial_capital")
        n = validate_integer(num_trades, "num_trades", min_val=0)
        
        # Handle 100% loss ruin edge case
        if (1.0 + self.loss_return) <= 0.0:
            if self.win_probability < 1.0 and n > 0:
                return 0.0
            return b0 * ((1.0 + self.win_return) ** n)

        g = (
            self.win_probability * math.log(1.0 + self.win_return)
            + (1.0 - self.win_probability) * math.log(1.0 + self.loss_return)
        )
        return b0 * math.exp(n * g)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "FixedReturnModel",
            "name": self.name,
            "win_probability": self.win_probability,
            "win_return": self.win_return,
            "loss_return": self.loss_return,
            "assumptions": self.assumptions,
        }
