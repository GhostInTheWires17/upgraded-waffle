"""Abstract base class for stochastic return generating models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class ReturnModel(ABC):
    """
    Abstract base class for stochastic trading return models.
    
    Subclasses define specific stochastic mechanisms for generating trade-by-trade returns
    (e.g., Bernoulli fixed returns, normal/student-t distributions, regime-switching, GBM).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the model."""

    @property
    @abstractmethod
    def assumptions(self) -> list[str]:
        """Explicit list of mathematical and financial assumptions underlying the model."""

    @abstractmethod
    def generate_returns(
        self, num_simulations: int, num_trades: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate a 2D matrix of trade returns.
        
        Args:
            num_simulations: Number of paths (rows).
            num_trades: Number of sequential trades per path (columns).
            rng: NumPy Generator instance.
            
        Returns:
            np.ndarray of shape (num_simulations, num_trades) representing return rates R_t.
            Balance update rule is B_{t+1} = B_t * (1 + R_t).
        """

    def analytical_expected_trade_return(self) -> float | None:
        """
        Theoretical single-trade expectation E[R_t], or None if closed-form is unavailable.
        """
        return None

    def analytical_variance_trade_return(self) -> float | None:
        """
        Theoretical single-trade variance Var[R_t], or None if closed-form is unavailable.
        """
        return None

    def analytical_expected_balance(
        self, initial_capital: float, num_trades: int
    ) -> float | None:
        """
        Theoretical expected final balance E[B_N] after N independent trades,
        or None if not analytically tractable.
        """
        return None

    def analytical_median_balance(
        self, initial_capital: float, num_trades: int
    ) -> float | None:
        """
        Theoretical or log-approximate median final balance after N trades.
        """
        return None

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize model configuration for reporting and metadata."""
