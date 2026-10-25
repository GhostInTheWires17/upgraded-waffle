"""Discrete Markov regime-switching stochastic return model."""

from __future__ import annotations

from typing import Any
import numpy as np

from quant_finance_lab.core.validation import (
    validate_probability,
    validate_positive,
    validate_finite,
    validate_integer,
)
from quant_finance_lab.models.base import ReturnModel


class RegimeSwitchingModel(ReturnModel):
    """
    Two-State Markov Regime-Switching Return Model.
    
    Models market transitions between two distinct macroeconomic/volatility regimes:
      - Regime 0 (e.g. Calm / Bull): mean mu_0, volatility sigma_0
      - Regime 1 (e.g. Turbulent / Bear): mean mu_1, volatility sigma_1
      
    State transitions follow a 2x2 Markov transition matrix:
      P = [[p_00, 1 - p_00],
           [1 - p_11, p_11]]
           
    where p_00 is probability of staying in Regime 0, and p_11 is probability of staying in Regime 1.
    """

    def __init__(
        self,
        mu_0: float = 0.001,
        sigma_0: float = 0.010,
        mu_1: float = -0.003,
        sigma_1: float = 0.030,
        p_00: float = 0.95,
        p_11: float = 0.90,
        initial_regime: int = 0,
    ) -> None:
        self.mu_0 = validate_finite(mu_0, "mu_0")
        self.sigma_0 = validate_positive(sigma_0, "sigma_0")
        self.mu_1 = validate_finite(mu_1, "mu_1")
        self.sigma_1 = validate_positive(sigma_1, "sigma_1")
        self.p_00 = validate_probability(p_00, "p_00")
        self.p_11 = validate_probability(p_11, "p_11")
        if initial_regime not in (0, 1):
            raise ValueError(f"initial_regime must be 0 or 1, got {initial_regime}")
        self.initial_regime = initial_regime

    @property
    def name(self) -> str:
        return f"Markov Regime-Switching Model (p00={self.p_00:.2f}, p11={self.p_11:.2f})"

    @property
    def assumptions(self) -> list[str]:
        return [
            "Market switches between two discrete hidden states via a first-order Markov chain.",
            "Conditional on the current regime, trade returns are independent Gaussian variates.",
            "Stationary transition probabilities between regimes across time.",
            "Losses bounded below at -1.0 to reflect asset non-negativity.",
        ]

    def generate_returns(
        self, num_simulations: int, num_trades: int, rng: np.random.Generator
    ) -> np.ndarray:
        num_sims = validate_integer(num_simulations, "num_simulations", min_val=1)
        n_trades = validate_integer(num_trades, "num_trades", min_val=0)
        if n_trades == 0:
            return np.empty((num_sims, 0), dtype=np.float64)

        states = np.zeros((num_sims, n_trades), dtype=np.int32)
        current_state = np.full(num_sims, self.initial_regime, dtype=np.int32)

        p_01 = 1.0 - self.p_00
        p_10 = 1.0 - self.p_11

        # Simulate Markov state transitions sequentially
        for t in range(n_trades):
            states[:, t] = current_state
            u = rng.random(size=num_sims)
            
            # Transition rule
            stay_0 = (current_state == 0) & (u < self.p_00)
            switch_0_to_1 = (current_state == 0) & (~stay_0)
            stay_1 = (current_state == 1) & (u < self.p_11)
            switch_1_to_0 = (current_state == 1) & (~stay_1)

            current_state = np.where(switch_0_to_1, 1, np.where(switch_1_to_0, 0, current_state))

        # Vectorized Gaussian returns conditional on state
        returns_0 = rng.normal(self.mu_0, self.sigma_0, size=(num_sims, n_trades))
        returns_1 = rng.normal(self.mu_1, self.sigma_1, size=(num_sims, n_trades))
        
        returns = np.where(states == 0, returns_0, returns_1)
        return np.maximum(returns, -1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "RegimeSwitchingModel",
            "name": self.name,
            "mu_0": self.mu_0,
            "sigma_0": self.sigma_0,
            "mu_1": self.mu_1,
            "sigma_1": self.sigma_1,
            "p_00": self.p_00,
            "p_11": self.p_11,
            "initial_regime": self.initial_regime,
            "assumptions": self.assumptions,
        }
