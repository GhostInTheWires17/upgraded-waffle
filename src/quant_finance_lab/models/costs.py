"""Transaction cost and slippage models."""

from __future__ import annotations

from typing import Any
import numpy as np

from quant_finance_lab.core.validation import validate_positive, validate_finite


class TransactionCostModel:
    """
    Models trading friction including fixed fees, broker commissions, and percentage slippage.
    
    Attributes:
        fixed_cost: Dollar cost deducted per executed trade (e.g., $1.00 per trade).
        proportional_cost: Fraction of traded capital lost to spread/slippage/commission (e.g. 0.001 = 10 bps).
    """

    def __init__(self, fixed_cost: float = 0.0, proportional_cost: float = 0.0) -> None:
        self.fixed_cost = validate_positive(fixed_cost, "fixed_cost", allow_zero=True)
        self.proportional_cost = validate_positive(
            proportional_cost, "proportional_cost", allow_zero=True
        )

    @property
    def has_costs(self) -> bool:
        return (self.fixed_cost > 0.0) or (self.proportional_cost > 0.0)

    def net_balance_update(self, current_balance: float, trade_return: float) -> float:
        """
        Compute balance after trade return and transaction cost deduction.
        Ensures balance never falls below 0.0 due to cost deduction alone.
        """
        gross_pnl = current_balance * trade_return
        friction = self.fixed_cost + (current_balance * self.proportional_cost)
        net_balance = current_balance + gross_pnl - friction
        return max(0.0, net_balance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed_cost": self.fixed_cost,
            "proportional_cost": self.proportional_cost,
        }
