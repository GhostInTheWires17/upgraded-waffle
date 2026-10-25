"""Position sizing strategies and Kelly Criterion calculations."""

from __future__ import annotations

from typing import Any
from quant_finance_lab.core.exceptions import ValidationError
from quant_finance_lab.core.validation import validate_positive, validate_probability


class PositionSizer:
    """
    Position sizing framework supporting full compounding, fixed fraction, and fixed dollar sizing.
    
    Attributes:
        sizing_type: One of 'full', 'fixed_fractional', 'fixed_dollar'.
        fraction: Allocation fraction in (0.0, 1.0] for fixed_fractional.
        dollar_amount: Fixed dollar stake for fixed_dollar sizing.
    """

    def __init__(
        self,
        sizing_type: str = "full",
        fraction: float = 1.0,
        dollar_amount: float = 100.0,
    ) -> None:
        valid_types = {"full", "fixed_fractional", "fixed_dollar"}
        if sizing_type not in valid_types:
            raise ValidationError(
                f"sizing_type must be one of {valid_types}, got '{sizing_type}'"
            )
        self.sizing_type = sizing_type
        if sizing_type == "fixed_fractional":
            self.fraction = validate_positive(fraction, "fraction")
            if self.fraction > 1.0:
                raise ValidationError(f"fraction must be <= 1.0 for unleveraged sizing, got {fraction}")
        else:
            self.fraction = fraction
        if sizing_type == "fixed_dollar":
            self.dollar_amount = validate_positive(dollar_amount, "dollar_amount")
        else:
            self.dollar_amount = dollar_amount

    def allocate(self, current_balance: float) -> float:
        """Calculate capital allocated to the upcoming trade."""
        if current_balance <= 0.0:
            return 0.0
        if self.sizing_type == "full":
            return current_balance
        if self.sizing_type == "fixed_fractional":
            return current_balance * self.fraction
        if self.sizing_type == "fixed_dollar":
            return min(current_balance, self.dollar_amount)
        return current_balance

    @staticmethod
    def kelly_fraction(
        win_probability: float, win_return: float, loss_magnitude: float
    ) -> float:
        """
        Calculate full Kelly Criterion fraction f* for discrete binary outcomes:
        f* = (p * b - (1 - p)) / b
        where b = win_return / loss_magnitude (reward-to-risk ratio).
        
        Returns optimal compounding fraction, or 0.0 if expectation is non-positive.
        """
        p = validate_probability(win_probability, "win_probability")
        w = validate_positive(win_return, "win_return")
        l = validate_positive(loss_magnitude, "loss_magnitude")
        b = w / l
        f_star = (p * b - (1.0 - p)) / b
        return max(0.0, f_star)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sizing_type": self.sizing_type,
            "fraction": self.fraction,
            "dollar_amount": self.dollar_amount,
        }
