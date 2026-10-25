"""Unit tests for transaction costs and position sizing."""

import pytest
from quant_finance_lab.models.costs import TransactionCostModel
from quant_finance_lab.models.position_sizing import PositionSizer
from quant_finance_lab.core.exceptions import ValidationError


def test_transaction_cost_deduction() -> None:
    # Starting $1000, trade return +10% (+$100)
    # Fixed fee $2.00, proportional cost 0.1% ($1.00)
    # Gross: $1100, Net: $1100 - $3 = $1097
    model = TransactionCostModel(fixed_cost=2.0, proportional_cost=0.001)
    net = model.net_balance_update(1000.0, 0.10)
    assert net == pytest.approx(1097.0)


def test_position_sizing_allocation() -> None:
    sizer_full = PositionSizer(sizing_type="full")
    assert sizer_full.allocate(1500.0) == 1500.0

    sizer_frac = PositionSizer(sizing_type="fixed_fractional", fraction=0.25)
    assert sizer_frac.allocate(1500.0) == pytest.approx(375.0)

    sizer_dollar = PositionSizer(sizing_type="fixed_dollar", dollar_amount=200.0)
    assert sizer_dollar.allocate(1500.0) == 200.0
    assert sizer_dollar.allocate(50.0) == 50.0  # Capped at balance


def test_kelly_criterion() -> None:
    # 60% win prob, win return +2% (0.02), loss return 1% (0.01)
    # b = 2 / 1 = 2
    # f* = (0.60 * 2 - 0.40) / 2 = (1.2 - 0.4) / 2 = 0.8 / 2 = 0.40 (40%)
    f_star = PositionSizer.kelly_fraction(0.60, 0.02, 0.01)
    assert f_star == pytest.approx(0.40)

    # Negative expectation -> 0.0
    f_neg = PositionSizer.kelly_fraction(0.30, 0.01, 0.02)
    assert f_neg == 0.0
