"""Unit tests for statistical reliability, drawdown, risk, and return metrics."""

import numpy as np
import pytest
from quant_finance_lab.metrics.confidence import wilson_score_interval, mean_confidence_interval
from quant_finance_lab.metrics.drawdown import (
    compute_drawdown_series,
    compute_max_drawdowns,
    analyze_drawdowns,
)
from quant_finance_lab.metrics.risk import compute_var_cvar
from quant_finance_lab.metrics.distributions import analyze_distribution


def test_wilson_score_interval_reference() -> None:
    # Known reference case: 70 successes out of 100 trials, 95% confidence
    ci = wilson_score_interval(70, 100, confidence_level=0.95)
    assert ci.point_estimate == 0.70
    assert 0.60 < ci.lower < 0.70
    assert 0.70 < ci.upper < 0.79
    assert ci.lower <= ci.point_estimate <= ci.upper

    # Edge cases: 0 and 100
    ci_zero = wilson_score_interval(0, 100, confidence_level=0.95)
    assert ci_zero.point_estimate == 0.0
    assert ci_zero.lower == 0.0
    assert ci_zero.upper > 0.0

    ci_full = wilson_score_interval(100, 100, confidence_level=0.95)
    assert ci_full.point_estimate == 1.0
    assert ci_full.lower < 1.0
    assert ci_full.upper == 1.0


def test_drawdown_known_sequence() -> None:
    # Known single balance path: [100, 120, 90, 110, 80]
    # Running peaks: [100, 120, 120, 120, 120]
    # Drawdowns: [0, 0, 90/120 - 1 = -0.25, 110/120 - 1 = -0.08333, 80/120 - 1 = -0.33333]
    # Max drawdown magnitude: 0.33333...
    paths = np.array([[100.0, 120.0, 90.0, 110.0, 80.0]])
    dds = compute_drawdown_series(paths)
    assert dds[0, 0] == 0.0
    assert dds[0, 1] == 0.0
    assert dds[0, 2] == pytest.approx(-0.25)
    assert dds[0, 3] == pytest.approx(-10.0 / 120.0)
    assert dds[0, 4] == pytest.approx(-40.0 / 120.0)

    mdds = compute_max_drawdowns(paths)
    assert mdds[0] == pytest.approx(40.0 / 120.0)


def test_var_cvar_calculation() -> None:
    # Initial capital = 1000
    # Final balances = [700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    # Losses = 1000 - final_balances = [300, 200, 100, 0, -100, -200, -300, -400, -500, -600]
    finals = np.array([700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0])
    var_d, var_p, cvar_d, cvar_p = compute_var_cvar(1000.0, finals, confidence_level=0.90)
    assert var_d > 0.0  # 90% VaR identifies tail loss
    assert cvar_d >= var_d  # CVaR is always >= VaR
