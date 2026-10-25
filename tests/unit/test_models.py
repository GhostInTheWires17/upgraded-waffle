"""Unit tests for stochastic return models."""

import numpy as np
import pytest
from quant_finance_lab.core.exceptions import ValidationError
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.models.variable_return import (
    NormalReturnModel,
    StudentTReturnModel,
    TwoStateVariableReturnModel,
)
from quant_finance_lab.models.stochastic_processes import GeometricBrownianMotion
from quant_finance_lab.models.market_regimes import RegimeSwitchingModel


def test_fixed_return_model() -> None:
    model = FixedReturnModel(win_probability=0.60, win_return=0.02, loss_return=-0.01)
    assert model.analytical_expected_trade_return() == pytest.approx(0.60 * 0.02 - 0.40 * 0.01)
    
    # Analytical expected balance
    e_b100 = model.analytical_expected_balance(1000.0, 100)
    assert e_b100 == pytest.approx(1000.0 * (1.008 ** 100))

    # Test generation shape
    rng = np.random.default_rng(42)
    rets = model.generate_returns(50, 20, rng)
    assert rets.shape == (50, 20)
    assert set(np.unique(rets)).issubset({0.02, -0.01})

    # Test validation
    with pytest.raises(ValidationError):
        FixedReturnModel(win_probability=1.5)
    with pytest.raises(ValidationError):
        FixedReturnModel(win_return=-0.05)
    with pytest.raises(ValidationError):
        FixedReturnModel(loss_return=0.01)  # Loss return must be non-positive


def test_fixed_return_from_magnitudes() -> None:
    model = FixedReturnModel.from_magnitudes(60.0, 2.0, 1.0)
    assert model.win_probability == 0.60
    assert model.win_return == 0.02
    assert model.loss_return == -0.01


def test_normal_return_model() -> None:
    model = NormalReturnModel(mean_return=0.001, volatility=0.02)
    assert model.analytical_expected_trade_return() == 0.001
    assert model.analytical_variance_trade_return() == pytest.approx(0.02 ** 2)

    rng = np.random.default_rng(42)
    rets = model.generate_returns(1000, 10, rng)
    assert rets.shape == (1000, 10)
    assert np.all(rets >= -1.0)


def test_student_t_model() -> None:
    model = StudentTReturnModel(degrees_of_freedom=5.0, loc=0.001, scale=0.015)
    assert model.analytical_expected_trade_return() == 0.001
    # Var = scale^2 * (5 / 3)
    assert model.analytical_variance_trade_return() == pytest.approx((0.015 ** 2) * (5.0 / 3.0))

    with pytest.raises(ValidationError):
        StudentTReturnModel(degrees_of_freedom=1.5)  # Must be > 2 for finite variance

    rng = np.random.default_rng(42)
    rets = model.generate_returns(100, 10, rng)
    assert rets.shape == (100, 10)


def test_gbm_model() -> None:
    model = GeometricBrownianMotion(annual_drift=0.10, annual_volatility=0.20, dt=1.0/252.0)
    rng = np.random.default_rng(42)
    rets = model.generate_returns(100, 50, rng)
    assert rets.shape == (100, 50)
    assert np.all(rets > -1.0)  # GBM prices are strictly positive

    theo_e = model.analytical_expected_balance(1000.0, 252)
    assert theo_e == pytest.approx(1000.0 * np.exp(0.10))


def test_regime_switching_model() -> None:
    model = RegimeSwitchingModel(
        mu_0=0.01, sigma_0=0.01,
        mu_1=-0.02, sigma_1=0.03,
        p_00=0.9, p_11=0.85
    )
    rng = np.random.default_rng(42)
    rets = model.generate_returns(100, 20, rng)
    assert rets.shape == (100, 20)
