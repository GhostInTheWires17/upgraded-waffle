"""Unit tests for parameter validation utilities."""

import pytest
from quant_finance_lab.core.exceptions import ValidationError
from quant_finance_lab.core.validation import (
    validate_finite,
    validate_probability,
    validate_positive,
    validate_integer,
    validate_return,
    validate_confidence_level,
    validate_targets,
    validate_seed,
)


def test_validate_finite() -> None:
    assert validate_finite(10.5, "test") == 10.5
    with pytest.raises(ValidationError):
        validate_finite(float("nan"), "test")
    with pytest.raises(ValidationError):
        validate_finite(float("inf"), "test")
    with pytest.raises(ValidationError):
        validate_finite("not_a_number", "test")  # type: ignore


def test_validate_probability() -> None:
    assert validate_probability(0.0) == 0.0
    assert validate_probability(1.0) == 1.0
    assert validate_probability(0.55) == 0.55
    with pytest.raises(ValidationError):
        validate_probability(-0.01)
    with pytest.raises(ValidationError):
        validate_probability(1.01)


def test_validate_positive() -> None:
    assert validate_positive(100.0) == 100.0
    with pytest.raises(ValidationError):
        validate_positive(0.0, allow_zero=False)
    assert validate_positive(0.0, allow_zero=True) == 0.0
    with pytest.raises(ValidationError):
        validate_positive(-5.0, allow_zero=True)


def test_validate_integer() -> None:
    assert validate_integer(10) == 10
    with pytest.raises(ValidationError):
        validate_integer(-1, min_val=0)
    with pytest.raises(ValidationError):
        validate_integer(True)  # Booleans are not valid integer parameters  # type: ignore
    with pytest.raises(ValidationError):
        validate_integer(10.5)  # type: ignore


def test_validate_return() -> None:
    assert validate_return(0.05) == 0.05
    assert validate_return(-0.02) == -0.02
    assert validate_return(-1.0, allow_ruin=True) == -1.0
    with pytest.raises(ValidationError):
        validate_return(-1.05)


def test_validate_confidence_level() -> None:
    assert validate_confidence_level(0.95) == 0.95
    with pytest.raises(ValidationError):
        validate_confidence_level(0.0)
    with pytest.raises(ValidationError):
        validate_confidence_level(1.0)
    with pytest.raises(ValidationError):
        validate_confidence_level(1.5)


def test_validate_targets() -> None:
    cleaned = validate_targets([2000.0, 1500.0, 1500.0, 3000.0], 1000.0)
    assert cleaned == [1500.0, 2000.0, 3000.0]
    with pytest.raises(ValidationError):
        validate_targets([-500.0, 1500.0], 1000.0)


def test_validate_seed() -> None:
    assert validate_seed(None) is None
    assert validate_seed(42) == 42
    with pytest.raises(ValidationError):
        validate_seed(-1)
    with pytest.raises(ValidationError):
        validate_seed(True)  # type: ignore
