"""Parameter validation utilities for financial and statistical inputs."""

from __future__ import annotations

import math
from typing import Sequence
from quant_finance_lab.core.exceptions import ValidationError


def validate_finite(value: float, name: str) -> float:
    """Validate that a float value is finite (not NaN, not Inf)."""
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}: {value}")
    val_float = float(value)
    if not math.isfinite(val_float):
        raise ValidationError(f"{name} must be finite, got {val_float}")
    return val_float


def validate_probability(p: float, name: str = "probability") -> float:
    """Validate that a probability lies in [0.0, 1.0]."""
    p_val = validate_finite(p, name)
    if not (0.0 <= p_val <= 1.0):
        raise ValidationError(f"{name} must be in range [0.0, 1.0], got {p_val}")
    return p_val


def validate_positive(value: float, name: str = "value", allow_zero: bool = False) -> float:
    """Validate that a value is positive (or non-negative if allow_zero=True)."""
    val = validate_finite(value, name)
    if allow_zero:
        if val < 0.0:
            raise ValidationError(f"{name} must be non-negative (>= 0), got {val}")
    else:
        if val <= 0.0:
            raise ValidationError(f"{name} must be strictly positive (> 0), got {val}")
    return val


def validate_integer(value: int, name: str = "value", min_val: int = 0) -> int:
    """Validate an integer parameter with a minimum bound."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}: {value}")
    if value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    return value


def validate_return(r: float, name: str = "return", allow_ruin: bool = True) -> float:
    """
    Validate a single trade return rate.
    
    A return rate of -1.0 means a 100% loss (total loss of capital).
    Rates below -1.0 imply liability beyond initial capital (margin call / negative equity).
    If allow_ruin is True, returns >= -1.0 are valid.
    """
    val = validate_finite(r, name)
    min_bound = -1.0 if allow_ruin else -0.999999
    if val < min_bound:
        raise ValidationError(
            f"{name} cannot be less than {min_bound} for unleveraged asset returns, got {val}"
        )
    return val


def validate_confidence_level(cl: float, name: str = "confidence_level") -> float:
    """Validate statistical confidence level, typically in (0.0, 1.0) like 0.95."""
    val = validate_finite(cl, name)
    if not (0.0 < val < 1.0):
        raise ValidationError(f"{name} must be strictly between 0 and 1, got {val}")
    return val


def validate_targets(targets: Sequence[float], initial_capital: float) -> list[float]:
    """Validate and sort target balance thresholds."""
    if not targets:
        return []
    cleaned: list[float] = []
    for i, t in enumerate(targets):
        val = validate_positive(t, f"targets[{i}]")
        cleaned.append(val)
    return sorted(list(set(cleaned)))


def validate_seed(seed: int | None) -> int | None:
    """Validate random seed if provided."""
    if seed is None:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError(f"Seed must be an integer or None, got {type(seed).__name__}: {seed}")
    if seed < 0 or seed >= 2**63:
        raise ValidationError(f"Seed must be in range [0, 2^63 - 1], got {seed}")
    return seed
