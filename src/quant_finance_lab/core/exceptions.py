"""Domain-specific exceptions for Quant Finance Lab."""

from __future__ import annotations


class QuantFinanceLabError(Exception):
    """Base exception for all Quant Finance Lab errors."""


class ValidationError(QuantFinanceLabError):
    """Raised when an input parameter, configuration, or data schema violates constraints."""


class SimulationError(QuantFinanceLabError):
    """Raised when an error occurs during simulation setup, execution, or convergence."""


class ModelError(QuantFinanceLabError):
    """Raised when a stochastic return model encounters an invalid state or parameter combination."""


class DataError(QuantFinanceLabError):
    """Raised when historical or input data fails validation, is corrupted, or missing required fields."""


class ResourceLimitError(SimulationError):
    """Raised when requested simulation dimensions exceed safe hardware or memory budgets."""
