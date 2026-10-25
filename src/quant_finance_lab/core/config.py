"""Simulation configuration dataclasses with boundary validation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence
from quant_finance_lab.core.validation import (
    validate_positive,
    validate_integer,
    validate_confidence_level,
    validate_targets,
    validate_seed,
)


@dataclass
class RiskThresholds:
    """Threshold configurations for ruin, drawdowns, and loss boundaries."""

    ruin_level: float | None = None
    drawdown_thresholds: list[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.30, 0.50]
    )
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.ruin_level is not None:
            self.ruin_level = validate_positive(self.ruin_level, "ruin_level", allow_zero=True)
        self.confidence_level = validate_confidence_level(self.confidence_level)
        for i, d in enumerate(self.drawdown_thresholds):
            if not (0.0 < d <= 1.0):
                raise ValueError(f"drawdown_thresholds[{i}] must be in (0.0, 1.0], got {d}")


@dataclass
class SimulationConfig:
    """
    Primary configuration container for Monte Carlo trading simulations.
    
    Attributes:
        initial_capital: Starting capital (balance at t=0). Must be > 0.
        num_simulations: Total number of independent paths (experiments). Must be >= 1.
        num_trades: Number of sequential trades per path (t=1..N). Must be >= 0.
        seed: Random seed for reproducibility. If None, system entropy is used.
        targets: Target balance thresholds for success rate analysis.
        ruin_threshold: Capital level at or below which path is classified as ruined (first passage).
        confidence_level: Statistical confidence level for confidence intervals (default 0.95).
        chunk_size: Maximum paths to process concurrently in memory to prevent OOM.
        store_paths: If True, stores full balance history (num_simulations, num_trades+1).
                     If False, retains only final balances, min balances, and max drawdowns.
    """

    initial_capital: float = 1000.0
    num_simulations: int = 10000
    num_trades: int = 100
    seed: int | None = None
    targets: list[float] = field(
        default_factory=lambda: [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0]
    )
    ruin_threshold: float | None = 500.0
    confidence_level: float = 0.95
    chunk_size: int = 50000
    store_paths: bool = True

    def __post_init__(self) -> None:
        self.initial_capital = validate_positive(self.initial_capital, "initial_capital")
        self.num_simulations = validate_integer(self.num_simulations, "num_simulations", min_val=1)
        self.num_trades = validate_integer(self.num_trades, "num_trades", min_val=0)
        self.seed = validate_seed(self.seed)
        self.confidence_level = validate_confidence_level(self.confidence_level, "confidence_level")
        self.chunk_size = validate_integer(self.chunk_size, "chunk_size", min_val=1)
        self.targets = validate_targets(self.targets, self.initial_capital)
        if self.ruin_threshold is not None:
            self.ruin_threshold = validate_positive(
                self.ruin_threshold, "ruin_threshold", allow_zero=True
            )
            if self.ruin_threshold >= self.initial_capital:
                raise ValueError(
                    f"ruin_threshold ({self.ruin_threshold}) must be strictly less than "
                    f"initial_capital ({self.initial_capital})"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        """Construct configuration from dictionary."""
        return cls(**data)
