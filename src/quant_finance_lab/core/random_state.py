"""Reproducible random number generation and stream management."""

from __future__ import annotations

import numpy as np
from typing import Sequence
from quant_finance_lab.core.validation import validate_seed


class RandomStateManager:
    """
    Manages local, reproducible NumPy random generators.
    
    Adheres to modern NumPy best practices:
    - Never modifies global random state.
    - Uses SeedSequence for independent streams.
    - Provides transparent seed logging for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.original_seed = validate_seed(seed)
        self.seed_seq = np.random.SeedSequence(self.original_seed)
        self._rng = np.random.default_rng(self.seed_seq)

    @property
    def rng(self) -> np.random.Generator:
        """Return the primary random generator."""
        return self._rng

    def spawn(self, n_streams: int) -> list[np.random.Generator]:
        """
        Spawn independent child random generators using NumPy SeedSequence.
        
        This prevents inter-stream correlation and avoids seed collision
        across batched or parallel experiments.
        """
        if n_streams < 1:
            raise ValueError(f"n_streams must be >= 1, got {n_streams}")
        child_sequences = self.seed_seq.spawn(n_streams)
        return [np.random.default_rng(s) for s in child_sequences]

    def reset(self) -> None:
        """Reset the generator back to its original seed state."""
        self.seed_seq = np.random.SeedSequence(self.original_seed)
        self._rng = np.random.default_rng(self.seed_seq)

    def to_metadata(self) -> dict[str, str | int | None]:
        """Return metadata dictionary for reporting and persistence."""
        return {
            "seed": self.original_seed,
            "generator": "numpy.random.default_rng",
            "bit_generator": type(self._rng.bit_generator).__name__,
        }


def get_generator(seed_or_rng: int | np.random.Generator | None = None) -> np.random.Generator:
    """Helper to convert int, None, or existing Generator to a valid Generator."""
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    val_seed = validate_seed(seed_or_rng)
    return np.random.default_rng(val_seed)
