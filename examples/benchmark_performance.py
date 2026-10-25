"""Performance benchmarking for vectorized Monte Carlo execution.

Measures throughput and execution time across 1,000, 10,000, and 100,000 paths
with and without full path retention to evaluate memory-throughput trade-offs.
"""

import time
import platform
import numpy as np
from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.simulation.monte_carlo import MonteCarloSimulator


def benchmark() -> None:
    print("=" * 72)
    print(" QUANT FINANCE LAB — PERFORMANCE BENCHMARK SUITE")
    print("=" * 72)
    print(f"Platform: {platform.platform()} | Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print("-" * 72)
    print(" Simulations | Trades | Mode        | Time (s) | Throughput (paths/sec)")
    print("-------------|--------|-------------|----------|-----------------------")

    model = FixedReturnModel(win_probability=0.60, win_return=0.02, loss_return=-0.01)
    runs = [
        (1000, 100, True),
        (10000, 100, True),
        (50000, 100, True),
        (100000, 100, False),  # Memory-safe running statistics mode
    ]

    for n_sims, n_trades, store_paths in runs:
        cfg = SimulationConfig(
            initial_capital=1000.0,
            num_simulations=n_sims,
            num_trades=n_trades,
            seed=42,
            store_paths=store_paths,
            chunk_size=25000,
        )
        sim = MonteCarloSimulator(config=cfg, model=model)
        
        # Warmup
        if n_sims <= 1000:
            sim.run()

        t0 = time.perf_counter()
        res = sim.run()
        elapsed = time.perf_counter() - t0

        mode_str = "Full Paths" if store_paths else "Memory-Safe"
        throughput = n_sims / elapsed
        print(f" {n_sims:11,d} | {n_trades:6d} | {mode_str:11s} | {elapsed:8.4f} | {throughput:16,.0f}")

    print("=" * 72)


if __name__ == "__main__":
    benchmark()
