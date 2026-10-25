"""Example 1: Basic Reproducible Monte Carlo Simulation.

Runs the enhanced core Bernoulli fixed-return trading model with explicit seed,
computes full statistical and risk metrics, prints the console report,
and exports publication-grade charts to the results/ directory.
"""

from pathlib import Path
from quant_finance_lab.simulation.monte_carlo import run_simulation
from quant_finance_lab.reporting.console import print_console_report
from quant_finance_lab.reporting.export import export_all_plots, export_results_to_json


def main() -> None:
    print("Running Quant Finance Lab — Basic Simulation Example...")

    # Run simulation with 10,000 paths and 100 trades, seeded for exact reproducibility
    result = run_simulation(
        initial_capital=1000.0,
        win_probability=0.60,
        win_return=0.02,
        loss_return=-0.01,
        num_simulations=10000,
        num_trades=100,
        seed=42,
        targets=[1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0],
        ruin_threshold=500.0,
        store_paths=True,
    )

    # Print comprehensive terminal report
    print_console_report(result)

    # Export outputs
    out_dir = Path(__file__).resolve().parent.parent / "results" / "basic_simulation"
    saved_plots = export_all_plots(result, out_dir)
    json_path = export_results_to_json(result, out_dir / "simulation_results.json")

    print(f"\n[Artifacts Exported]")
    print(f"  JSON Metadata : {json_path}")
    for name, p in saved_plots.items():
        print(f"  Chart ({name}) : {p}")


if __name__ == "__main__":
    main()
