"""Command-Line Interface (CLI) for Quant Finance Lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from quant_finance_lab import __version__
from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.simulation.monte_carlo import MonteCarloSimulator
from quant_finance_lab.reporting.console import print_console_report
from quant_finance_lab.reporting.export import (
    export_results_to_json,
    export_results_to_csv,
    export_all_plots,
)
from quant_finance_lab.analysis.sensitivity import sweep_parameter_1d
from quant_finance_lab.analysis.scenarios import run_scenario_analysis


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="quant-finance-lab",
        description="Quant Finance Lab: Modular Quantitative Research & Stochastic Simulation Laboratory.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: simulate
    p_sim = subparsers.add_parser("simulate", help="Run Monte Carlo trading simulation")
    p_sim.add_argument("--initial-capital", type=float, default=1000.0, help="Initial capital balance (default: 1000.0)")
    p_sim.add_argument("--win-prob", type=float, default=0.60, help="Win probability in [0, 1] (default: 0.60)")
    p_sim.add_argument("--win-return", type=float, default=0.02, help="Return on winning trade (default: 0.02 for +2%%)")
    p_sim.add_argument("--loss-return", type=float, default=-0.01, help="Return on losing trade (default: -0.01 for -1%%)")
    p_sim.add_argument("--simulations", type=int, default=10000, help="Number of simulated paths (default: 10000)")
    p_sim.add_argument("--trades", type=int, default=100, help="Number of sequential trades per path (default: 100)")
    p_sim.add_argument("--seed", type=int, default=None, help="Integer random seed for reproducibility")
    p_sim.add_argument("--ruin-threshold", type=float, default=500.0, help="Ruin capital threshold (default: 500.0)")
    p_sim.add_argument("--targets", type=str, default="1000,1500,2000,2500,3000,3500,4000,4500,5000", help="Comma-separated target balances")
    p_sim.add_argument("--export-json", type=str, default=None, help="Path to export JSON results and metadata")
    p_sim.add_argument("--export-csv", type=str, default=None, help="Path to export simulation paths CSV")
    p_sim.add_argument("--save-plots", type=str, default=None, help="Directory to save PNG research charts")

    # Command: sensitivity
    p_sens = subparsers.add_parser("sensitivity", help="Run 1D sensitivity analysis sweep")
    p_sens.add_argument("--param", type=str, default="win_probability", choices=["win_probability", "win_return", "loss_return", "num_trades"], help="Parameter to sweep")
    p_sens.add_argument("--start", type=float, default=0.40, help="Start value")
    p_sens.add_argument("--stop", type=float, default=0.70, help="Stop value")
    p_sens.add_argument("--steps", type=int, default=7, help="Number of evaluation steps")
    p_sens.add_argument("--simulations", type=int, default=5000, help="Simulations per step")
    p_sens.add_argument("--seed", type=int, default=42, help="Reproducible seed for Common Random Numbers")

    # Command: scenarios
    p_scen = subparsers.add_parser("scenarios", help="Run Pessimistic, Base-Case, and Optimistic scenario evaluation")
    p_scen.add_argument("--capital", type=float, default=1000.0, help="Starting balance")
    p_scen.add_argument("--trades", type=int, default=100, help="Number of trades")
    p_scen.add_argument("--simulations", type=int, default=5000, help="Number of simulations")
    p_scen.add_argument("--seed", type=int, default=42, help="Random seed")

    # Command: dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch interactive Streamlit research dashboard")
    p_dash.add_argument("--port", type=int, default=8501, help="Port to host Streamlit dashboard on")

    return parser


def handle_simulate(args: argparse.Namespace) -> int:
    """Execute simulate subcommand."""
    targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
    cfg = SimulationConfig(
        initial_capital=args.initial_capital,
        num_simulations=args.simulations,
        num_trades=args.trades,
        seed=args.seed,
        targets=targets,
        ruin_threshold=args.ruin_threshold,
        store_paths=True,
    )
    model = FixedReturnModel(
        win_probability=args.win_prob,
        win_return=args.win_return,
        loss_return=args.loss_return,
    )

    sim = MonteCarloSimulator(config=cfg, model=model)
    res = sim.run()

    print_console_report(res)

    if args.export_json:
        p = export_results_to_json(res, args.export_json)
        print(f"\n[Export] Results saved to JSON: {p.resolve()}")

    if args.export_csv:
        p = export_results_to_csv(res, args.export_csv)
        print(f"[Export] Path data saved to CSV: {p.resolve()}")

    if args.save_plots:
        saved = export_all_plots(res, args.save_plots)
        print(f"[Export] {len(saved)} plots saved to directory: {Path(args.save_plots).resolve()}")

    return 0


def handle_sensitivity(args: argparse.Namespace) -> int:
    """Execute sensitivity sweep subcommand."""
    values = [float(x) for x in sys.modules["numpy"].linspace(args.start, args.stop, args.steps)]
    print(f"\nRunning 1D Sensitivity Sweep on parameter '{args.param}' across {len(values)} steps...")
    res = sweep_parameter_1d(
        parameter_name=args.param,
        parameter_values=values,
        base_config=SimulationConfig(num_simulations=args.simulations, seed=args.seed),
        use_crn=True,
    )
    df = res.to_dataframe()
    print("\n" + "=" * 70)
    print(" SENSITIVITY SWEEP RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")
    return 0


def handle_scenarios(args: argparse.Namespace) -> int:
    """Execute scenarios subcommand."""
    print("\nExecuting Standard Scenario Analysis (Pessimistic, Base-Case, Optimistic)...")
    df_summary, _ = run_scenario_analysis(
        initial_capital=args.capital,
        num_simulations=args.simulations,
        num_trades=args.trades,
        seed=args.seed,
    )
    print("\n" + "=" * 80)
    print(" SCENARIO EVALUATION SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")
    return 0


def handle_dashboard(args: argparse.Namespace) -> int:
    """Launch Streamlit dashboard."""
    import subprocess
    dash_path = Path(__file__).parent / "dashboard" / "app.py"
    if not dash_path.exists():
        print(f"Error: Dashboard file not found at {dash_path}")
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(dash_path), "--server.port", str(args.port)]
    print(f"Starting Streamlit dashboard on http://localhost:{args.port} ...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    return 0


def main(args: Optional[Sequence[str]] = None) -> int:
    """Entry point for command line execution."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    if parsed_args.command == "simulate":
        return handle_simulate(parsed_args)
    if parsed_args.command == "sensitivity":
        return handle_sensitivity(parsed_args)
    if parsed_args.command == "scenarios":
        return handle_scenarios(parsed_args)
    if parsed_args.command == "dashboard":
        return handle_dashboard(parsed_args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
