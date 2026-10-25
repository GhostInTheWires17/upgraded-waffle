"""End-to-end integration tests for complete simulation workflows."""

from pathlib import Path
from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.simulation.monte_carlo import MonteCarloSimulator
from quant_finance_lab.reporting.console import format_console_report
from quant_finance_lab.reporting.export import export_results_to_json, export_results_to_csv, export_all_plots


def test_end_to_end_simulation(tmp_path: Path) -> None:
    cfg = SimulationConfig(
        initial_capital=1000.0,
        num_simulations=500,
        num_trades=50,
        seed=42,
        targets=[1000.0, 1500.0, 2000.0],
        ruin_threshold=500.0,
        store_paths=True,
    )
    model = FixedReturnModel(win_probability=0.60, win_return=0.02, loss_return=-0.01)
    sim = MonteCarloSimulator(config=cfg, model=model)
    res = sim.run()

    # Verify result structure
    assert res.final_balances.shape == (500,)
    assert res.balance_paths is not None and res.balance_paths.shape == (500, 51)
    assert res.drawdown_metrics.median_max_drawdown >= 0.0

    # Verify console formatting
    report = format_console_report(res)
    assert "QUANT FINANCE LAB" in report
    assert "THEORETICAL BENCHMARK" in report

    # Verify exports
    json_path = export_results_to_json(res, tmp_path / "res.json")
    assert json_path.is_file()

    csv_path = export_results_to_csv(res, tmp_path / "res.csv")
    assert csv_path.is_file()

    saved_plots = export_all_plots(res, tmp_path / "plots")
    assert len(saved_plots) >= 4
    for p in saved_plots.values():
        assert p.is_file()
