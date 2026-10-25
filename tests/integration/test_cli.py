"""Integration tests for the Command-Line Interface (CLI)."""

from pathlib import Path
from quant_finance_lab.cli import main


def test_cli_simulate(tmp_path: Path, capsys) -> None:
    json_out = str(tmp_path / "cli_res.json")
    code = main([
        "simulate",
        "--initial-capital", "1000",
        "--win-prob", "0.60",
        "--win-return", "0.02",
        "--loss-return", "-0.01",
        "--simulations", "500",
        "--trades", "50",
        "--seed", "42",
        "--export-json", json_out,
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "QUANT FINANCE LAB" in captured.out
    assert Path(json_out).is_file()


def test_cli_sensitivity(capsys) -> None:
    code = main([
        "sensitivity",
        "--param", "win_probability",
        "--start", "0.50",
        "--stop", "0.60",
        "--steps", "3",
        "--simulations", "200",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "SENSITIVITY SWEEP RESULTS" in captured.out


def test_cli_scenarios(capsys) -> None:
    code = main([
        "scenarios",
        "--simulations", "200",
        "--trades", "30",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "SCENARIO EVALUATION SUMMARY" in captured.out
