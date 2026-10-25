"""Analysis package exports."""

from quant_finance_lab.analysis.sensitivity import (
    SensitivityResult1D,
    SensitivityResult2D,
    sweep_parameter_1d,
    sweep_grid_2d,
)
from quant_finance_lab.analysis.strategy_comparison import (
    StrategySpec,
    compare_strategies,
)
from quant_finance_lab.analysis.scenarios import run_scenario_analysis

__all__ = [
    "SensitivityResult1D",
    "SensitivityResult2D",
    "sweep_parameter_1d",
    "sweep_grid_2d",
    "StrategySpec",
    "compare_strategies",
    "run_scenario_analysis",
]
