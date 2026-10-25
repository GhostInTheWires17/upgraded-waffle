"""Quant Finance Lab: Modular Quantitative Research and Stochastic Simulation Laboratory."""

from quant_finance_lab.core.config import SimulationConfig, RiskThresholds
from quant_finance_lab.core.exceptions import (
    QuantFinanceLabError,
    ValidationError,
    SimulationError,
    ModelError,
    DataError,
)
from quant_finance_lab.core.random_state import RandomStateManager, get_generator

from quant_finance_lab.models.base import ReturnModel
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.models.variable_return import (
    NormalReturnModel,
    StudentTReturnModel,
    TwoStateVariableReturnModel,
)
from quant_finance_lab.models.stochastic_processes import GeometricBrownianMotion
from quant_finance_lab.models.market_regimes import RegimeSwitchingModel
from quant_finance_lab.models.costs import TransactionCostModel
from quant_finance_lab.models.position_sizing import PositionSizer

from quant_finance_lab.simulation.monte_carlo import (
    SimulationResult,
    MonteCarloSimulator,
    run_simulation,
)
from quant_finance_lab.simulation.analytical_benchmarks import (
    AnalyticalBenchmark,
    compare_analytical_vs_mc,
)

from quant_finance_lab.metrics.confidence import (
    ConfidenceInterval,
    wilson_score_interval,
    mean_confidence_interval,
)
from quant_finance_lab.metrics.drawdown import (
    DrawdownMetrics,
    compute_drawdown_series,
    compute_max_drawdowns,
    analyze_drawdowns,
)
from quant_finance_lab.metrics.risk import (
    RiskMetrics,
    compute_var_cvar,
    analyze_risk,
)
from quant_finance_lab.metrics.returns import (
    ReturnMetrics,
    compute_return_metrics,
)
from quant_finance_lab.metrics.distributions import (
    DistributionMetrics,
    TargetMetrics,
    analyze_distribution,
)

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

__version__ = "1.0.0"
__author__ = "Manish Sai"

__all__ = [
    "__version__",
    "__author__",
    # Core
    "SimulationConfig",
    "RiskThresholds",
    "QuantFinanceLabError",
    "ValidationError",
    "SimulationError",
    "ModelError",
    "DataError",
    "RandomStateManager",
    "get_generator",
    # Models
    "ReturnModel",
    "FixedReturnModel",
    "NormalReturnModel",
    "StudentTReturnModel",
    "TwoStateVariableReturnModel",
    "GeometricBrownianMotion",
    "RegimeSwitchingModel",
    "TransactionCostModel",
    "PositionSizer",
    # Simulation
    "SimulationResult",
    "MonteCarloSimulator",
    "run_simulation",
    "AnalyticalBenchmark",
    "compare_analytical_vs_mc",
    # Metrics
    "ConfidenceInterval",
    "wilson_score_interval",
    "mean_confidence_interval",
    "DrawdownMetrics",
    "compute_drawdown_series",
    "compute_max_drawdowns",
    "analyze_drawdowns",
    "RiskMetrics",
    "compute_var_cvar",
    "analyze_risk",
    "ReturnMetrics",
    "compute_return_metrics",
    "DistributionMetrics",
    "TargetMetrics",
    "analyze_distribution",
    # Analysis
    "SensitivityResult1D",
    "SensitivityResult2D",
    "sweep_parameter_1d",
    "sweep_grid_2d",
    "StrategySpec",
    "compare_strategies",
    "run_scenario_analysis",
]
