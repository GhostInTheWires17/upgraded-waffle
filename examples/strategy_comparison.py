"""Example 5: Strategy and Model Comparison under Identical Seeds.

Compares:
  1. Base Bernoulli Strategy (+2% win, -1% loss, 60% win rate)
  2. Fat-Tailed Student-t Strategy (degrees of freedom = 4)
  3. Regime-Switching Strategy (Bull vs Crisis regime)
  4. Half-Kelly Fixed Fractional Sizing Strategy
"""

from quant_finance_lab.core.config import SimulationConfig
from quant_finance_lab.models.fixed_return import FixedReturnModel
from quant_finance_lab.models.variable_return import StudentTReturnModel
from quant_finance_lab.models.market_regimes import RegimeSwitchingModel
from quant_finance_lab.models.position_sizing import PositionSizer
from quant_finance_lab.analysis.strategy_comparison import StrategySpec, compare_strategies


def main() -> None:
    print("=" * 72)
    print(" QUANT FINANCE LAB — MULTI-STRATEGY COMPARATIVE ANALYSIS")
    print("=" * 72)

    cfg = SimulationConfig(
        initial_capital=1000.0,
        num_simulations=5000,
        num_trades=100,
        seed=42,
    )

    strategies = {
        "1. Base Bernoulli": StrategySpec(
            config=cfg,
            model=FixedReturnModel(win_probability=0.60, win_return=0.02, loss_return=-0.01),
        ),
        "2. Fat-Tailed Student-t": StrategySpec(
            config=cfg,
            model=StudentTReturnModel(degrees_of_freedom=4.0, loc=0.008, scale=0.018),
        ),
        "3. Regime-Switching": StrategySpec(
            config=cfg,
            model=RegimeSwitchingModel(mu_0=0.015, sigma_0=0.010, mu_1=-0.020, sigma_1=0.035, p_00=0.92, p_11=0.85),
        ),
        "4. Fractional Sizing (50%)": StrategySpec(
            config=cfg,
            model=FixedReturnModel(win_probability=0.60, win_return=0.02, loss_return=-0.01),
            position_sizer=PositionSizer(sizing_type="fixed_fractional", fraction=0.50),
        ),
    }

    df_summary, _ = compare_strategies(strategies, shared_seed=42)

    print("\nComparison Summary Table:")
    print(df_summary[["Strategy", "MC Mean Balance", "MC Median Balance", "Profit Probability", "Median Max Drawdown", "95% VaR ($)"]].to_string(index=False))
    print("\n" + "=" * 72)
    print("Takeaway: Sizing rules and regime persistence heavily alter drawdown magnitude")
    print("and tail risk even when expected arithmetic returns look favorable.")
    print("=" * 72)


if __name__ == "__main__":
    main()
