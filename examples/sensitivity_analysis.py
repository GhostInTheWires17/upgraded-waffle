"""Example 4: Sensitivity Analysis and Common Random Numbers (CRN).

Varies win probability from 40% to 70% and examines the impact on final balances,
profit probability, and median drawdowns under coupled random streams.
"""

from quant_finance_lab.analysis.sensitivity import sweep_parameter_1d, sweep_grid_2d
from quant_finance_lab.core.config import SimulationConfig


def main() -> None:
    print("=" * 72)
    print(" QUANT FINANCE LAB — SENSITIVITY ANALYSIS (CRN)")
    print("=" * 72)

    cfg = SimulationConfig(
        initial_capital=1000.0,
        num_simulations=5000,
        num_trades=100,
        seed=42,
    )

    win_probs = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    print(f"Sweeping win_probability from 40% to 70% with fixed +2% win / -1% loss...")

    res_1d = sweep_parameter_1d(
        parameter_name="win_probability",
        parameter_values=win_probs,
        base_config=cfg,
        use_crn=True,
    )

    df = res_1d.to_dataframe()
    print("\n" + df.to_string(index=False))

    print("\n" + "-" * 72)
    print("Running 2D Grid Sweep: Win Probability vs Loss Return Magnitude...")
    win_grid = [0.50, 0.55, 0.60]
    loss_grid = [-0.005, -0.010, -0.015, -0.020]

    res_2d = sweep_grid_2d(
        param_x_name="win_probability",
        param_x_values=win_grid,
        param_y_name="loss_return",
        param_y_values=loss_grid,
        base_config=cfg,
        use_crn=True,
    )

    print("\nExpected Final Balance Matrix ($):")
    print(res_2d.to_dataframe(metric="mean_balance"))
    print("=" * 72)


if __name__ == "__main__":
    main()
