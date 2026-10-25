"""Publication-grade Matplotlib visualizers for simulation outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant_finance_lab.simulation.monte_carlo import SimulationResult
from quant_finance_lab.analysis.sensitivity import SensitivityResult2D
from quant_finance_lab.visualization.style import set_research_style


def _handle_save_and_show(fig: plt.Figure, save_path: Optional[str | Path], show: bool) -> plt.Figure:
    """Helper to save and optionally display figure."""
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_final_balance_distribution(
    result: SimulationResult,
    bins: int = 50,
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot histogram of final balances with mean, median, and initial balance markers."""
    set_research_style()
    fig, ax = plt.subplots()

    finals = result.final_balances
    b0 = result.config.initial_capital
    mean_val = result.distribution_metrics.mean
    med_val = result.distribution_metrics.median

    counts, bin_edges, _ = ax.hist(
        finals,
        bins=bins,
        color="#2b5c8f",
        edgecolor="#183654",
        alpha=0.75,
        density=False,
    )

    ax.axvline(b0, color="#d9534f", linestyle="--", linewidth=1.8, label=f"Initial Capital (${b0:,.0f})")
    ax.axvline(med_val, color="#f0ad4e", linestyle="-", linewidth=2.0, label=f"Median (${med_val:,.0f})")
    ax.axvline(mean_val, color="#5cb85c", linestyle="-.", linewidth=2.0, label=f"Mean (${mean_val:,.0f})")

    if result.analytical_benchmark:
        theo_val = result.analytical_benchmark.theoretical_expected_balance
        ax.axvline(theo_val, color="#0275d8", linestyle=":", linewidth=2.0, label=f"Theoretical E[B_N] (${theo_val:,.0f})")

    ax.set_title("Distribution of Simulated Final Balances")
    ax.set_xlabel("Final Balance ($)")
    ax.set_ylabel("Number of Simulations")
    ax.legend(loc="upper right")

    return _handle_save_and_show(fig, save_path, show)


def plot_equity_curves(
    result: SimulationResult,
    num_sample_paths: int = 40,
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot representative sample equity curves with median and percentile confidence bands."""
    if result.balance_paths is None:
        raise ValueError("balance_paths are not stored in this simulation result (store_paths was False).")

    set_research_style()
    fig, ax = plt.subplots()

    paths = result.balance_paths
    n_sims, n_steps = paths.shape
    trades = np.arange(n_steps)

    # Plot faint individual paths
    samples_to_plot = min(num_sample_paths, n_sims)
    for i in range(samples_to_plot):
        ax.plot(trades, paths[i], color="#4a90e2", alpha=0.15, linewidth=0.8)

    # Compute pathwise percentiles
    p10 = np.percentile(paths, 10, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    ax.fill_between(trades, p10, p90, color="#4a90e2", alpha=0.20, label="10th–90th Percentile Band")
    ax.plot(trades, p50, color="#1c3d5a", linewidth=2.5, label="Median Path")
    ax.axhline(result.config.initial_capital, color="#d9534f", linestyle="--", linewidth=1.5, label="Initial Capital")

    if result.config.ruin_threshold is not None:
        ax.axhline(result.config.ruin_threshold, color="#990000", linestyle=":", linewidth=1.5, label=f"Ruin Threshold (${result.config.ruin_threshold:,.0f})")

    ax.set_title(f"Representative Equity Curves ({samples_to_plot} paths shown)")
    ax.set_xlabel("Trade Number (t)")
    ax.set_ylabel("Portfolio Balance ($)")
    ax.legend(loc="upper left")

    return _handle_save_and_show(fig, save_path, show)


def plot_drawdown_distribution(
    result: SimulationResult,
    bins: int = 40,
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot distribution of Maximum Drawdowns (MDD) across simulated paths."""
    set_research_style()
    fig, ax = plt.subplots()

    mdds_pct = result.max_drawdowns * 100.0  # Percentage scale
    med_mdd = result.drawdown_metrics.median_max_drawdown * 100.0
    p95_mdd = result.drawdown_metrics.percentile_95_max_drawdown * 100.0

    ax.hist(mdds_pct, bins=bins, color="#d9534f", edgecolor="#843534", alpha=0.75)
    ax.axvline(med_mdd, color="#2b5c8f", linestyle="-", linewidth=2.0, label=f"Median MDD ({med_mdd:.1f}%)")
    ax.axvline(p95_mdd, color="#000000", linestyle="--", linewidth=2.0, label=f"95th Pct MDD ({p95_mdd:.1f}%)")

    ax.set_title("Distribution of Maximum Drawdowns (MDD)")
    ax.set_xlabel("Maximum Drawdown Magnitude (%)")
    ax.set_ylabel("Number of Simulations")
    ax.legend(loc="upper right")

    return _handle_save_and_show(fig, save_path, show)


def plot_target_probabilities(
    result: SimulationResult,
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot success probability versus target balance with Wilson score confidence intervals."""
    set_research_style()
    fig, ax = plt.subplots()

    t_metrics = result.distribution_metrics.target_probabilities
    if not t_metrics:
        raise ValueError("No target metrics available to plot.")

    targets = [m.target for m in t_metrics]
    term_probs = [m.terminal_probability.point_estimate * 100.0 for m in t_metrics]
    term_err_lower = [
        (m.terminal_probability.point_estimate - m.terminal_probability.lower) * 100.0
        for m in t_metrics
    ]
    term_err_upper = [
        (m.terminal_probability.upper - m.terminal_probability.point_estimate) * 100.0
        for m in t_metrics
    ]

    ax.errorbar(
        targets,
        term_probs,
        yerr=[term_err_lower, term_err_upper],
        fmt="-o",
        color="#2b5c8f",
        capsize=4,
        label="Terminal Target Probability (with 95% Wilson CI)",
    )

    if t_metrics[0].path_probability is not None:
        path_probs = [m.path_probability.point_estimate * 100.0 for m in t_metrics]
        ax.plot(targets, path_probs, "-s", color="#5cb85c", label="Path-Hitting Probability (First Passage)")

    ax.set_title("Success Probability vs Target Balance")
    ax.set_xlabel("Target Balance ($)")
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(-2, 105)
    ax.legend(loc="upper right")

    return _handle_save_and_show(fig, save_path, show)


def plot_sensitivity_heatmap(
    sensitivity_2d: SensitivityResult2D,
    metric: str = "mean_balance",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot 2D parameter sensitivity heatmap."""
    set_research_style()
    fig, ax = plt.subplots()

    df = sensitivity_2d.to_dataframe(metric=metric)
    data = df.values

    cax = ax.imshow(data, cmap="viridis", aspect="auto", origin="lower")
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(metric.replace("_", " ").title())

    ax.set_xticks(np.arange(len(sensitivity_2d.param_x_values)))
    ax.set_xticklabels([f"{x:.3g}" for x in sensitivity_2d.param_x_values])
    ax.set_yticks(np.arange(len(sensitivity_2d.param_y_values)))
    ax.set_yticklabels([f"{y:.3g}" for y in sensitivity_2d.param_y_values])

    ax.set_xlabel(sensitivity_2d.param_x_name.replace("_", " ").title())
    ax.set_ylabel(sensitivity_2d.param_y_name.replace("_", " ").title())
    ax.set_title(f"Parameter Sensitivity Heatmap: {metric.replace('_', ' ').title()}")

    return _handle_save_and_show(fig, save_path, show)


def plot_analytical_vs_mc(
    result: SimulationResult,
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot comparison between Analytical Theoretical Expectation and Monte Carlo Estimate."""
    if result.analytical_benchmark is None:
        raise ValueError("Simulation result does not contain analytical benchmark data.")

    set_research_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bench = result.analytical_benchmark
    categories = ["Analytical\nExpectation", "Monte Carlo\nSample Mean"]
    values = [bench.theoretical_expected_balance, bench.mc_estimated_mean]
    errors = [0.0, bench.standard_error * 1.96]  # 95% margin of error

    bars = ax.bar(categories, values, yerr=errors, capsize=6, color=["#0275d8", "#5cb85c"], width=0.45)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val * 1.01,
            f"${val:,.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Theoretical vs Monte Carlo Estimated Expected Balance")
    ax.set_ylabel("Expected Final Balance ($)")
    y_min = min(values) * 0.90
    y_max = max(values) * 1.10
    ax.set_ylim(y_min, y_max)

    return _handle_save_and_show(fig, save_path, show)
