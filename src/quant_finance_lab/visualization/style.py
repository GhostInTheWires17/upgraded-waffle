import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_research_style() -> None:
    """Configure matplotlib with a clean, academic research aesthetic."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5.5),
            "figure.dpi": 150,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "medium",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 1.75,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "sans-serif",
        }
    )
