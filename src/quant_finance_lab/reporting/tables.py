"""Textual and ASCII table formatting for simulation reporting."""

from __future__ import annotations

from typing import Sequence
from quant_finance_lab.metrics.distributions import TargetMetrics
from quant_finance_lab.metrics.drawdown import DrawdownMetrics


def format_target_table(target_metrics: Sequence[TargetMetrics]) -> str:
    """Format target balances and Wilson confidence intervals into a clean ASCII table."""
    lines = [
        " Target ($) | Success Rate | 95% Wilson CI     | Path-Hit Rate",
        "------------|--------------|-------------------|--------------",
    ]
    for tm in target_metrics:
        p_est = tm.terminal_probability.point_estimate * 100.0
        p_low = tm.terminal_probability.lower * 100.0
        p_high = tm.terminal_probability.upper * 100.0
        path_str = (
            f"{tm.path_probability.point_estimate * 100.0:6.2f}%"
            if tm.path_probability
            else "   N/A  "
        )
        lines.append(
            f" {tm.target:10,.0f} | {p_est:11.2f}% | [{p_low:5.2f}%, {p_high:5.2f}%] | {path_str}"
        )
    return "\n".join(lines)


def format_drawdown_table(dd_metrics: DrawdownMetrics) -> str:
    """Format drawdown exceedance probabilities into an ASCII table."""
    lines = [
        " Drawdown Threshold | Exceedance Prob | 95% Wilson CI",
        "---------------------|-----------------|-------------------",
    ]
    for thresh, ci in sorted(dd_metrics.threshold_exceedance.items()):
        p_est = ci.point_estimate * 100.0
        p_low = ci.lower * 100.0
        p_high = ci.upper * 100.0
        lines.append(
            f"         {thresh:5.0%}       |     {p_est:6.2f}%     | [{p_low:5.2f}%, {p_high:5.2f}%]"
        )
    return "\n".join(lines)
