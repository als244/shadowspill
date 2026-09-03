"""Measured against simulated throughput, one point per run budget."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure

_GIB = 1 << 30


def plot_step_run(
    entries: Sequence[tuple[int, float, float]],
    directory: str | Path,
    *,
    tokens_per_step: int,
) -> Path:
    """Write the run-versus-simulation throughput figure and return its path.

    Each entry is one executed budget: ``(execution_budget_bytes,
    simulated_step_seconds, measured_step_seconds)``. The simulated line is
    dashed and muted; the measured line is solid.
    """

    if not entries:
        raise ValueError("at least one executed budget is required")
    ordered = sorted(entries)
    budgets = [item[0] / _GIB for item in ordered]
    figure = Figure(figsize=(6.4, 4.0), dpi=150)
    axes = figure.subplots()
    axes.plot(
        budgets,
        [tokens_per_step / item[1] for item in ordered],
        linestyle="--",
        color="gray",
        marker="o",
        label="Simulated",
    )
    axes.plot(
        budgets,
        [tokens_per_step / item[2] for item in ordered],
        marker="o",
        label="Measured",
    )
    axes.set_title("Throughput, measured against simulated")
    axes.set_xlabel("Execution budget (GiB)")
    axes.set_ylabel("Tokens per second")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "run_throughput.png"
    figure.savefig(path)
    return path
