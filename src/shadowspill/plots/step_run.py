"""What a run measured, beside what its plan predicted.

Two figures, written under ``real`` beside the search's ``sim``: throughput
measured against simulated, and where the difference between them comes
from. Both need a step to have actually run, which is what separates them
from everything the search writes.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch

_GIB = 1 << 30


@dataclass(frozen=True, slots=True)
class RunBudgetOutcome:
    """One executed budget, with the plan's prediction beside the measurement.

    The step splits the same way on both clocks: the tasks' own compute, the
    stall between them, and -- on the measured side only -- the opening
    restore the simulator does not model. Keeping the parts rather than only
    the totals is what lets a difference be attributed instead of just
    reported.
    """

    execution_budget_bytes: int
    simulated_step_seconds: float
    measured_step_seconds: float
    #: The tasks' own time: what the profiles priced, and what the device
    #: events measured.
    profiled_task_seconds: float
    real_task_seconds: float
    #: Stalled between tasks, simulated and measured.
    simulated_idle_seconds: float
    real_idle_seconds: float
    #: The opening restore, measured only: the simulator assumes the step's
    #: initial objects are already resident.
    prologue_seconds: float
    #: The writeback after the last task, which the simulator prices and the
    #: measured step spends after its last event.
    terminal_tail_seconds: float
    #: Every step this budget ran, in order. ``measured_step_seconds`` is
    #: their median; keeping the rest is what says whether a budget was steady
    #: or erratic, which a median cannot.
    step_seconds: tuple[float, ...] = ()

    @property
    def relative_error(self) -> float:
        """How far the prediction fell from the measurement, signed."""

        return (
            self.simulated_step_seconds - self.measured_step_seconds
        ) / self.measured_step_seconds


def plot_step_run(
    entries: Sequence[RunBudgetOutcome],
    directory: str | Path,
    *,
    tokens_per_step: int,
) -> tuple[Path, ...]:
    """Write the run figures and return their paths."""

    if not entries:
        raise ValueError("at least one executed budget is required")
    ordered = sorted(entries, key=lambda item: item.execution_budget_bytes)
    # `real` beside the search's `sim`, so a run and the search it came from
    # sit together. The caller owns the directory and what distinguishes it.
    target = Path(directory) / "real"
    target.mkdir(parents=True, exist_ok=True)
    return (
        _throughput(target / "throughput.png", ordered, tokens_per_step),
        _fidelity(target / "sim_fidelity.png", ordered),
        _raw_data(target.parent / "raw_data", ordered, tokens_per_step),
    )


def _raw_data(
    target: Path, ordered: Sequence[RunBudgetOutcome], tokens_per_step: int
) -> Path:
    """The numbers both run figures draw, as one tidy table."""

    target.mkdir(parents=True, exist_ok=True)
    path = target / "run_budgets.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "execution_budget_gib",
                "simulated_step_seconds",
                "measured_step_seconds",
                "simulated_tokens_per_second",
                "measured_tokens_per_second",
                "relative_error",
                "profiled_task_seconds",
                "real_task_seconds",
                "simulated_idle_seconds",
                "real_idle_seconds",
                "prologue_seconds",
                "terminal_tail_seconds",
            )
        )
        writer.writerows(
            (
                item.execution_budget_bytes / _GIB,
                item.simulated_step_seconds,
                item.measured_step_seconds,
                tokens_per_step / item.simulated_step_seconds,
                tokens_per_step / item.measured_step_seconds,
                item.relative_error,
                item.profiled_task_seconds,
                item.real_task_seconds,
                item.simulated_idle_seconds,
                item.real_idle_seconds,
                item.prologue_seconds,
                item.terminal_tail_seconds,
            )
            for item in ordered
        )

    steps = target / "steps.csv"
    with steps.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("execution_budget_gib", "step", "seconds", "tokens_per_second")
        )
        writer.writerows(
            (
                item.execution_budget_bytes / _GIB,
                index,
                seconds,
                tokens_per_step / seconds,
            )
            for item in ordered
            for index, seconds in enumerate(item.step_seconds, start=1)
        )
    return path


def _throughput(
    path: Path, ordered: Sequence[RunBudgetOutcome], tokens_per_step: int
) -> Path:
    """Measured throughput over the simulated line, one point per budget."""

    budgets = [item.execution_budget_bytes / _GIB for item in ordered]
    figure = Figure(figsize=(6.4, 4.0), dpi=150)
    axes = figure.subplots()
    axes.plot(
        budgets,
        [tokens_per_step / item.simulated_step_seconds for item in ordered],
        linestyle="--",
        color="gray",
        marker="o",
        label="Simulated",
    )
    axes.plot(
        budgets,
        [tokens_per_step / item.measured_step_seconds for item in ordered],
        marker="o",
        label="Measured",
    )
    axes.set_xticks(budgets)
    axes.set_xticklabels(
        [f"{value:g}" for value in budgets],
        rotation=45 if len(budgets) > 8 else 0,
        ha="right" if len(budgets) > 8 else "center",
    )
    axes.set_title("Throughput, Measured Against Simulated")
    axes.set_xlabel("Execution Budget (GiB)")
    axes.set_ylabel("Tokens per Second")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path)
    return path


def _fidelity(path: Path, ordered: Sequence[RunBudgetOutcome]) -> Path:
    """How far the prediction fell, and which part of the step it missed.

    The upper panel is the headline: the signed error at each budget against
    the bounds the performance gate holds the simulator to. The lower panel
    is the answer to "which part", because a step is compute plus stall and
    the two are modelled by different things -- the profiles and the transfer
    pricing. The opening restore sits on the measured bar alone, since the
    simulator assumes the step's initial objects are already resident.
    """

    labels = [f"{item.execution_budget_bytes / _GIB:g}" for item in ordered]
    places = range(len(ordered))
    # Capped like the search figures: past this the bars narrow rather
    # than the file growing without bound.
    figure = Figure(
        figsize=(min(max(6.4, 1.6 * len(ordered) + 3.0), 26.0), 6.6), dpi=150
    )
    error, parts = figure.subplots(2, 1, sharex=True, height_ratios=(1.0, 1.6))

    for bound, shade in ((0.10, "0.92"), (0.05, "0.84")):
        error.axhspan(-bound, bound, color=shade, zorder=0)
    error.axhline(0.0, color="0.25", linewidth=1.4, zorder=1)
    errors = [item.relative_error for item in ordered]
    error.bar(
        list(places),
        errors,
        width=0.5,
        color=["tab:red" if abs(item) > 0.05 else "tab:blue" for item in errors],
        zorder=2,
    )
    for place, value in zip(places, errors, strict=True):
        error.annotate(
            f"{value:+.1%}",
            (place, value),
            textcoords="offset points",
            xytext=(0, 4 if value >= 0 else -12),
            ha="center",
            fontsize=8.0,
        )
    # Room for the label under a negative bar and over a positive one, and
    # never so tight that the bands become invisible slivers.
    reach = max(0.13, max(abs(value) for value in errors) * 1.35)
    error.set_ylim(-reach, reach)
    error.set_title("Simulator Fidelity")
    error.set_ylabel("Simulated Minus Measured")
    error.yaxis.set_major_formatter(lambda value, _pos: f"{value:+.0%}")
    error.grid(True, axis="y", alpha=0.3)
    error.set_axisbelow(True)
    error.legend(
        handles=[
            Patch(facecolor="0.84", label="Within 5%"),
            Patch(facecolor="0.92", label="Within 10%"),
        ],
        fontsize="x-small",
        loc="upper right",
    )

    width = 0.38
    for offset, (name, compute, idle, prologue) in enumerate(
        (
            (
                "Simulated",
                [item.profiled_task_seconds for item in ordered],
                [item.simulated_idle_seconds for item in ordered],
                [item.terminal_tail_seconds for item in ordered],
            ),
            (
                "Measured",
                [item.real_task_seconds for item in ordered],
                [item.real_idle_seconds for item in ordered],
                [item.prologue_seconds for item in ordered],
            ),
        )
    ):
        centres = [place - width / 2 + width * offset for place in places]
        parts.bar(centres, compute, width=width * 0.92, color="tab:blue", alpha=1.0)
        parts.bar(
            centres,
            idle,
            width=width * 0.92,
            bottom=compute,
            color="tab:orange",
            alpha=0.85,
        )
        parts.bar(
            centres,
            prologue,
            width=width * 0.92,
            bottom=[a + b for a, b in zip(compute, idle, strict=True)],
            color="tab:green",
            alpha=0.7,
        )
        for centre, a, b, c in zip(centres, compute, idle, prologue, strict=True):
            parts.annotate(
                f"{name}\n{a + b + c:.2f} s",
                (centre, a + b + c),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7.0,
                color="0.1",
            )

    parts.set_xticks(list(places))
    parts.set_xticklabels(labels)
    parts.set_xlabel("Execution Budget (GiB)")
    parts.set_ylabel("Seconds")
    parts.set_title("Where the Difference Is", fontsize="small")
    parts.grid(True, axis="y", alpha=0.3)
    parts.set_axisbelow(True)
    parts.set_ylim(
        0.0,
        max(
            item.profiled_task_seconds
            + item.simulated_idle_seconds
            + item.terminal_tail_seconds
            for item in ordered
        )
        * 1.30,
    )
    parts.legend(
        handles=[
            Patch(facecolor="tab:blue", label="Task Compute"),
            Patch(facecolor="tab:orange", alpha=0.85, label="Stalled Between Tasks"),
            Patch(
                facecolor="tab:green",
                alpha=0.7,
                label="Terminal Writeback / Opening Restore",
            ),
        ],
        fontsize="x-small",
        loc="upper left",
    )
    figure.tight_layout()
    figure.savefig(path)
    return path
