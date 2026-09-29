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

from shadowspill.plots._axis import budget_label

_GIB = 1 << 30


@dataclass(frozen=True, slots=True)
class RunBudgetOutcome:
    """A budget's simulated invocation, traced invocation, and measured cycles.

    The trace and simulation each split into entry delay, useful compute,
    recomputation, inter-task idle, and terminal transfers. Those disjoint parts
    sum to the corresponding invocation. Median whole-cycle time additionally
    includes caller work and is used separately for throughput.

    Recomputation is the profiled excess over the save-only floor, shown on
    both clocks; differences in actual kernel time stay in useful compute.
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
    #: Invocation entry through the first computation, on each clock.
    simulated_entry_delay_seconds: float
    real_entry_delay_seconds: float
    #: Last computation through completion of required terminal transfers.
    terminal_tail_seconds: float
    real_terminal_tail_seconds: float
    #: What the chosen recomputation costs over the save-only floor. A
    #: counterfactual, so there is nothing to measure it against: the same figure
    #: stands on both clocks. Declared here, among the defaulted fields, because
    #: a dataclass will not take a defaulted field before an undefaulted one.
    recomputation_seconds: float = 0.0
    #: Every step this budget ran, in order -- including the first, which pays
    #: first-call setup, and the traced last one, which pays for its own
    #: collection. ``measured_step_seconds`` is the median of the ones in
    #: between; keeping all of them here is what says whether a budget was steady
    #: or erratic, which a median cannot.
    step_seconds: tuple[float, ...] = ()

    @property
    def relative_error(self) -> float:
        """How far the measurement fell from the prediction, signed.

        Positive means the step ran **slower** than the simulator said it
        would, negative that it ran faster. The sign is worth stating because
        it is the direction that carries the meaning: an optimistic prediction
        is time the plan spends somewhere the simulator does not model, and
        that is a thing to go and find. This is the same convention the
        performance gate reports, so a number here and a number there mean the
        same thing.
        """

        return (
            self.measured_step_seconds - self.simulated_step_seconds
        ) / self.simulated_step_seconds

    @property
    def traced_step_seconds(self) -> float:
        """The traced invocation including required terminal copies."""

        return sum(self.components(measured=True))

    @property
    def trace_relative_error(self) -> float:
        return (
            self.traced_step_seconds - self.simulated_step_seconds
        ) / self.simulated_step_seconds

    def components(self, *, measured: bool) -> tuple[float, ...]:
        """Disjoint entry, compute, recompute, idle, and terminal durations."""

        return (
            self.real_entry_delay_seconds
            if measured
            else self.simulated_entry_delay_seconds,
            (self.real_task_seconds if measured else self.profiled_task_seconds)
            - self.recomputation_seconds,
            self.recomputation_seconds,
            self.real_idle_seconds if measured else self.simulated_idle_seconds,
            self.real_terminal_tail_seconds if measured else self.terminal_tail_seconds,
        )


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
        write_run_tables(
            ordered, target.parent / "raw_data", tokens_per_step=tokens_per_step
        ),
    )


def write_run_tables(
    entries: Sequence[RunBudgetOutcome],
    directory: str | Path,
    *,
    tokens_per_step: int,
) -> Path:
    """The numbers both run figures draw, as one tidy table.

    Separate from drawing so a caller can keep the record current as it goes.
    Figures are worth rendering once at the end; the tables behind them are
    worth having after every budget, because a run that stops early should still
    leave what it measured, and the figures can be redrawn from tables.
    """

    target = Path(directory)
    ordered = sorted(entries, key=lambda item: item.execution_budget_bytes)

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
                "recomputation_seconds",
                "simulated_entry_delay_seconds",
                "real_entry_delay_seconds",
                "traced_step_seconds",
                "trace_relative_error",
                "terminal_tail_seconds",
                "real_terminal_tail_seconds",
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
                item.recomputation_seconds,
                item.simulated_entry_delay_seconds,
                item.real_entry_delay_seconds,
                item.traced_step_seconds,
                item.trace_relative_error,
                item.terminal_tail_seconds,
                item.real_terminal_tail_seconds,
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
        [budget_label(value) for value in budgets],
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
    """Compare the complete traced invocation with the complete simulation."""

    labels = [budget_label(item.execution_budget_bytes / _GIB) for item in ordered]
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
    errors = [item.trace_relative_error for item in ordered]
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
    error.set_title("Traced Invocation Fidelity (positive = ran slower than predicted)")
    error.set_ylabel("Measured Minus Simulated")
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
    colors = ("tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:brown")
    component_labels = (
        "Entry Delay",
        "Effective Compute",
        "Recompute",
        "Inter-task Idle",
        "Terminal Transfers",
    )
    for offset, (name, measured) in enumerate((("Simulated", False), ("Traced", True))):
        centres = [place - width / 2 + width * offset for place in places]
        opacity = 0.95 if measured else 0.45
        totals = [0.0] * len(ordered)
        columns = zip(
            *(item.components(measured=measured) for item in ordered), strict=True
        )
        for values, color in zip(columns, colors, strict=True):
            parts.bar(
                centres,
                values,
                width=width * 0.92,
                bottom=totals,
                color=color,
                alpha=opacity,
            )
            totals = [a + b for a, b in zip(totals, values, strict=True)]
        for centre, total in zip(centres, totals, strict=True):
            parts.annotate(
                f"{name}\n{total:.2f} s",
                (centre, total),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7.0,
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
            max(item.simulated_step_seconds, item.traced_step_seconds)
            for item in ordered
        )
        * 1.30,
    )
    parts.legend(
        handles=[
            Patch(facecolor=color, alpha=0.95, label=label)
            for color, label in zip(colors, component_labels, strict=True)
        ],
        fontsize="x-small",
        loc="upper left",
    )
    figure.tight_layout()
    figure.savefig(path)
    return path
