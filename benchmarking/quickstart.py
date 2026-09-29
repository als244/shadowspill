"""Search, plan, run, and plot one model under ShadowSpill.

The quickstart takes a sequence length and a per-step sequence total,
searches every microbatch-by-accumulation split of that total through
planning across the requested execution budgets, optionally renders
figures over the winners, and runs the requested budgets' winners: what
each chosen plan promises, what each step delivers, and how a traced step
compares with the simulator's prediction. See benchmarking/quickstart.md
for the full guide.

Run from the repository root, for example:

    python -m benchmarking.quickstart mlops_llama3 \\
        --sequence-length 1024 --sequences-per-step 64 \\
        --search-budget-gib 6,7,8,9,10,12,16,20,24,28,30 \\
        --run-budget-gib 6,7,8,9,10,12,16,20,24,28,30 \\
        --spill-gib 112 --steps 5 \\
        --min-tokens-per-microbatch 4096 \\
        --plots

Every flag defaults to the model's retained qualification value, so
`python -m benchmarking.quickstart mlops_olmoe` searches and runs the
known cell.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import functools
import gc
import json
import resource
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import torch
from mlops.dispatch import set_weight_gradient_dtype

from shadowspill.diagnostics.step import TaskRecord, TransferRecord
from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.pipeline.common import planned_transfer_bandwidths
from shadowspill.planner import (
    GenericPlanningOptions,
    SearchOptions,
    StepDataOrdering,
)
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import (
    PressureFitOptions,
)
from shadowspill.planner.search.toolkit.resolution import (
    NAMED_RESOLUTION_OPTIONS,
    named_resolution_options,
    validate_resolution_options,
)
from shadowspill.plots import (
    RunBudgetOutcome,
    plot_step_run,
    plot_step_search,
    write_run_tables,
)
from shadowspill.pytorch import (
    ProfilingOptions,
    Runtime,
    StepSearchReport,
    plan_step,
    plan_step_search,
)
from shadowspill.runtime.configuration import (
    resolve_execution_budget,
)
from shadowspill.runtime.failures import RuntimeExecutionError
from shadowspill.schema import artifact_schema
from shadowspill.search import search_geometries
from shadowspill.store import STORE_MODES
from tools.diagnostics.occupancy import write_run_timelines
from tools.qualification.model_state import release_case_model
from workloads.common.training import LEARNING_RATE
from workloads.full_model import HEAD_LOSS_METRIC, build_case, manifest_for
from workloads.providers import ModelImplementation


def _budget_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


#: Named resolution options. ``quarters`` is the library's own default,
#: spelled out here so every run records the shares it actually planned.
def _transfer_bandwidths(value: str) -> TransferBandwidths:
    """``FETCH,EVICT[,FETCH_US,EVICT_US]``, or a search.json to pin to."""

    path = Path(value)
    if path.suffix == ".json":
        try:
            recorded = StepSearchReport.load(path).planned_lanes
        except (OSError, ValueError) as error:
            raise argparse.ArgumentTypeError(f"{value}: {error}") from error
        if recorded is None:
            raise argparse.ArgumentTypeError(f"{value} records no transfer calibration")
        return recorded
    parts = [item.strip() for item in value.split(",")]
    if len(parts) not in (2, 4):
        raise argparse.ArgumentTypeError(
            "expected FETCH,EVICT in GB/s, optionally followed by the fetch and"
            " evict latencies in microseconds, or the path of a search.json"
        )
    try:
        fetch, evict = (int(float(item) * 1e9) for item in parts[:2])
        latencies = tuple(int(float(item) * 1e3) for item in parts[2:])
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value}: {error}") from error
    return TransferBandwidths(
        fetch,
        evict,
        provenance=f"quickstart --transfer-bandwidths {value}",
        fetch_latency_ns=latencies[0] if latencies else None,
        evict_latency_ns=latencies[1] if latencies else None,
    )


IDENTITIES = (
    "mlops_llama3",
    "mlops_qwen35",
    "mlops_olmoe",
    "pytorch_llama3",
    "pytorch_qwen35",
)
_GIB = 1 << 30
_GB = 1_000_000_000


def rule(title: str) -> str:
    head = f"── {title} "
    return head + "─" * max(0, 68 - len(head))


def bar(fraction: float) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * 16)
    return "█" * filled + "░" * (16 - filled)


def _revision() -> str:
    """The revision a run measured, so its outputs name the code they describe.

    A modified tree is marked, because a run from one is not reproducible from
    the hash alone. Falls back to `nogit` outside a checkout rather than
    failing: the outputs are still worth keeping, they just cannot be traced
    to a commit.
    """

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        modified = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return f"{revision}_dirty" if modified else revision


def _started_at() -> str:
    """When a run began, as `MMDD_HHMM`, so reruns of one revision stay apart.

    A revision does not identify a run on its own: the same commit gets
    measured more than once -- on a quiet machine, after a rebuild, against
    another run's store -- and each of those is a measurement worth keeping
    beside the others rather than on top of them.
    """

    return time.strftime("%m%d_%H%M")


def gib(value: float) -> str:
    return f"{value / _GIB:.2f} GiB"


def gb_s(value: float) -> str:
    """Bandwidth in decimal GB/s, fine enough to show a coarsened rate exactly."""

    return f"{value / _GB:.1f} GB/s"


def host_memory() -> tuple[int, int]:
    """Return this process's resident and peak-resident host bytes.

    A **pinned** spill arena is one page-locked mapping, so it counts in full
    from the moment the runtime registers it; everything the frontend holds
    on the host -- imported state, captured optimizer state, compiled
    artifacts -- counts on top of it.

    A **remote** arena counts for nothing here, because it is the peer's
    memory. So these figures are not comparable between a local tour and a
    remote one without saying which: the local number carries the arena and the
    remote number does not. The closing report says which it is rather than
    leaving the two to be read side by side.
    """

    pages = int(Path("/proc/self/statm").read_text().split()[1])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return pages * resource.getpagesize(), peak * 1024


def host_memory_ceiling() -> int | None:
    """Return the tightest cgroup host-memory limit in force, or None.

    A batch scheduler enforces its memory reservation as a cgroup limit, and
    the kernel answers an overrun with SIGKILL, not with an error this
    process could catch and report. Reading the ceiling is what lets a run
    say how much margin it had while it still had some.
    """

    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    unified = [line for line in entries if line.startswith("0::")]
    if not unified:
        return None
    root = Path("/sys/fs/cgroup")
    directory = root / unified[0][3:].strip().lstrip("/")
    limits: list[int] = []
    while True:
        try:
            value = (directory / "memory.max").read_text().strip()
        except OSError:
            value = "max"
        if value.isdigit():
            limits.append(int(value))
        if directory == root:
            return min(limits, default=None)
        directory = directory.parent


def note_host_memory(log: Any, label: str) -> None:
    """Stamp the host-memory high-water mark into the progress log."""

    resident, peak = host_memory()
    ceiling = host_memory_ceiling()
    message = (
        f"host memory {gib(resident)} resident, {gib(peak)} peak"
        f"{'' if ceiling is None else f' of {gib(ceiling)}'}: {label}"
    )
    if log is not None:
        log.note(message)
    else:
        print(f"  {message}", flush=True)


def deltas(label: str, values: list[float], worst_key: str = "") -> str:
    magnitudes = sorted(abs(item) for item in values)
    p95 = magnitudes[min(len(magnitudes) - 1, int(len(magnitudes) * 0.95))]
    worst = max(values, key=abs)
    tail = f"  ({worst_key})" if worst_key else ""
    return (
        f"    {label}   median {statistics.median(values) * 1e3:+7.2f} ms"
        f"   p95 {p95 * 1e3:7.2f} ms   worst {worst * 1e3:+8.2f} ms{tail}"
    )


class PlanLog:
    """Stream planner phase lines to a log file; keep stdout for the story.

    Every line written while this sink is installed lands in the log with a
    wall-clock stamp, so the file reads like ``plan_step``'s own progress and
    can be followed live. Planner phase lines (``[shadowspill.plan …]``,
    ``PressureFit: …``) go only to the file; everything else also reaches the
    terminal.
    """

    def __init__(self, handle: Any, stdout: Any) -> None:
        self._handle = handle
        self._stdout = stdout
        self._started = time.perf_counter()
        self._line_start = True
        self._quiet = False

    def note(self, message: str) -> None:
        elapsed = time.perf_counter() - self._started
        self.write(f"[shadowspill.search +{elapsed:8.3f}s] {message}\n")

    def write(self, text: str) -> int:
        # print() delivers the message and its newline as separate writes,
        # so track line boundaries across calls.
        for line in text.splitlines(keepends=True):
            if self._line_start:
                self._quiet = line.startswith(("[shadowspill.plan", "PressureFit:"))
                self._handle.write(time.strftime("%H:%M:%S "))
            self._handle.write(line)
            if not self._quiet:
                self._stdout.write(line)
            self._line_start = line.endswith("\n")
        self._handle.flush()
        self._stdout.flush()
        return len(text)

    def flush(self) -> None:
        self._handle.flush()
        self._stdout.flush()


def search_policy(arguments: argparse.Namespace) -> SearchOptions:
    """The one search policy this tour plans with.

    Both planning phases have to be told the same thing: the geometry search ranks
    candidates under a policy, and the run then plans the plan that search promised.
    Building the policy twice is how the two drift -- a flag added to one call and not
    the other changes what is measured without changing what is reported -- so it is
    built here and threaded, and every phase is handed this value.
    """

    return SearchOptions(
        generic=GenericPlanningOptions(deterministic=arguments.deterministic),
        algorithm=PressureFit(
            PressureFitOptions(
                resolution_options=tuple(
                    Fraction(share) for share in arguments.resolution_options
                ),
            )
        ),
    )


def profiling_policy(arguments: argparse.Namespace) -> ProfilingOptions:
    """One effective policy shared by search and execution planning."""

    return ProfilingOptions(
        **{
            option.name: getattr(arguments, f"profile_{option.name}")
            for option in fields(ProfilingOptions)
        }
    )


def print_search(report: StepSearchReport, tokens_per_step: int) -> None:
    print(rule("Geometry search"))
    print(
        "  seqs/microbatch x accumulation, then the walk as depth x breadth"
        " (r: reversed backward, p: paired loss); fastest simulated step wins"
    )
    lanes = report.planned_lanes
    if lanes is not None:
        print(
            f"  planned against fetch {gb_s(lanes.fetch_bytes_per_second)},"
            f" evict {gb_s(lanes.evict_bytes_per_second)}"
            + (
                " (pinned)"
                if report.transfer_bandwidths is not None
                else " (calibrated)"
            )
        )
    for execution, spill in report.budgets:
        if len(report.budgets) > 1:
            print(f"  execution {gib(execution)}:")
        winner = report.winner(execution, spill)
        for point in report.points:
            if (
                point.execution_budget_bytes != execution
                or point.spill_budget_bytes != spill
            ):
                continue
            mark = "►" if point is winner else " "
            shape = (
                f"{point.sequences_per_microbatch} x {point.accumulation_count}"
                f" {point.ordering.label}"
            )
            if point.status == "succeeded" and point.makespan_seconds is not None:
                outcome = (
                    f"{point.makespan_seconds:8.3f} s"
                    f"   {tokens_per_step / point.makespan_seconds:>10,.0f} tok/s"
                )
                if point.incumbent_budget_bytes is not None:
                    outcome += f"   plan from {gib(point.incumbent_budget_bytes)}"
            else:
                outcome = point.status
            print(f"  {mark} {shape:>16}   {outcome}")
    for sequences, accumulation, reason in report.skipped:
        print(f"    {sequences} x {accumulation:<4} skipped: {reason}")
    print(
        f"  builds {report.total_build_seconds:.1f} s across"
        f" {len(report.geometries)} geometries"
        f"   searches {report.total_search_seconds:.1f} s"
    )
    print()


def print_breakdown(report: Any, tokens: int) -> None:
    summary = report.summary
    simulated = summary.simulated_step_seconds
    print(rule("The chosen plan's breakdown"))
    print(f"  simulated step   {simulated:8.3f} s   {tokens / simulated:>10,.0f} tok/s")
    print(
        f"  unconstrained    {summary.unconstrained_step_seconds:8.3f} s"
        f"   {tokens / summary.unconstrained_step_seconds:>10,.0f} tok/s"
        "   (cheapest graphs, no waiting)"
    )
    print()
    print(f"  where the simulated {simulated:.3f} s goes")
    # `PlanSummary` identifies the step in four parts exactly; the terminal
    # writeback folds in with the idle because the simulator prices it inside
    # the same makespan. That is the same three the figures draw, so the
    # console and the plots partition the step the same way, and the shares
    # sum to the whole rather than to whatever is left over.
    for label, value in (
        ("effective compute", summary.unconstrained_step_seconds),
        ("recomputation", summary.recomputation_overhead_seconds),
        (
            "stalled",
            summary.idle_seconds + summary.terminal_writeback_seconds,
        ),
    ):
        share = value / simulated if simulated > 0 else 0.0
        print(f"    {label:<22}{value:8.3f} s  {bar(share)}  {share:6.1%}")
    forced = summary.task_alternative_group_count - summary.flexible_group_count
    print(
        f"  recomputation chosen for {summary.recomputing_group_count}"
        f" of {summary.flexible_group_count} groups"
        f" ({summary.recomputing_group_fraction:.0%})"
        + (f", with {forced} more forced" if forced else "")
    )
    chosen = summary.selected_candidate
    if "incumbent" in chosen:
        # the plan in hand won: say where it came from
        in_hand = chosen["incumbent"]
        print(
            f"  chosen plan        the plan in hand, found by {in_hand['found_by']}"
            f" at {gib(in_hand['found_at_capacity_bytes'])}"
        )
    elif chosen:
        repairs = chosen["repairs_at_best"]
        print(
            f"  chosen candidate   {chosen['residency_strategy']} /"
            f" {chosen['fetch_rule']}"
            f"{' / coalesced' if chosen['coalesced'] else ''}"
            f"   {repairs} repairs to its plan"
            if repairs is not None
            else ""
        )
        unplaced = chosen.get("best_unplaced_makespan_ns")
        if unplaced:
            # what placing cost this candidate: its answer over the fastest
            # plan it simulated but could not place
            print(
                f"  placement gap      {chosen['placement_gap']:.3f}x over"
                f" {unplaced / 1e9:.3f} s, the fastest of"
                f" {chosen['unplaced_plans']} plans whose layout missed"
            )
    print()
    print(
        f"  traffic per step   fetch {gib(report.transfer_bytes_fetched)}"
        f"   evict {gib(report.transfer_bytes_evicted)}"
        + (
            f"   spill peak {gib(summary.spill_peak_bytes)}"
            if summary.spill_peak_bytes
            else ""
        )
    )
    print(
        f"  planning capacity  execution {gib(report.execution_budget_bytes)}"
        f"   spill {gib(report.spill_budget_bytes)}"
    )
    # "assumed" is the rate the plan was priced against, which the summary
    # carries: a measured rate is coarsened before it reaches the simulator, so
    # it is not the profile's own figure. The profile holds what was measured.
    fetch, evict = report.fetch_profile, report.evict_profile
    for name, planned_rate, planned_latency_ns, profile in (
        (
            "fetch",
            summary.fetch_bandwidth_bytes_per_second,
            summary.fetch_latency_ns,
            fetch,
        ),
        (
            "evict",
            summary.evict_bandwidth_bytes_per_second,
            summary.evict_latency_ns,
            evict,
        ),
    ):
        print(
            f"  {name} lane         {gb_s(planned_rate)} assumed,"
            f" latency {planned_latency_ns / 1e3:.0f} us"
            f"   (measured {gb_s(profile.bandwidth_bytes_per_second)} effective,"
            f" {gb_s(profile.solo_bandwidth_bytes_per_second)} solo,"
            f" latency {profile.latency_nanoseconds / 1e3:.1f} us)"
        )
    print()


def _task_duration_delta(record: TaskRecord) -> float:
    """How much longer the task's kernels ran than the profile priced."""

    measured = record.compute_finished_at_seconds - record.compute_started_at_seconds
    return measured - record.expected_profile_seconds


def _transfer_duration_delta(record: TransferRecord) -> float:
    """How much longer the copy held its lane than the simulator priced."""

    if (
        record.lane_started_at_seconds is None
        or record.lane_finished_at_seconds is None
        or record.simulated_started_at_seconds is None
        or record.simulated_finished_at_seconds is None
    ):
        return 0.0
    measured = record.lane_finished_at_seconds - record.lane_started_at_seconds
    simulated = (
        record.simulated_finished_at_seconds - record.simulated_started_at_seconds
    )
    return measured - simulated


def print_epilogue(diagnostics: Any) -> None:
    summary = diagnostics.summary
    timelines = diagnostics.timelines
    simulated_step = summary.simulator_makespan_seconds
    real_step = summary.real_invocation_seconds
    if real_step is None:
        print("  traced invocation: incomplete transfer timestamps")
    else:
        print(
            f"  traced invocation  real {real_step:.3f} s"
            f"   simulated {simulated_step:.3f} s"
            f"   ({(real_step - simulated_step) / simulated_step:+.2%})"
        )
    for label, real, simulated in (
        (
            "entry delay",
            summary.entry_delay_seconds,
            summary.simulated_entry_delay_seconds,
        ),
        (
            "task window",
            summary.real_selected_span_seconds,
            summary.simulated_selected_span_seconds,
        ),
        (
            "terminal tail",
            summary.real_terminal_tail_seconds,
            summary.simulator_terminal_tail_seconds,
        ),
    ):
        measured_text = "unavailable" if real is None else f"{real:.3f} s"
        print(f"    {label:14} real {measured_text}   simulated {simulated:.3f} s")
    if summary.cycle_seconds is not None:
        print(
            f"  whole cycle        {summary.cycle_seconds:.3f} s"
            " (includes caller and instrumentation work)"
        )
    print(
        "  stalled          real"
        f" {summary.real_inter_task_readiness_wait_seconds:.3f} s"
        f"   simulated {summary.simulated_inter_task_readiness_wait_seconds:.3f} s"
    )
    print(
        "  first task       waited"
        f" {summary.real_initial_readiness_wait_seconds * 1e3:.1f} ms"
        " for its own inputs, inside the entry delay above"
    )
    compute = [diagnostics.tasks[task_id] for task_id in timelines.compute]
    worst = max(compute, key=lambda item: abs(_task_duration_delta(item)))
    print(f"  compute lane ({len(compute)} tasks), real minus simulated")
    print(deltas("starts   ", [item.start_delta_seconds for item in compute]))
    print(
        deltas(
            "durations",
            [_task_duration_delta(item) for item in compute],
            worst_key=worst.execution_task_id,
        )
    )
    for lane in (timelines.fetch, timelines.evict):
        lane_summary = lane.summary
        measured = [
            record
            for record in (
                getattr(diagnostics.transfers, lane.summary.direction)[key]
                for key in lane.order
            )
            if record.start_delta_seconds is not None
        ]
        effective = lane_summary.effective_bandwidth_bytes_per_second
        print(
            f"  {lane_summary.direction} lane ({lane_summary.transfers} transfers,"
            f" {lane_summary.bytes / 2**30:.1f} GiB), real minus simulated;"
            f" lane busy real {lane_summary.lane_busy_seconds:.3f} s"
            f" simulated {lane_summary.simulated_busy_seconds:.3f} s"
            + (f"; effective {gb_s(effective)}" if effective is not None else "")
        )
        if measured:
            print(
                deltas(
                    "starts   ", [item.start_delta_seconds or 0.0 for item in measured]
                )
            )
            print(
                deltas(
                    "durations",
                    [_transfer_duration_delta(item) for item in measured],
                )
            )
    print()


_DTYPE_NAMES = ("bfloat16", "float16", "float32")
_ROUNDINGS = ("nearest", "stochastic")


def _dtype_name(value: str) -> str | None:
    """A floating dtype by its torch name, or ``none`` for the default."""

    lowered = value.strip().lower().removeprefix("torch.")
    if lowered == "none":
        return None
    if lowered not in _DTYPE_NAMES:
        raise argparse.ArgumentTypeError(
            f"a dtype is one of {', '.join(_DTYPE_NAMES)}, or none; not {value!r}"
        )
    return lowered


@dataclass(frozen=True)
class Precision:
    """The dtypes and roundings a step trains under, named as the training
    harness names them.

    ``master_dtype``, ``grad_dtype`` and ``round_accumulation_once`` are
    ``plan_step``'s. The rest are the optimizer's own, given to it when it is
    built: the dtype it keeps its state at and how it rounds what it stores
    and the weights it steps. A gradient dtype reaches two more places on its
    own, because a step that names one and sets neither silently rounds: the
    mlops kernels are asked for weight gradients at it, and the optimizer
    reads gradients at it. ``None`` everywhere is the library's and the
    optimizer's defaults, which is what a request that names nothing gets.
    """

    master_dtype: str | None = None
    grad_dtype: str | None = None
    opt_state_dtype: str | None = None
    parameter_rounding: str | None = None
    opt_state_rounding: str | None = None
    round_accumulation_once: bool = False

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> Precision:
        return cls(
            master_dtype=arguments.master_dtype,
            grad_dtype=arguments.grad_dtype,
            opt_state_dtype=arguments.opt_state_dtype,
            parameter_rounding=arguments.parameter_rounding,
            opt_state_rounding=arguments.opt_state_rounding,
            round_accumulation_once=bool(arguments.round_accumulation_once),
        )

    @staticmethod
    def _dtype(name: str | None) -> torch.dtype | None:
        return None if name is None else cast(torch.dtype, getattr(torch, name))

    @property
    def master(self) -> torch.dtype | None:
        """``plan_step``'s ``master_dtype``."""

        return self._dtype(self.master_dtype)

    @property
    def gradients(self) -> torch.dtype | None:
        """``plan_step``'s ``grad_dtype``."""

        return self._dtype(self.grad_dtype)

    def plan_arguments(self) -> dict[str, object]:
        """What planning is told: the same three keywords the harness passes."""

        return {
            "master_dtype": self.master,
            "grad_dtype": self.gradients,
            "round_accumulation_once": self.round_accumulation_once,
        }

    def optimizer_arguments(self) -> dict[str, object]:
        """What the optimizer is built with, beyond its defaults."""

        given: dict[str, object] = {}
        if self.grad_dtype is not None:
            given["gradient_dtype"] = self._dtype(self.grad_dtype)
        if self.opt_state_dtype is not None:
            given["opt_state_dtype"] = (
                "parameter"
                if self.opt_state_dtype == "parameter"
                else self._dtype(self.opt_state_dtype)
            )
        if self.parameter_rounding is not None:
            given["parameter_rounding"] = self.parameter_rounding
        if self.opt_state_rounding is not None:
            given["opt_state_rounding"] = self.opt_state_rounding
        return given

    def optimizer(self, base: Any) -> Any:
        """``base`` built with these settings, or ``base`` itself when it
        names none, so a request that names nothing plans exactly as before."""

        given = self.optimizer_arguments()
        return functools.partial(base, **given) if given else base

    def apply(self) -> None:
        """Ask the mlops kernels for weight gradients at the gradient dtype,
        so what the step sums comes back unrounded."""

        if self.grad_dtype is not None:
            set_weight_gradient_dtype(self._dtype(self.grad_dtype))

    def lines(self) -> tuple[tuple[str, str, str], ...]:
        """The banner's rows: a label, the value, and what it means."""

        gradients = self.grad_dtype or "the weights'"
        return (
            (
                "master dtype",
                self.master_dtype or "none",
                "the optimizer steps the weights themselves"
                if self.master_dtype is None
                else "a master copy of every weight trained at another dtype",
            ),
            (
                "grad dtype",
                self.grad_dtype or "weights'",
                f"gradients summed over the microbatches at {gradients} dtype;"
                + (
                    " a multiply adds its product in as it writes, rounded once"
                    if self.round_accumulation_once
                    else " a product rounded to a narrower gradient is added after"
                ),
            ),
            (
                "opt state dtype",
                self.opt_state_dtype or "default",
                "the moments at the optimizer's default dtype"
                if self.opt_state_dtype is None
                else "the moments at that dtype ('parameter': what it steps)",
            ),
            (
                "parameter rounding",
                self.parameter_rounding or "default",
                "how the optimizer rounds the weights it steps; its default"
                " is to nearest",
            ),
            (
                "opt state rounding",
                self.opt_state_rounding or "default",
                "how it rounds the state it stores; its default is to nearest",
            ),
        )


_REQUEST_SCHEMA = artifact_schema("quickstart_request")

#: What a request is made of: every argument except the ones that say where
#: this run writes or what it draws, which a reproduction chooses for itself.
_NOT_REQUEST = frozenset(
    {
        "reproduce",
        "output_dir",
        "force_overwrite",
        "plots",
        "timelines",
        "resolution_plans",
    }
)
_REPRODUCE_MAY_TAKE = frozenset(
    {
        "--reproduce",
        "--output-dir",
        "--plots",
        "--timelines",
        "--no-timelines",
        "--resolution-plans",
        "--no-resolution-plans",
    }
)


def _request_record(
    arguments: argparse.Namespace,
    store: Path,
    build_store: Path | None,
    plan_store: Path,
) -> dict[str, object]:
    """The request as given, with the stores resolved, for `--reproduce`."""

    request: dict[str, object] = {}
    for name, value in vars(arguments).items():
        if name in _NOT_REQUEST:
            continue
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, TransferBandwidths):
            value = value.to_dict()
        elif isinstance(value, tuple):
            value = list(value)
        request[name] = value
    request["artifact_store"] = str(store)
    request["build_store"] = None if build_store is None else str(build_store)
    request["plan_store"] = str(plan_store)
    return {
        "schema": _REQUEST_SCHEMA,
        "command": sys.argv[1:],
        "revision": _revision(),
        "started_at": _started_at(),
        "request": request,
    }


def _reproduced_arguments(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> argparse.Namespace:
    """The request a run recorded, pinned to its calibration, refusing a miss."""

    given = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
    foreign = sorted(given - _REPRODUCE_MAY_TAKE)
    if foreign or arguments.model is not None:
        named = [*foreign, *([arguments.model] if arguments.model else [])]
        parser.error(
            "--reproduce takes the whole request from the run; "
            f"{', '.join(named)} would contradict it"
        )
    run = arguments.reproduce
    record_path = run / "request.json"
    report_path = run / "search.json"
    for path in (record_path, report_path):
        if not path.is_file():
            parser.error(f"--reproduce: {path} is missing")
    try:
        record = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"--reproduce: {record_path}: {error}")
    if not isinstance(record, dict) or record.get("schema") != _REQUEST_SCHEMA:
        parser.error(f"--reproduce: {record_path} is not a quickstart request")
    request = record.get("request")
    if not isinstance(request, dict):
        parser.error(f"--reproduce: {record_path} records no request")
    for name, value in request.items():
        if name in ("artifact_store", "build_store", "plan_store"):
            value = None if value is None else Path(value)
        elif name == "resolution_options":
            value = tuple(value)
        setattr(arguments, name, value)
    arguments.transfer_bandwidths = _transfer_bandwidths(str(report_path))
    arguments.plan_store_mode = "require"
    return arguments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=IDENTITIES,
        help="which workload to run; taken from the run when --reproduce is given",
    )
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--sequences-per-step", type=int)
    parser.add_argument(
        "--sequences-per-microbatch",
        type=int,
        help="choose the geometry yourself and skip the search; must divide"
        " the sequences per step",
    )
    parser.add_argument("--min-tokens-per-microbatch", type=int)
    parser.add_argument("--max-tokens-per-microbatch", type=int)
    parser.add_argument(
        "--search-budget-gib",
        type=_budget_list,
        help="comma-separated execution budgets to search and plot across,"
        " for example 10,12,16",
    )
    parser.add_argument(
        "--run-budget-gib",
        type=_budget_list,
        help="comma-separated execution budgets to actually run steps at."
        " Every run budget must appear among the search budgets. With"
        " neither flag the retained budget is searched and run; with only"
        " search budgets, nothing executes",
    )
    parser.add_argument("--spill-gib", type=float)
    parser.add_argument(
        "--remote-spill",
        metavar="HOST:PORT",
        help=(
            "spill to a memory daemon on another machine instead of to pinned "
            "host memory. The pool is the same size either way -- --spill-gib "
            "still sets it -- so the only thing that differs is where it lives"
        ),
    )
    parser.add_argument(
        "--orderings",
        choices=("factors", "depth-first"),
        default="factors",
        help="which microbatch orderings the search tries per geometry:"
        " every depth x breadth factor pair (the default), or only the"
        " depth-first walk. The loss stays paired and the backward walk"
        " reversed either way; the search does not toggle those",
    )
    parser.add_argument(
        "--resolution-options",
        type=named_resolution_options,
        default=NAMED_RESOLUTION_OPTIONS["quarters"],
        help="which resolutions the search and the runs plan: the shares of"
        " flexible groups to recompute, as 'quarters' (the library"
        " default), 'eighths', 'halves', or a comma-separated list of exact"
        " fractions such as 0,1/2,7/8,1. More shares plan more programs per"
        " point; on the llama3 frontier eighths cost 1.75x the search for a"
        " median gain of nothing",
    )
    parser.add_argument(
        "--transfer-bandwidths",
        type=_transfer_bandwidths,
        default=None,
        help="plan against this calibration instead of the one the runtime"
        " measures at start: FETCH,EVICT in GB/s, optionally followed by the"
        " fetch and evict latencies in microseconds, or the path of another"
        " run's search.json to pin to what that run planned against. The run"
        " phase plans against the same lanes as the search either way",
    )
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--timelines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write every plan's pages under timelines/ as the run closes: the"
        " pools and the lanes over the step, on the simulated clock for every"
        " plan the search made and on the device's too for every budget that"
        " ran; --no-timelines skips them",
    )
    parser.add_argument(
        "--resolution-plans",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep every resolution's best plan in the plan store beside the"
        " answer, certified, so the timelines carry a page per resolution;"
        " several times the plan store, so off by default",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="replace an existing run at the output directory. Its artifact"
        " store is kept, being a content-addressed cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where this run's search report, log, traced steps and figures"
        " are written; defaults to benchmarking/quickstart_reports/"
        "<model>_<revision>_<MMDD_HHMM>/seq<length>/seqsperstep<n>",
    )
    parser.add_argument(
        "--reproduce",
        type=Path,
        default=None,
        metavar="RUN",
        help="repeat the run at RUN (its seq<length>/seqsperstep<n> directory)"
        " exactly: every setting is read from its request.json, the search is"
        " pinned to the calibration its search.json records, and plan-store"
        " mode is require, so a plan the store lacks refuses instead of being"
        " searched again. Only --output-dir, --plots, --timelines and"
        " --resolution-plans may be given with it",
    )
    parser.add_argument(
        "--export-bypass-key",
        default=None,
        help="the caller's name for the code this run builds from; with it, a"
        " build reads each ordering's step program back from the build store"
        " and captures only what is not there. Without it every build captures",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--master-dtype",
        type=_dtype_name,
        default=None,
        metavar="DTYPE",
        help="keep a master copy of every weight trained at another dtype at"
        " this one -- float32, say -- and step the masters in the weights'"
        " place; none, the default, steps the weights themselves",
    )
    parser.add_argument(
        "--grad-dtype",
        type=_dtype_name,
        default=None,
        metavar="DTYPE",
        help="the dtype gradients are created and summed at over a step's"
        " microbatches, the weights' own by default. Naming one also asks"
        " the mlops kernels for weight gradients at it and has the optimizer"
        " read gradients at it, so nothing rounds them on the way",
    )
    parser.add_argument(
        "--opt-state-dtype",
        choices=(*_DTYPE_NAMES, "parameter"),
        default=None,
        help="the dtype the optimizer keeps its state at: a dtype, or"
        " parameter for the dtype of what it steps. Its own default when"
        " not given",
    )
    parser.add_argument(
        "--parameter-rounding",
        choices=_ROUNDINGS,
        default=None,
        help="how the optimizer rounds the weights it steps: to nearest, its"
        " default, or stochastically, which keeps small updates in expectation",
    )
    parser.add_argument(
        "--opt-state-rounding",
        choices=_ROUNDINGS,
        default=None,
        help="how the optimizer rounds the state it stores, the same two ways",
    )
    parser.add_argument(
        "--round-accumulation-once",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="let a matrix multiply add its product into running gradients"
        " kept narrower than it sums at as it writes them, rounding the sum"
        " once instead of twice; off by default, which is what PyTorch's"
        " own step computes",
    )
    parser.add_argument(
        "--artifact-store",
        type=Path,
        default=None,
        help="roots both stores under one directory. Defaults to"
        " <output-dir>/artifact_store",
    )
    parser.add_argument(
        "--build-store",
        type=Path,
        default=None,
        help="the captures, graph pairs, profiles and compiled artifacts to"
        " read and write; point it at another run's store to skip work"
        " already paid for there. Overrides --artifact-store for the build"
        " tree",
    )
    parser.add_argument(
        "--plan-store",
        type=Path,
        default=None,
        help="where this run's plans go: every request, result and plan"
        " manifest, kept apart from the build store so a shared store never"
        " hands a run another run's plans. Overrides --artifact-store for the"
        " planning tree. Defaults to <output-dir>/plan_store",
    )
    for tree in ("build", "plan"):
        parser.add_argument(
            f"--{tree}-store-mode",
            choices=STORE_MODES,
            default="contribute",
            help=f"what this run may do about a {tree} artifact the store does"
            " not hold: contribute builds it and writes it back, reuse builds"
            " it and persists nothing, require refuses and names it",
        )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="make the search reproduce exactly at any worker count: a"
        " candidate's placement gate consults only its own placed plans"
        " rather than the shared best-placed record, so every graph-pair"
        " selection reports the plan it actually found. On by default,"
        " because the figures compare selections; --no-deterministic lets"
        " the shared bound skip measuring plans that cannot win, at the"
        " cost of selections that show up or not depending on timing",
    )
    parser.add_argument(
        "--incumbents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hand each budget the best plan found at a smaller budget of the"
        " same program as the plan to beat, so no program plans worse with"
        " more memory; a point that did not beat it answers with it and says"
        " which budget it came from. --no-incumbents searches every point"
        " alone, for comparing the two",
    )
    profiling = parser.add_argument_group("Task profiling")
    defaults = ProfilingOptions()
    for option in fields(ProfilingOptions):
        default = getattr(defaults, option.name)
        profiling.add_argument(
            "--profile-" + option.name.replace("_", "-"),
            dest="profile_" + option.name,
            type=type(default),
            default=default,
            help=f"{option.name.replace('_', ' ')} (default: {default}); "
            "duration targets use exact-task device time, wall limits use host time",
        )
    return parser


def parse_arguments() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    """The flags, with the choices a flag alone can settle already settled."""

    parser = _parser()
    arguments = parser.parse_args()
    if arguments.reproduce is not None:
        arguments = _reproduced_arguments(parser, arguments)
    elif arguments.model is None:
        parser.error("a model is required unless --reproduce names a run")
    # An unspecified placement is the library's to choose. Resolving it here
    # rather than defaulting the flag keeps one answer to the question: a change
    # to the library default reaches this tour, and the banner reports what the
    # planner will actually do rather than what this file last believed.
    if arguments.steps < 1:
        parser.error("--steps must be at least 1")
    try:
        profiling_policy(arguments)
        validate_resolution_options(arguments.resolution_options)
    except ValueError as error:
        parser.error(str(error))

    return parser, arguments


@dataclass(frozen=True)
class Request:
    """What the tour was asked to do, resolved from the flags."""

    manifest: Any
    search_budgets: list[int]
    run_budgets: list[int]
    physical_capacity: int
    sequence_length: int
    sequences_per_step: int
    tokens_per_step: int
    manual: int | None
    #: Where the spill pool lives, when it is not this machine's pinned host
    #: memory. ``None`` is the local tour.
    remote_spill: tuple[str, int] | None = None


def _remote_peer(
    parser: argparse.ArgumentParser, value: str | None
) -> tuple[str, int] | None:
    """The daemon named by ``--remote-spill``, or ``None`` for pinned host."""

    if value is None:
        return None
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port.isdigit():
        parser.error(f"--remote-spill must read HOST:PORT, not {value!r}")
    return host, int(port)


def resolve_request(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> Request:
    """The manifest and the budgets, or a parser error naming the flag."""

    implementation, family = arguments.model.split("_", 1)
    manifest = manifest_for(family, cast(ModelImplementation, implementation))
    manifest = replace(
        manifest,
        sequence_length=arguments.sequence_length or manifest.sequence_length,
        spill_budget_bytes=(
            int(arguments.spill_gib * _GIB)
            if arguments.spill_gib
            else manifest.spill_budget_bytes
        ),
    )
    run_budgets = [int(value * _GIB) for value in (arguments.run_budget_gib or [])]
    search_budgets = [
        int(value * _GIB) for value in (arguments.search_budget_gib or [])
    ]
    if not run_budgets and not search_budgets:
        run_budgets = [manifest.device_physical_capacity_bytes]
        search_budgets = list(run_budgets)
    elif not search_budgets:
        search_budgets = list(run_budgets)
    else:
        outside = [item for item in run_budgets if item not in search_budgets]
        if outside:
            parser.error(
                "every --run-budget-gib must appear among the search"
                f" budgets; {', '.join(gib(item) for item in outside)}"
                " does not"
            )
    search_budgets.sort()
    # The largest budget is the process's device-memory cap, not its slab: a pool's
    # physical capacity covers the accelerator problem and the provider headroom as
    # well as the suballocatable slab. Sizing the pool above it would let the process
    # exceed the budget the caller asked for.
    physical_capacity = max(search_budgets)
    sequence_length = manifest.sequence_length
    sequences_per_step = arguments.sequences_per_step or (
        manifest.sequences_per_microbatch * manifest.accumulation_count
    )
    tokens_per_step = sequence_length * sequences_per_step
    manual = arguments.sequences_per_microbatch
    if manual is not None and (manual < 1 or sequences_per_step % manual):
        parser.error(
            f"--sequences-per-microbatch {manual} does not divide"
            f" {sequences_per_step} sequences per step"
        )
    if manual is not None and not run_budgets:
        parser.error(
            "--sequences-per-microbatch chooses a geometry to run; give at"
            " least one --run-budget-gib"
        )
    # The objective normalizes each microbatch by this whole-step total.
    # Search chooses execution geometry later; the manifest must already
    # describe the requested step rather than the model family's default.
    template_sequences = manual or sequences_per_step
    manifest = replace(
        manifest,
        sequences_per_microbatch=template_sequences,
        accumulation_count=sequences_per_step // template_sequences,
    )
    return Request(
        manifest=manifest,
        search_budgets=search_budgets,
        run_budgets=run_budgets,
        physical_capacity=physical_capacity,
        sequence_length=sequence_length,
        sequences_per_step=sequences_per_step,
        tokens_per_step=tokens_per_step,
        manual=manual,
        remote_spill=_remote_peer(parser, arguments.remote_spill),
    )


def _head_loss_share(metrics: Any) -> float | None:
    """The microbatch's head loss, when the objective reports it separately."""

    if isinstance(metrics, Mapping) and HEAD_LOSS_METRIC in metrics:
        return float(metrics[HEAD_LOSS_METRIC])
    return None


@dataclass(frozen=True)
class RunPaths:
    """Where one run writes, and the stores it reads and writes."""

    root: Path
    store: Path
    build_store: Path | None
    plan_store: Path


def prepare_run_root(
    arguments: argparse.Namespace, request: Request
) -> RunPaths | None:
    """Claim the run directory, record the request, copy the console into it.

    `None` when the directory already holds a run and `--force-overwrite` was
    not given; the refusal has been printed.
    """

    # Everything a run leaves behind lands together: the search report, its
    # log, and one step trace per run budget.
    # One directory per run: the model, the revision it measured and when it
    # started, then sequence length, then sequences per step, so each level is
    # exactly one parameter and another run is a sibling rather than an
    # overwrite. The start time is what keeps two runs of one revision apart.
    run_root = arguments.output_dir or (
        Path("benchmarking/quickstart_reports")
        / f"{arguments.model}_{_revision()}_{_started_at()}"
        / f"seq{request.sequence_length}"
        / f"seqsperstep{request.sequences_per_step}"
    )
    # A run directory is written once, and silently replacing one loses a
    # measurement that cost real time. The default path carries the start
    # minute, so this guards an explicit --output-dir and the two runs that
    # begin within the same minute. Refuse, and say both ways out.
    written = tuple(
        name
        for name in ("search.json", "progress.log", "steps", "figures", "timelines")
        if (run_root / name).exists()
    )
    if written and not arguments.force_overwrite:
        print(
            f"  {run_root} already holds a run ({', '.join(written)}).\n"
            "  Pass --force-overwrite to replace it, or --output-dir to write"
            " somewhere else.",
            file=sys.stderr,
        )
        return None
    for name in written:
        target = run_root / name
        # The artifact store is deliberately not cleared: it is a
        # content-addressed cache, so stale entries are unreachable rather
        # than wrong, and rebuilding it costs capture, compilation and
        # profiling over again.
        shutil.rmtree(target) if target.is_dir() else target.unlink()

    # A run owns both stores by default, so what it measured is self-contained
    # and nothing it reused is ambiguous. Point `--artifact-store` at
    # another run's store, or at a shared one, to skip capture, compilation
    # and profiling that has already been paid for elsewhere; the plans stay
    # this run's own either way, so a shared store never answers a point
    # with a plan another run searched.
    store = arguments.artifact_store or (run_root / "artifact_store")
    build_store = arguments.build_store
    plan_store = arguments.plan_store or (run_root / "plan_store")

    # Everything printed is also kept beside the run it describes. The
    # progress log records what the search and the runtime were doing at each
    # moment; this is the report a person actually read -- the geometry table,
    # the chosen plans, the per-step numbers -- and a run whose console has
    # scrolled away is a measurement that has to be taken again to be read.
    run_root.mkdir(parents=True, exist_ok=True)
    # The request in full, so a later run can repeat it without the command.
    (run_root / "request.json").write_text(
        json.dumps(
            _request_record(arguments, store, build_store, plan_store),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    console = (run_root / "console.log").open("w", encoding="utf-8")
    stdout = sys.stdout

    class _Tee:
        """Write to the terminal and to the run's own copy."""

        def write(self, text: str) -> int:
            console.write(text)
            # Flushed as it goes, so the copy is readable while the run is
            # still going and survives a run that is killed rather than ended.
            console.flush()
            return stdout.write(text)

        def flush(self) -> None:
            console.flush()
            stdout.flush()

        def __getattr__(self, name: str) -> object:
            return getattr(stdout, name)

    sys.stdout = _Tee()
    # However the run ends -- finished, interrupted, or failed -- the copy is
    # closed and the terminal is handed back, so a partial run still leaves a
    # readable record of how far it got.

    def restore_console() -> None:
        sys.stdout = stdout
        console.close()

    atexit.register(restore_console)

    return RunPaths(run_root, store, build_store, plan_store)


class Ledger(dict[str, float]):
    """Where the wall clock went, by category, for the closing table."""

    def __init__(self) -> None:
        super().__init__()
        self.started = time.perf_counter()

    def charge(self, category: str, started: float) -> None:
        self[category] = self.get(category, 0.0) + (time.perf_counter() - started)


def open_runtime(request: Request, ledger: Ledger) -> Runtime:
    """The runtime, calibrated once here and reused by every geometry."""

    marker = time.perf_counter()
    capacity = request.manifest.spill_budget_bytes
    if request.remote_spill is None:
        spill: Any = pinned_host(capacity=capacity)
    else:
        # Imported here: a local tour should not load the network library to
        # decide it does not need it.
        from shadowspill.network import remote

        host, port = request.remote_spill
        spill = remote(capacity=capacity, host=host, port=port)
        print(f"spilling to {host}:{port}, {capacity >> 30} GiB")
    runtime = Runtime(
        pools={
            "execution": device(physical_capacity=request.physical_capacity),
            # The same size either way, so the only thing that differs is where
            # it lives -- which is what makes the two tours comparable.
            "spill": spill,
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
    )
    ledger.charge("runtime construction and calibration", marker)
    return runtime


@dataclass(frozen=True)
class Budgets:
    """The budgets as requested and as the pool resolves them."""

    requested_search: list[int]
    requested_run: list[int]
    planned: dict[int, int]

    @property
    def search(self) -> list[int]:
        """Each search request's slab, once each.

        Two requests can resolve to one slab once they reach the pool's
        capacity, and planning the same budget twice would search it twice.
        """

        return list(dict.fromkeys(self.planned[item] for item in self.requested_search))

    @property
    def run(self) -> list[int]:
        return list(dict.fromkeys(self.planned[item] for item in self.requested_run))

    def asked(self, values: list[int]) -> str:
        """Each request, naming the slab it resolved to wherever that is smaller."""

        return ", ".join(
            gib(item)
            if self.planned[item] == item
            else f"{gib(item)}->{gib(self.planned[item])}"
            for item in values
        )


def plan_budgets(runtime: Runtime, request: Request) -> Budgets:
    """Resolve every requested budget against the pool it will run in.

    A requested budget is the process's device-memory cap; the slab a plan may
    fill is what is left after the accelerator problem and the provider
    headroom. Planning resolves that, so both phases are given the resolved
    figure -- the search used to take its budgets literally and rank against a
    slab the cap cannot hold, which made it promise plans the run could not
    reproduce.
    """

    execution_pool = runtime.pools["execution"]
    return Budgets(
        requested_search=request.search_budgets,
        requested_run=request.run_budgets,
        planned={
            item: resolve_execution_budget(item, execution_pool)
            for item in dict.fromkeys([*request.search_budgets, *request.run_budgets])
        },
    )


def print_banner(
    arguments: argparse.Namespace, request: Request, runtime: Runtime, budgets: Budgets
) -> None:
    """The run's settings, and the lanes as the simulator will be built with them."""

    manifest = request.manifest
    sequence_length = request.sequence_length
    sequences_per_step = request.sequences_per_step
    tokens_per_step = request.tokens_per_step
    execution_pool = runtime.pools["execution"]
    print("═" * 68)
    print(f"  ShadowSpill quickstart — {arguments.model}")
    print("═" * 68)
    searched = budgets.asked(budgets.requested_search)
    ran = budgets.asked(budgets.requested_run) or "none (search only)"
    print(
        f"  sequence length     {sequence_length:>10,}      search budgets   {searched}"
    )
    print(
        f"  sequences per step  {sequences_per_step:>10,}      run budgets      {ran}"
    )
    print(
        f"  tokens per step     {tokens_per_step:>10,}"
        f"      spill budget     {gib(manifest.spill_budget_bytes)}"
    )
    print(
        f"  execution pool      {gib(execution_pool.physical_capacity or 0):>10}"
        f"      suballocatable   {gib(execution_pool.capacity)}"
        f"   (runtime init carved"
        f" {gib((execution_pool.physical_capacity or 0) - execution_pool.capacity)})"
    )

    admitted, skipped = search_geometries(
        sequences_per_step,
        sequence_length=sequence_length,
        min_tokens_per_microbatch=arguments.min_tokens_per_microbatch,
        max_tokens_per_microbatch=arguments.max_tokens_per_microbatch,
    )
    print(
        f"  geometries          {len(admitted):>10}      "
        + ", ".join(f"{item[0]}x{item[1]}" for item in admitted)
        + (f"   ({len(skipped)} skipped by the token bounds)" if skipped else "")
    )
    print(
        f"  orderings           {arguments.orderings:>10}      "
        + (
            "every depth x breadth factor pair"
            if arguments.orderings == "factors"
            else "the depth-first walk only"
        )
        + "; loss paired, backward reversed"
    )
    shares = arguments.resolution_options
    print(
        "  resolutions         "
        + f"{len(shares):>10}      "
        + ", ".join(str(item) for item in shares)
        + " of the flexible groups recomputing"
    )
    print("  initial state            fresh      plan-owned values begin in spill")
    # The precision a step trains under is not recoverable from its figures,
    # and two runs at different precisions are not the same arithmetic, so
    # the banner names it even when every setting is the default.
    for label, value, meaning in Precision.from_arguments(arguments).lines():
        print(f"  {label:<20}{value:>10}      {meaning}")

    # One calibration serves every geometry, so it is a property of the run.
    # The planned figures come from the planner rather than from rounding a
    # measurement here, so this banner says what the simulator will actually be
    # built with. Effective is the measured rate planning coarsens, concurrent
    # is what a copy gets against other traffic, solo is what it gets alone.
    pinned = arguments.transfer_bandwidths
    if pinned is not None:
        print(
            "  transfer lanes      "
            f"pinned to fetch {gb_s(pinned.fetch_bytes_per_second)},"
            f" evict {gb_s(pinned.evict_bytes_per_second)}"
            + (
                f", latency {pinned.fetch_latency_ns / 1e3:.0f}/"
                f"{pinned.evict_latency_ns / 1e3:.0f} us"
                if pinned.fetch_latency_ns is not None
                and pinned.evict_latency_ns is not None
                else ""
            )
        )
    else:
        capabilities = runtime.transfer_capabilities
        planned = planned_transfer_bandwidths(
            capabilities.route("spill", "execution"),
            capabilities.route("execution", "spill"),
        )
        # This producer always names both latencies; only an override naming
        # bandwidths alone leaves them unset.
        for name, source, destination, rate, latency_ns in (
            (
                "fetch",
                "spill",
                "execution",
                planned.fetch_bytes_per_second,
                planned.fetch_latency_ns or 0,
            ),
            (
                "evict",
                "execution",
                "spill",
                planned.evict_bytes_per_second,
                planned.evict_latency_ns or 0,
            ),
        ):
            profile = capabilities.route(source, destination)
            print(
                f"  {name + ' lane':<19} "
                f"{gb_s(rate)} planned, latency {latency_ns / 1e3:.0f} us"
                f"   (effective {gb_s(profile.bandwidth_bytes_per_second)},"
                f" concurrent {gb_s(profile.concurrent_bandwidth_bytes_per_second)},"
                f" solo {gb_s(profile.solo_bandwidth_bytes_per_second)},"
                f" latency {profile.latency_nanoseconds / 1e3:.1f} us)"
            )
    print()

    note_host_memory(None, "runtime pools registered")


@dataclass(frozen=True)
class _Steps:
    """What one budget's steps measured, once the last result is released."""

    diagnostics: Any
    walls: tuple[float, ...]
    median_step_seconds: float
    simulated_step_seconds: float
    host_seconds: float


class Tour:
    """One quickstart run: the model in the spill pool, the search, then each budget.

    Holds what every phase reads -- the request, the run's paths, the runtime,
    the ledger, the progress log -- and what the run phase changes: the case,
    rebuilt between budgets so each starts from the same weights.
    """

    def __init__(
        self,
        arguments: argparse.Namespace,
        request: Request,
        paths: RunPaths,
        budgets: Budgets,
        runtime: Runtime,
        ledger: Ledger,
    ) -> None:
        self.arguments = arguments
        self.request = request
        self.paths = paths
        self.budgets = budgets
        self.runtime = runtime
        self.ledger = ledger
        manifest = request.manifest
        marker = time.perf_counter()
        # Built inside the pool it will live in, so the parameters are written
        # Declared on meta and materialised straight into the spill pool, so the
        # model's values are written where they will live and no host memory
        # proportional to it is ever allocated.
        case = build_case(manifest, seed=arguments.seed, runtime=runtime)
        ledger.charge("model construction in the spill pool", marker)
        note_host_memory(None, "model constructed in the spill pool")
        self.case = case
        self.vocabulary = int(manifest.model_config.vocab_size)
        # Built once and handed to both planning phases; see `search_policy`.
        self.policy = search_policy(arguments)
        self.profiling = profiling_policy(arguments)
        # Likewise the precision: the optimizer both phases build, the dtypes
        # both phases plan with, and the gradient dtype the kernels are asked
        # for, applied once here before anything is captured.
        self.precision = Precision.from_arguments(arguments)
        self.precision.apply()
        self.optimizer = self.precision.optimizer(case.optimizer)
        self.trained = False
        self.report: StepSearchReport | None = None
        # Opened before the branch: a run that chose its geometry by hand still
        # plans once per budget, and that is the same granular output a search
        # produces. Only the search is optional; the log is not.
        progress_log = paths.root / "progress.log"
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = progress_log.open("w")
        self.plan_log = PlanLog(self.log_handle, sys.stdout)
        print(f"  progress log: {progress_log}   (tail -f it to follow)")

    def example_microbatches(
        self,
        sequences: int,
        accumulation: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[tuple[object, ...], ...]:
        sequence_length = self.request.sequence_length
        per_microbatch = sequences * sequence_length
        lengths = (sequence_length,) * sequences
        # One draw for the whole step, then split: the tokens and targets a
        # step trains on are then the same at every geometry, so two
        # geometries' losses differ only in what was computed. Drawing each
        # microbatch's tokens and then its targets would hand every geometry
        # another interleaving of one stream -- a different batch.
        whole = (1, per_microbatch * accumulation)
        tokens = torch.randint(self.vocabulary, whole, generator=generator)
        targets = torch.randint(self.vocabulary, whole, generator=generator)
        return tuple(
            (
                tokens[:, start : start + per_microbatch].clone(),
                targets[:, start : start + per_microbatch].clone(),
                lengths,
            )
            for start in range(0, whole[1], per_microbatch)
        )

    def step_microbatches(
        self, sequences: int, accumulation: int
    ) -> tuple[tuple[object, ...], ...]:
        """The same tokens for every step, budget and geometry.

        Every step trains on one batch, so the five steps overfit it and the
        loss visibly falls -- five steps of fresh data show nothing. It also
        makes the loss curve identical across budgets, so a divergence between
        two of them is a difference in what was computed rather than in what
        was sampled.
        """

        generator = torch.Generator().manual_seed(self.arguments.seed * 1_000_003)
        return self.example_microbatches(sequences, accumulation, generator=generator)

    @property
    def lanes(self) -> TransferBandwidths | None:
        """The lanes the run phase plans against.

        The search's, pinned or calibrated once for this run, so each budget
        asks the store the search's question and executes the plan the search
        chose. Priced against a fresh calibration, the same request would be a
        different key and, under `require`, a refusal.
        """

        if self.report is None:
            return cast(TransferBandwidths | None, self.arguments.transfer_bandwidths)
        return self.report.planned_lanes

    def search(self) -> None:
        """The geometry search, or the manual geometry's announcement."""

        arguments, request, paths = self.arguments, self.request, self.paths
        case, plan_log, ledger = self.case, self.plan_log, self.ledger
        manifest, runtime = request.manifest, self.runtime
        if request.manual is not None:
            geometry = (request.manual, request.sequences_per_step // request.manual)
            print(
                f"  geometry chosen manually: {geometry[0]} sequences per"
                f" microbatch x {geometry[1]} accumulation rounds"
            )
            print()
        else:
            print("  searching… (fresh geometries compile and profile first;")
            print("              warm reruns reuse the artifact store)", flush=True)

            def progress(message: str) -> None:
                plan_log.note(message)
                # Each geometry materializes model and optimizer state and
                # tears it down again, so a geometry boundary is where host
                # growth across builds would show.
                if message.startswith("geometry"):
                    note_host_memory(plan_log, message.split(":")[0])

            with contextlib.redirect_stdout(plan_log):
                report = self.report = plan_step_search(
                    case.model,
                    objective=case.objective,
                    optimizer=self.optimizer,
                    hyperparams=("lr",),
                    master_dtype=self.precision.master,
                    grad_dtype=self.precision.gradients,
                    round_accumulation_once=self.precision.round_accumulation_once,
                    example_microbatches=self.example_microbatches,
                    total_sequences_per_step=request.sequences_per_step,
                    sequence_length=request.sequence_length,
                    budgets=[
                        (budget, manifest.spill_budget_bytes)
                        for budget in self.budgets.search
                    ],
                    runtime=runtime,
                    execution="execution",
                    spill="spill",
                    min_tokens_per_microbatch=arguments.min_tokens_per_microbatch,
                    max_tokens_per_microbatch=arguments.max_tokens_per_microbatch,
                    artifact_store=paths.store,
                    build_store=paths.build_store,
                    plan_store=paths.plan_store,
                    build_store_mode=arguments.build_store_mode,
                    plan_store_mode=arguments.plan_store_mode,
                    verbose=True,
                    progress=progress,
                    incumbents=arguments.incumbents,
                    orderings=(
                        None
                        if arguments.orderings == "factors"
                        else lambda accumulation: (
                            StepDataOrdering.depth_first(accumulation),
                        )
                    ),
                    search_options=self.policy,
                    profiling_options=self.profiling,
                    transfer_bandwidths=arguments.transfer_bandwidths,
                    export_bypass_key=arguments.export_bypass_key,
                    keep_resolutions=arguments.resolution_plans,
                )
            print()
            print_search(report, request.tokens_per_step)
            report_path = paths.root / "search.json"
            print(f"  search report: {report.save(report_path)}")
            note_host_memory(plan_log, "geometry search finished")
            print()
            for build in report.geometries:
                for name, value in build.phase_seconds.items():
                    if name != "total":
                        ledger[f"build: {name}"] = (
                            ledger.get(f"build: {name}", 0.0) + value
                        )
            ledger["search"] = report.total_search_seconds
            ledger["build: unattributed"] = max(
                0.0,
                report.total_build_seconds
                - sum(
                    value
                    for name, value in ledger.items()
                    if name.startswith("build: ")
                ),
            )

    def plot_search(self) -> None:
        arguments, report = self.arguments, self.report
        if arguments.plots:
            if report is None:
                print("  plots need a search; skipped for a manual geometry")
            else:
                plot_dir = self.paths.root / "figures"
                marker = time.perf_counter()
                written = plot_step_search(report, plot_dir)
                self.ledger.charge("figures", marker)
                print(rule("Figures"))
                for path in written:
                    print(f"  {path}")
                print()

    def _run_steps(
        self, training: Any, plan_report: Any, geometry: tuple[int, int]
    ) -> _Steps:
        """Run the steps on one plan and say what each one took."""

        arguments, plan_log = self.arguments, self.plan_log
        tokens_per_step = self.request.tokens_per_step
        losses: dict[int, float] = {}
        cycles: dict[int, float] = {}
        hosts: dict[int, float] = {}

        def report_cycles() -> None:
            # A step's cycle closes when the next step begins, or at the
            # end marker after the last one, so each line appears one
            # step late. Through the log rather than print(), so every
            # step time is in the run directory as well as on the
            # terminal. Only planner phase lines are filtered out of
            # stdout.
            for timing in training.invocation_timings():
                step = timing.step_number
                cycles[step] = timing.cycle_seconds
                note = ""
                # A cycle closes where the next step opens, so a step is
                # reported one step late and the traced step's predecessor
                # arrives beside it. Naming the traced one is what keeps that
                # from reading as two traced steps.
                if step == arguments.steps:
                    note += "   (traced; not in the median)"
                plan_log.write(
                    f"  step {step:>3}   {timing.cycle_seconds:7.3f} s"
                    f"   {tokens_per_step / timing.cycle_seconds:>10,.0f} tok/s"
                    f"   loss {losses[step]:.4f}{note}\n"
                )

        def run_step(step: int, *, traced: bool) -> Any:
            started = time.perf_counter()
            result = training(
                self.step_microbatches(*geometry),
                hyperparams={"lr": LEARNING_RATE},
                runtime_trace=traced,
            )
            # Each microbatch returns its share of the step's mean loss over
            # trained tokens, so the step's loss is their sum.
            losses[step] = sum(
                float(objective)
                if (head := _head_loss_share(metrics)) is None
                else head
                for objective, metrics in zip(
                    result.objectives, result.metrics, strict=True
                )
            )
            hosts[step] = time.perf_counter() - started
            report_cycles()
            return result

        training.prepare_runtime_trace()
        if arguments.steps > 1:
            print(rule("Steps"))
            for step in range(1, arguments.steps):
                result = run_step(step, traced=False)
        print(rule("Traced step versus simulation"))
        result = run_step(arguments.steps, traced=True)
        # Close the last step's cycle where a next step would begin, so
        # its time reads like every other step's, then resolve the trace
        # with that cycle in it.
        training.mark_cycle_end()
        report_cycles()
        # Every cycle runs origin to next origin, so consecutive cycles
        # tile the run: their sum is the span from the first step's start
        # to the last one's end, and tokens over that span is the one
        # throughput a boundary between steps cannot hide in.
        elapsed = sum(cycles.values())
        walls = [cycles[step] for step in sorted(cycles)]
        # Two of these steps are not the step this reports. The first pays
        # the plan's reconciliation of its initial state. The last is the
        # traced one, and tracing costs it tens of milliseconds of collection
        # -- enough that it came out slower than its predecessor at almost
        # every budget measured -- so including it biases the median upward
        # every time. The qualification gate makes the same exclusion.
        untraced = walls[:-1] if len(walls) > 1 else walls
        measured = untraced[1:] if len(untraced) > 1 else untraced
        # Both sides are the whole step -- the median on the device clock
        # against the plan's makespan -- and these are the same two values
        # the figures use, so the percentage here and the figure's relative
        # error cannot drift apart. Written with the steps it summarizes,
        # rather than after the epilogue, where it would read as part of the
        # trace.
        median_step = statistics.median(measured)
        simulated_step = plan_report.summary.simulated_step_seconds
        plan_log.write(
            f"\n  end to end {elapsed:8.3f} s"
            f"   ({elapsed / len(cycles):.3f} s per step)"
            f"   {len(cycles) * tokens_per_step / elapsed:>10,.0f} tok/s"
            f"   ({len(cycles)} steps, every boundary included)\n"
        )
        plan_log.write(
            f"  median step {median_step:7.3f} s"
            f"   simulated {simulated_step:7.3f} s"
            f"   ({(median_step - simulated_step) / simulated_step:+.2%})"
            f"   ({len(measured)} untraced"
            f" step{'' if len(measured) == 1 else 's'} after the first)\n"
        )
        assert result.diagnostics is not None
        diagnostics = result.diagnostics.result()
        # The final StepResult's public outputs are caller-owned device
        # tensors; the runtime refuses to close while they are alive.
        del result
        gc.collect()
        return _Steps(
            diagnostics=diagnostics,
            walls=tuple(walls),
            median_step_seconds=median_step,
            simulated_step_seconds=simulated_step,
            host_seconds=sum(hosts.values()),
        )

    def run_one_budget(
        self,
        budget: int,
        geometry: tuple[int, int],
        ordering: StepDataOrdering,
        incumbent: AnnotatedProgramPlan | None = None,
    ) -> RunBudgetOutcome:
        """Plan one budget, run its steps, close it, and own nothing after.

        Returns the budget beside its simulated and measured step times.

        One budget's plan must be entirely gone before the next one is
        built: they hold model, optimizer, and compiled state at the same
        scale, and the host has room for one of them beside the pinned
        spill arena. Every reference to this budget's plan lives in this
        frame, so returning is what releases them.
        """

        arguments, request, paths = self.arguments, self.request, self.paths
        manifest, runtime, ledger = request.manifest, self.runtime, self.ledger
        plan_log, tokens_per_step = self.plan_log, request.tokens_per_step
        if self.trained:
            # Every budget starts from the same weights and a fresh
            # optimizer, on the same tokens per step, so its losses agree
            # with every other budget's bar reduction order: the run is a
            # correctness check as well as a measurement.
            marker = time.perf_counter()
            release_case_model(self.case, runtime=runtime)
            self.case = build_case(manifest, seed=arguments.seed, runtime=runtime)
            ledger.charge("model construction", marker)
            plan_log.note("model and optimizer state reset for a comparable run")
        self.trained = True
        case = self.case
        microbatches = self.example_microbatches(*geometry)
        marker = time.perf_counter()
        plan_sink: Any = contextlib.redirect_stdout(plan_log)
        plan_log.note(f"run planning at execution {gib(budget)}")
        with plan_sink:
            training = plan_step(
                case.model,
                objective=case.objective,
                optimizer=self.optimizer,
                hyperparams=("lr",),
                example_inputs=microbatches,
                master_dtype=self.precision.master,
                grad_dtype=self.precision.gradients,
                round_accumulation_once=self.precision.round_accumulation_once,
                runtime=runtime,
                execution="execution",
                spill="spill",
                execution_budget=budget,
                optimizer_ordering="stage_interleaved",
                depth=ordering.depth,
                breadth=ordering.breadth,
                reverse_breadth=ordering.reverse_breadth,
                pair_loss=ordering.pair_loss,
                artifact_store=paths.store,
                build_store=paths.build_store,
                plan_store=paths.plan_store,
                build_store_mode=arguments.build_store_mode,
                plan_store_mode=arguments.plan_store_mode,
                export_bypass_key=arguments.export_bypass_key,
                # The search policy the geometry search used, so the run
                # plans the plan the search promised rather than missing
                # the store and searching again under other options.
                search_options=self.policy,
                profiling_options=self.profiling,
                # The search's winning plan is the plan to beat, so the
                # step executes what the search chose, or better, even
                # when the replan's facts differ from the search's and
                # the store cannot hand the plan back.
                incumbent=incumbent,
                transfer_bandwidths=self.lanes,
                keep_resolutions=arguments.resolution_plans,
            )
        ledger.charge("run planning", marker)
        note_host_memory(plan_log, f"planned {gib(budget)}")
        plan_report = training.plan_report
        print_breakdown(plan_report, tokens_per_step)

        steps = self._run_steps(training, plan_report, geometry)
        diagnostics, walls = steps.diagnostics, steps.walls
        median_step, simulated_step = (
            steps.median_step_seconds,
            steps.simulated_step_seconds,
        )
        print()
        print(rule("Traced step versus simulation"))
        print()
        print_epilogue(diagnostics)
        trace_path = paths.root / "steps" / f"{budget / _GIB:g}gib.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(diagnostics.as_dict(), indent=2, sort_keys=True)
        )
        print(f"  step diagnostics: {trace_path}")
        ledger["steps execution"] = (
            ledger.get("steps execution", 0.0) + steps.host_seconds
        )
        training.close()
        step_summary = diagnostics.summary
        if step_summary.real_terminal_tail_seconds is None:
            raise RuntimeError("run trace omitted terminal transfer timestamps")
        return RunBudgetOutcome(
            execution_budget_bytes=budget,
            simulated_step_seconds=simulated_step,
            measured_step_seconds=median_step,
            step_seconds=walls,
            profiled_task_seconds=step_summary.profiled_task_seconds,
            real_task_seconds=step_summary.real_task_event_seconds,
            simulated_idle_seconds=(step_summary.simulated_inter_task_idle_seconds),
            real_idle_seconds=step_summary.real_inter_task_idle_seconds,
            recomputation_seconds=(plan_report.summary.recomputation_overhead_seconds),
            simulated_entry_delay_seconds=step_summary.simulated_entry_delay_seconds,
            real_entry_delay_seconds=step_summary.entry_delay_seconds,
            terminal_tail_seconds=step_summary.simulator_terminal_tail_seconds,
            real_terminal_tail_seconds=step_summary.real_terminal_tail_seconds,
        )

    def run(self) -> list[RunBudgetOutcome]:
        """Every run budget, on the search's winner or the manual geometry."""

        arguments, request, plan_log = self.arguments, self.request, self.plan_log
        manifest, report = request.manifest, self.report
        run_entries: list[RunBudgetOutcome] = []
        for budget in self.budgets.run:
            incumbent: AnnotatedProgramPlan | None = None
            if request.manual is not None:
                geometry = (
                    request.manual,
                    request.sequences_per_step // request.manual,
                )
                ordering = StepDataOrdering.depth_first(geometry[1])
            else:
                assert report is not None
                winner = report.winner(budget, manifest.spill_budget_bytes)
                if winner is None:
                    print(
                        f"  no geometry planned successfully at {gib(budget)};"
                        " skipping this run budget"
                    )
                    continue
                geometry = (
                    winner.sequences_per_microbatch,
                    winner.accumulation_count,
                )
                ordering = winner.ordering
                incumbent = report.winner_plans.get(
                    (budget, manifest.spill_budget_bytes)
                )
            print(rule(f"Run at execution {gib(budget)}"))
            print(
                f"  geometry {geometry[0]} sequences per microbatch"
                f" x {geometry[1]} microbatches, walked {ordering.label}"
            )
            print()
            # A budget whose plan could not be admitted is a result about that
            # budget, not about the tour: an infeasible *plan* already skips with
            # a message, and a refused layout should read the same way rather
            # than discarding every budget after it. The figures for the budgets
            # that did run are worth more than a stack trace.
            try:
                run_entries.append(
                    self.run_one_budget(budget, geometry, ordering, incumbent)
                )
            except RuntimeExecutionError as error:
                print(f"  {gib(budget)} could not be admitted: {error}")
                plan_log.note(f"{gib(budget)} refused admission; skipping")
                gc.collect()
            if arguments.plots and run_entries:
                # The record is kept current rather than written once at the
                # end, so a run that stops early still leaves what it measured
                # and its figures can be redrawn from the tables.
                write_run_tables(
                    run_entries,
                    self.paths.root / "figures" / "raw_data",
                    tokens_per_step=request.tokens_per_step,
                )
            # The frame that owned the closed plan is gone; collect what its
            # internals hold in cycles, so the host memory that plan still
            # occupies is free before the next budget plans.
            gc.collect()
            note_host_memory(plan_log, f"closed the {gib(budget)} plan")
        return run_entries

    def close(self, run_entries: list[RunBudgetOutcome]) -> None:
        """The run's figures, then the model's state back to the pool."""

        arguments = self.arguments
        self.log_handle.close()
        if arguments.plots and run_entries:
            plot_dir = self.paths.root / "figures"
            marker = time.perf_counter()
            written_run = plot_step_run(
                run_entries,
                plot_dir,
                tokens_per_step=self.request.tokens_per_step,
            )
            self.ledger.charge("figures", marker)
            for path in written_run:
                print(f"  figure: {path}")
            print()
        if arguments.timelines:
            # Every plan the run made, as pages: the pools over the step and
            # the fetch, compute and evict lanes, on the simulated clock for
            # every search point and on the device's too for every budget
            # that ran.
            marker = time.perf_counter()
            # Minutes of silence otherwise, on a tour: say what is happening.
            try:
                index = write_run_timelines(
                    self.paths.root,
                    progress=lambda line: print(f"  {line}", flush=True),
                )
            except Exception as error:
                # The run's data is complete on disk; a page that cannot be
                # drawn is reported, and the tool can be run on it later.
                print(f"  timelines: not written ({type(error).__name__}: {error})")
            else:
                self.ledger.charge("timelines", marker)
                print(f"  timelines: {index}")
            print()
        release_case_model(self.case, runtime=self.runtime)


def print_closing(ledger: Ledger, request: Request) -> None:
    """Where the time and the host memory went."""

    total = time.perf_counter() - ledger.started
    ledger["everything else"] = max(0.0, total - sum(ledger.values()))
    print(rule("Where the time went"))
    for name, value in sorted(ledger.items(), key=lambda item: -item[1]):
        share = value / total if total else 0.0
        print(f"  {name:<38}{value:9.1f} s  {bar(share)}  {share:6.1%}")
    print(f"  {'total':<38}{total:9.1f} s")
    print()
    resident, peak = host_memory()
    ceiling = host_memory_ceiling()
    capacity = request.manifest.spill_budget_bytes
    print(rule("Where the host memory went"))
    # A remote arena is the peer's memory, not this host's, so it is named
    # rather than counted here -- the heading above says where the host's
    # memory went, and those gibibytes did not go there.
    if request.remote_spill is None:
        print(f"  spill arena (pinned)  {gib(capacity):>12}   counted below")
    else:
        host, port = request.remote_spill
        print(
            f"  spill arena (remote)  {gib(capacity):>12}   on {host}:{port},"
            " not counted below"
        )
    print(f"  peak resident         {gib(peak):>12}")
    print(f"  resident at exit      {gib(resident):>12}")
    if ceiling is not None:
        print(
            f"  cgroup ceiling        {gib(ceiling):>12}"
            f"   ({gib(max(0, ceiling - peak))} unused at the peak)"
        )
    print()


def main() -> int:
    parser, arguments = parse_arguments()
    request = resolve_request(parser, arguments)
    paths = prepare_run_root(arguments, request)
    if paths is None:
        return 1
    ledger = Ledger()
    runtime = open_runtime(request, ledger)
    budgets = plan_budgets(runtime, request)
    print_banner(arguments, request, runtime, budgets)
    tour = Tour(arguments, request, paths, budgets, runtime, ledger)
    with tour.case.implementations():
        tour.search()
        tour.plot_search()
        entries = tour.run()
        tour.close(entries)
    runtime.close()
    print_closing(ledger, request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
