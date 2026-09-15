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
import gc
import json
import resource
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.planner import (
    GenericPlanningOptions,
    InitialPlacement,
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
    DEFAULT_RESOLUTION_OPTIONS,
    validate_resolution_options,
)
from shadowspill.plots import (
    RunBudgetOutcome,
    plot_step_run,
    plot_step_search,
    write_run_tables,
)
from shadowspill.pytorch import Runtime, StepSearchReport, plan_step, plan_step_search
from shadowspill.pytorch.diagnostics.execution import TaskRecord, TransferRecord
from shadowspill.pytorch.planning import planned_transfer_bandwidths
from shadowspill.pytorch.runtime_adapter.failures import RuntimeExecutionError
from shadowspill.pytorch.runtime_adapter.runtime.configuration import (
    resolve_execution_budget,
)
from shadowspill.pytorch.step_search import search_geometries
from shadowspill.schema import artifact_schema
from shadowspill.store import STORE_MODES
from tools.qualification.model_state import release_case_model
from workloads.common.training import LEARNING_RATE, optimizer_state_init
from workloads.full_model import build_case, manifest_for
from workloads.providers import ModelImplementation


def _budget_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


#: Named resolution options. ``quarters`` is the library's own default,
#: spelled out here so every run records the shares it actually planned.
_NAMED_RESOLUTION_OPTIONS: dict[str, tuple[str, ...]] = {
    "quarters": tuple(str(share) for share in DEFAULT_RESOLUTION_OPTIONS),
    "eighths": tuple(f"{numerator}/8" for numerator in range(9)),
    "halves": ("0", "1/2", "1"),
}


def _named_resolution_options(value: str) -> tuple[str, ...]:
    """A named set, or a comma-separated list of exact fractions."""

    if value in _NAMED_RESOLUTION_OPTIONS:
        return _NAMED_RESOLUTION_OPTIONS[value]
    return tuple(item.strip() for item in value.split(",") if item.strip())


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

    The pinned spill arena is one page-locked mapping, so it counts in full
    from the moment the runtime registers it; everything the frontend holds
    on the host -- imported state, captured optimizer state, compiled
    artifacts -- counts on top of it.
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
                initial_placement=InitialPlacement(arguments.initial_placement),
                resolution_options=tuple(
                    Fraction(share) for share in arguments.resolution_options
                ),
            )
        ),
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
    print()
    print(
        f"  traffic per step   fetch {gib(report.transfer_bytes_fetched)}"
        f"   evict {gib(report.transfer_bytes_evicted)}"
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
    # The headline compares the step against the step. The simulated step is the
    # makespan -- the same figure the plan's breakdown advertises -- and it is the
    # task window plus the terminal writeback priced after the last task. The real
    # step is the cycle, and it is the opening delay plus the task window plus the
    # tail the stream actually exposed. Each side decomposes into exactly those
    # three parts, so the rows below sum to their own total and the one the
    # simulator gets wrong is visible rather than inferred.
    #
    # Comparing the task window against the makespan instead would flatter the
    # simulator: it would credit a tail while ignoring an opening the simulator
    # prices at zero.
    simulated_step = summary.simulator_makespan_seconds
    real_step = summary.cycle_seconds
    real_span = summary.real_selected_span_seconds
    simulated_span = summary.simulated_selected_span_seconds
    simulated_tail = summary.simulator_terminal_tail_seconds
    if real_step is None:
        print(
            "  step: the trace resolved before the next step opened, so the step"
            " itself is not comparable here -- only the task window below"
        )
    else:
        # This is the traced step, which is the last one run, and steps drift
        # slightly slower across a budget -- so this figure reads a few tenths of
        # a percent worse than the one the figures draw, which is the median of
        # the untraced steps. Saying which step this is keeps the two from
        # looking like they disagree.
        print(
            "  traced step: origin on the compute stream through the next step's"
            " origin, against the simulated makespan"
        )
        print(
            f"  traced step      real {real_step:.3f} s"
            f"   simulated {simulated_step:.3f} s"
            f"   ({(real_step - simulated_step) / simulated_step:+.2%})"
        )
        print(
            f"    opening        real {summary.opening_delay_seconds:.3f} s"
            f"   simulated {simulated_step - simulated_span - simulated_tail:.3f} s"
            "   (the restore; unmodeled, docs/architecture/step-boundaries.md)"
        )
    print(
        f"    task window    real {real_span:.3f} s"
        f"   simulated {simulated_span:.3f} s"
        f"   ({(real_span - simulated_span) / simulated_span:+.2%})"
        "   (first task's compute start through the last task's end)"
    )
    if real_step is not None and summary.exposed_tail_seconds is not None:
        print(
            f"    terminal tail  real {summary.exposed_tail_seconds:.3f} s"
            f"   simulated {simulated_tail:.3f} s"
            "   (writeback; the simulator overlaps none of it with the next step)"
        )
    print(
        "  stalled          real"
        f" {summary.real_inter_task_readiness_wait_seconds:.3f} s"
        f"   simulated {summary.simulated_inter_task_readiness_wait_seconds:.3f} s"
    )
    print(
        "  first task       waited"
        f" {summary.real_initial_readiness_wait_seconds * 1e3:.1f} ms"
        " for its own inputs, inside the opening above"
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


_REQUEST_SCHEMA = artifact_schema("quickstart_request")

#: What a request is made of: every argument except the ones that say where
#: this run writes or what it draws, which a reproduction chooses for itself.
_NOT_REQUEST = frozenset({"reproduce", "output_dir", "force_overwrite", "plots"})
_REPRODUCE_MAY_TAKE = frozenset({"--reproduce", "--output-dir", "--plots"})


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
        "--orderings",
        choices=("factors", "depth-first"),
        default="factors",
        help="which microbatch orderings the search tries per geometry:"
        " every depth x breadth factor pair (the default), or only the"
        " depth-first walk. The loss stays paired and the backward walk"
        " reversed either way; the search does not toggle those",
    )
    parser.add_argument(
        "--initial-placement",
        choices=("greedy", "required"),
        default=None,
        help="how objects the declaration leaves in spill may be placed before"
        " the first task. 'greedy' promotes a cold object to the opening boundary"
        " when its fetch would otherwise be late, which moves those bytes out of"
        " the schedule and into the opening restore -- where the simulated"
        " makespan does not count them. 'required' places only what the"
        " declaration asks for, plus what the first task reads and so cannot be"
        " fetched in time. Defaults to whichever the library chooses, rather than"
        " naming one here that a change to that choice would leave behind",
    )
    parser.add_argument(
        "--resolution-options",
        type=_named_resolution_options,
        default=_NAMED_RESOLUTION_OPTIONS["quarters"],
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
        " searched again. Only --output-dir and --plots may be given with it",
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
    if arguments.initial_placement is None:
        arguments.initial_placement = PressureFitOptions().initial_placement.value
    if arguments.steps < 1:
        parser.error("--steps must be at least 1")
    try:
        validate_resolution_options(arguments.resolution_options)
    except ValueError as error:
        parser.error(f"--resolution-options: {error}")

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
    return Request(
        manifest=manifest,
        search_budgets=search_budgets,
        run_budgets=run_budgets,
        physical_capacity=physical_capacity,
        sequence_length=sequence_length,
        sequences_per_step=sequences_per_step,
        tokens_per_step=tokens_per_step,
        manual=manual,
    )


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
        for name in ("search.json", "progress.log", "steps", "figures")
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
    runtime = Runtime(
        pools={
            "execution": device(physical_capacity=request.physical_capacity),
            "spill": pinned_host(capacity=request.manifest.spill_budget_bytes),
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
    # Which placement ran is not recoverable from the figures, and the two
    # price the opening differently -- greedy moves bytes into the unpriced
    # restore -- so a report that does not say which it used cannot be compared
    # against one that used the other.
    print(
        f"  initial placement   {arguments.initial_placement:>10}      "
        + (
            "cold objects may be promoted to the opening boundary,"
            " so their bytes land in the restore rather than the schedule"
            if arguments.initial_placement == "greedy"
            else "only what the declaration asks for, plus what the first"
            " task reads and cannot be fetched in time"
        )
    )

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
        shape = (1, sequences * sequence_length)
        lengths = (sequence_length,) * sequences
        return tuple(
            (
                torch.randint(self.vocabulary, shape, generator=generator),
                torch.randint(self.vocabulary, shape, generator=generator),
                lengths,
            )
            for _ in range(accumulation)
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
                    optimizer=case.optimizer,
                    optimizer_state_init=optimizer_state_init,
                    hyperparams=("lr",),
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
                    transfer_bandwidths=arguments.transfer_bandwidths,
                    export_bypass_key=arguments.export_bypass_key,
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
                if step == 1 and plan_report.initial_search_result is not None:
                    note = "   (first-step plan)"
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
            losses[step] = statistics.fmean(float(value) for value in result.objectives)
            hosts[step] = time.perf_counter() - started
            report_cycles()
            return result

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
                optimizer=case.optimizer,
                optimizer_state_init=optimizer_state_init,
                hyperparams=("lr",),
                example_inputs=microbatches,
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
                # The search's winning plan is the plan to beat, so the
                # step executes what the search chose, or better, even
                # when the replan's facts differ from the search's and
                # the store cannot hand the plan back.
                incumbent=incumbent,
                transfer_bandwidths=self.lanes,
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
            # The whole opening, not the first task's wait for its own
            # inputs: the restore runs before the first task's compute starts,
            # and the simulator prices none of it, so it is the measured
            # step's largest unmodelled part.
            prologue_seconds=step_summary.opening_delay_seconds,
            terminal_tail_seconds=(step_summary.simulator_terminal_tail_seconds),
            real_terminal_tail_seconds=(step_summary.exposed_tail_seconds or 0.0),
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
        release_case_model(self.case, runtime=self.runtime)


def print_closing(ledger: Ledger, manifest: Any) -> None:
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
    print(rule("Where the host memory went"))
    print(f"  spill arena (pinned)  {gib(manifest.spill_budget_bytes):>12}")
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
    print_closing(ledger, request.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
