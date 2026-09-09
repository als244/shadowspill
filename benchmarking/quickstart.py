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
        --run-budget-gib 6,7,8,9,10,12,16,20,24,28,30 --spill-gib 112 --steps 5 \\
        --plots

Every flag defaults to the model's retained qualification value, so
`python -m benchmarking.quickstart mlops_olmoe` searches and runs the
known cell.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import resource
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.planner import PressureFitOptions, StepDataOrdering
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.planner.recomputation import (
    DEFAULT_RESOLUTION_OPTIONS,
    validate_resolution_options,
)
from shadowspill.plots import RunBudgetOutcome, plot_step_run, plot_step_search
from shadowspill.pytorch import Runtime, StepSearchReport, plan_step, plan_step_search
from shadowspill.pytorch.diagnostics.execution import TaskRecord, TransferRecord
from shadowspill.pytorch.step_search import search_geometries
from tools.qualification.model_state import release_case_model
from workloads.common.training import LEARNING_RATE, optimizer_state_init
from workloads.full_model import build_case, manifest_for
from workloads.providers import ModelImplementation


def _budget_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


#: Named resolution options; ``None`` is the library's default of quarters.
_NAMED_RESOLUTION_OPTIONS: dict[str, tuple[str, ...] | None] = {
    "quarters": None,
    "eighths": tuple(f"{numerator}/8" for numerator in range(9)),
    "halves": ("0", "1/2", "1"),
}


def _named_resolution_options(value: str) -> tuple[str, ...] | None:
    """A named set, or a comma-separated list of exact fractions."""

    if value in _NAMED_RESOLUTION_OPTIONS:
        return _NAMED_RESOLUTION_OPTIONS[value]
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _transfer_bandwidths(value: str) -> TransferBandwidths:
    """``FETCH,EVICT[,FETCH_US,EVICT_US]``, or a search.json to pin to."""

    path = Path(value)
    if path.suffix == ".json":
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise argparse.ArgumentTypeError(f"{value}: {error}") from error
        recorded = report.get("transfer_bandwidths") or next(
            (
                item.get("transfer_bandwidths")
                for item in report.get("geometries", ())
                if item.get("transfer_bandwidths")
            ),
            None,
        )
        if recorded is None:
            raise argparse.ArgumentTypeError(f"{value} records no transfer calibration")
        return TransferBandwidths.from_value(recorded, "transfer_bandwidths")
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


def gib(value: float) -> str:
    return f"{value / _GIB:.2f} GiB"


def gb_s(value: float) -> str:
    """Bandwidth in decimal GB/s, which is how the planner rounds rates."""

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


def print_search(report: StepSearchReport, tokens_per_step: int) -> None:
    print(rule("Geometry search"))
    print(
        "  seqs/microbatch x accumulation, then the walk as depth x breadth"
        " (r: reversed backward, p: paired loss); fastest simulated step wins"
    )
    lanes = report.transfer_bandwidths or next(
        (
            item.transfer_bandwidths
            for item in report.geometries
            if item.transfer_bandwidths
        ),
        None,
    )
    if lanes is not None:
        print(
            f"  planned against fetch {lanes.fetch_bytes_per_second / 1e9:.0f} GB/s,"
            f" evict {lanes.evict_bytes_per_second / 1e9:.0f} GB/s"
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
    extra = simulated - summary.unconstrained_step_seconds
    print(rule("The chosen plan's breakdown"))
    print(f"  simulated step   {simulated:8.3f} s   {tokens / simulated:>10,.0f} tok/s")
    print(
        f"  unconstrained    {summary.unconstrained_step_seconds:8.3f} s"
        f"   {tokens / summary.unconstrained_step_seconds:>10,.0f} tok/s"
        "   (cheapest graphs, no waiting)"
    )
    print()
    print(f"  where the extra {extra:.3f} s goes, as shares of the step")
    for label, value in (
        ("recomputation", summary.recomputation_overhead_seconds),
        (
            "stalled",
            summary.idle_seconds + summary.terminal_writeback_seconds,
        ),
    ):
        share = value / simulated if simulated > 0 else 0.0
        print(f"    {label:<22}{value:+8.3f} s  {bar(share)}  {share:6.1%}")
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
    fetch, evict = report.fetch_profile, report.evict_profile
    print(
        f"  fetch bandwidth    {gb_s(fetch.bandwidth_bytes_per_second)} assumed"
        f"   ({gb_s(fetch.solo_bandwidth_bytes_per_second)} solo)"
    )
    print(
        f"  evict bandwidth    {gb_s(evict.bandwidth_bytes_per_second)} assumed"
        f"   ({gb_s(evict.solo_bandwidth_bytes_per_second)} solo)"
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
    real = summary.real_selected_span_seconds
    simulated = summary.simulated_selected_span_seconds
    print("  task window: first task's compute start through the last task's end")
    print(
        f"  task window      real {real:.3f} s   simulated {simulated:.3f} s"
        f"   ({(real - simulated) / simulated:+.2%})"
    )
    print(
        "  stalled          real"
        f" {summary.real_inter_task_readiness_wait_seconds:.3f} s"
        f"   simulated {summary.simulated_inter_task_readiness_wait_seconds:.3f} s"
    )
    print(
        "  opening restore  first task waited"
        f" {summary.real_initial_readiness_wait_seconds * 1e3:.1f} ms"
        "   (unmodeled; docs/architecture/step-boundaries.md)"
    )
    print(
        "  terminal tail    simulated"
        f" {summary.simulator_terminal_tail_seconds * 1e3:.1f} ms"
        " of writeback after the last task"
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
            + (
                f"; effective {gb_s(effective)}"
                if effective is not None
                else ""
            )
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", choices=IDENTITIES)
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
        "--resolution-options",
        type=_named_resolution_options,
        default="quarters",
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
        help="plan the search against this calibration instead of the one the"
        " runtime measures at start: FETCH,EVICT in GB/s, optionally followed"
        " by the fetch and evict latencies in microseconds, or the path of"
        " another run's search.json to pin to what that run planned against."
        " The run phase keeps the live calibration",
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
        "<model>_<revision>/seq<length>/seqsperstep<n>",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--artifact-store",
        type=Path,
        default=None,
        help="the captures, graph pairs, profiles and lowered programs to read"
        " and write; point it at another run's store to skip work already"
        " paid for there. Defaults to <output-dir>/artifact_store",
    )
    parser.add_argument(
        "--plan-store",
        type=Path,
        default=None,
        help="where this run's plans go: every selection request, selection"
        " and plan manifest, kept apart from the artifact store so a shared"
        " store never hands a run another run's plans. Defaults to"
        " <output-dir>/plan_store",
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
    arguments = parser.parse_args()
    if arguments.steps < 1:
        parser.error("--steps must be at least 1")
    if arguments.resolution_options is not None:
        try:
            validate_resolution_options(arguments.resolution_options)
        except ValueError as error:
            parser.error(f"--resolution-options: {error}")

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
    # Everything a run leaves behind lands together: the search report, its
    # log, and one step trace per run budget.
    # One directory per run: the model and the revision it measured, then
    # sequence length, then sequences per step, so each level is exactly one
    # parameter and another run is a sibling rather than an overwrite.
    run_root = arguments.output_dir or (
        Path("benchmarking/quickstart_reports")
        / f"{arguments.model}_{_revision()}"
        / f"seq{sequence_length}"
        / f"seqsperstep{sequences_per_step}"
    )
    # A run directory is written once. Changing budgets for the same model,
    # length and step size produces a different answer at the same path, and
    # silently replacing the old one loses a measurement that cost real time.
    # Refuse instead, and say both ways out.
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
        return 1
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
    plan_store = arguments.plan_store or (run_root / "plan_store")

    # Built before the banner so the banner can state the calibrated rates as
    # measurements rather than a promise. Calibration happens once here and is
    # reused by every geometry.
    ledger: dict[str, float] = {}
    command_started = time.perf_counter()

    def charge(category: str, started: float) -> None:
        ledger[category] = ledger.get(category, 0.0) + (time.perf_counter() - started)

    marker = time.perf_counter()
    runtime = Runtime(
        pools={
            "execution": device(physical_capacity=physical_capacity),
            "spill": pinned_host(capacity=manifest.spill_budget_bytes),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
    )
    charge("runtime construction and calibration", marker)

    print("═" * 68)
    print(f"  ShadowSpill quickstart — {arguments.model}")
    print("═" * 68)
    searched = ", ".join(gib(item) for item in search_budgets)
    ran = ", ".join(gib(item) for item in run_budgets) or "none (search only)"
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
    shares = arguments.resolution_options or DEFAULT_RESOLUTION_OPTIONS
    print(
        "  resolutions         "
        + f"{len(shares):>10}      "
        + ", ".join(str(item) for item in shares)
        + " of the flexible groups recomputing"
    )

    # One calibration serves every geometry, so it is a property of the run.
    # The planned rate is the concurrent one rounded to whole GB/s, which is
    # what the simulator is built with; solo is what a copy gets alone.
    pinned = arguments.transfer_bandwidths
    if pinned is not None:
        print(
            "  transfer lanes      "
            f"pinned to fetch {pinned.fetch_bytes_per_second / 1e9:.0f} GB/s,"
            f" evict {pinned.evict_bytes_per_second / 1e9:.0f} GB/s"
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
        for name, source, destination in (
            ("fetch", "spill", "execution"),
            ("evict", "execution", "spill"),
        ):
            profile = capabilities.route(source, destination)
            planned = round(profile.bandwidth_bytes_per_second / _GB)
            print(
                f"  {name + ' lane':<19} "
                f"{planned:>3} GB/s planned"
                f"   (concurrent {gb_s(profile.concurrent_bandwidth_bytes_per_second)},"
                f" solo {gb_s(profile.solo_bandwidth_bytes_per_second)},"
                f" latency {profile.latency_nanoseconds / 1e3:.0f} us)"
            )
    print()

    note_host_memory(None, "runtime pools registered")
    marker = time.perf_counter()
    # Built inside the pool it will live in, so the parameters are written
    # Declared on meta and materialised straight into the spill pool, so the
    # model's values are written where they will live and no host memory
    # proportional to it is ever allocated.
    case = build_case(manifest, seed=arguments.seed, runtime=runtime)
    charge("model construction in the spill pool", marker)
    note_host_memory(None, "model constructed in the spill pool")
    vocabulary = int(manifest.model_config.vocab_size)

    def example_microbatches(
        sequences: int,
        accumulation: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[tuple[object, ...], ...]:
        shape = (1, sequences * sequence_length)
        lengths = (sequence_length,) * sequences
        return tuple(
            (
                torch.randint(vocabulary, shape, generator=generator),
                torch.randint(vocabulary, shape, generator=generator),
                lengths,
            )
            for _ in range(accumulation)
        )

    def step_microbatches(
        sequences: int, accumulation: int
    ) -> tuple[tuple[object, ...], ...]:
        """The same tokens for every step, budget and geometry.

        Every step trains on one batch, so the five steps overfit it and the
        loss visibly falls -- five steps of fresh data show nothing. It also
        makes the loss curve identical across budgets, so a divergence between
        two of them is a difference in what was computed rather than in what
        was sampled.
        """

        generator = torch.Generator().manual_seed(arguments.seed * 1_000_003)
        return example_microbatches(sequences, accumulation, generator=generator)

    with case.implementations():
        trained = False
        report = None
        # Opened before the branch: a run that chose its geometry by hand still
        # plans once per budget, and that is the same granular output a search
        # produces. Only the search is optional; the log is not.
        progress_log = run_root / "progress.log"
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = progress_log.open("w")
        plan_log = PlanLog(log_handle, sys.stdout)
        print(f"  progress log: {progress_log}   (tail -f it to follow)")
        if manual is not None:
            geometry = (manual, sequences_per_step // manual)
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
                report = plan_step_search(
                    case.model,
                    objective=case.objective,
                    optimizer=case.optimizer,
                    optimizer_state_init=optimizer_state_init,
                    hyperparams=("lr",),
                    example_microbatches=example_microbatches,
                    total_sequences_per_step=sequences_per_step,
                    sequence_length=sequence_length,
                    budgets=[
                        (budget, manifest.spill_budget_bytes)
                        for budget in search_budgets
                    ],
                    runtime=runtime,
                    execution="execution",
                    spill="spill",
                    min_tokens_per_microbatch=arguments.min_tokens_per_microbatch,
                    max_tokens_per_microbatch=arguments.max_tokens_per_microbatch,
                    artifact_store_dir=store,
                    plan_store_dir=plan_store,
                    verbose=True,
                    progress=progress,
                    force_fresh=False,
                    options=PressureFitOptions(deterministic=arguments.deterministic),
                    incumbents=arguments.incumbents,
                    orderings=(
                        None
                        if arguments.orderings == "factors"
                        else lambda accumulation: (
                            StepDataOrdering.depth_first(accumulation),
                        )
                    ),
                    resolution_options=arguments.resolution_options,
                    transfer_bandwidths=arguments.transfer_bandwidths,
                )
            print()
            print_search(report, tokens_per_step)
            report_path = run_root / "search.json"
            print(f"  search report: {report.save(report_path)}")
            note_host_memory(plan_log, "geometry search finished")
            print()
            for build in report.geometries:
                for name, value in build.phase_seconds.items():
                    if name != "total":
                        ledger[f"build: {name}"] = (
                            ledger.get(f"build: {name}", 0.0) + value
                        )
            ledger["search: pressurefit"] = report.total_search_seconds
            ledger["build: unattributed"] = max(
                0.0,
                report.total_build_seconds
                - sum(
                    value
                    for name, value in ledger.items()
                    if name.startswith("build: ")
                ),
            )
        if arguments.plots:
            if report is None:
                print("  plots need a search; skipped for a manual geometry")
            else:
                plot_dir = run_root / "figures"
                marker = time.perf_counter()
                written = plot_step_search(report, plot_dir)
                charge("figures", marker)
                print(rule("Figures"))
                for path in written:
                    print(f"  {path}")
                print()

        def run_one_budget(
            budget: int,
            geometry: tuple[int, int],
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

            nonlocal case, trained
            if trained:
                # Every budget starts from the same weights and a fresh
                # optimizer, on the same tokens per step, so its losses agree
                # with every other budget's bar reduction order: the run is a
                # correctness check as well as a measurement.
                marker = time.perf_counter()
                release_case_model(case, runtime=runtime)
                case = build_case(
                    manifest, seed=arguments.seed, runtime=runtime
                )
                charge("model construction", marker)
                plan_log.note("model and optimizer state reset for a comparable run")
            trained = True
            microbatches = example_microbatches(*geometry)
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
                    artifact_store_dir=store,
                    plan_store_dir=plan_store,
                    save_plan=True,
                    force_fresh=False,
                    overwrite_plan=False,
                    # The search policy the geometry search used, so the run
                    # plans the plan the search promised rather than missing
                    # the store and searching again under other options.
                    deterministic=arguments.deterministic,
                    resolution_options=arguments.resolution_options,
                    # The search's winning plan is the plan to beat, so the
                    # step executes what the search chose, or better, even
                    # when the replan's calibration or facts differ from the
                    # search's and the store cannot hand the plan back.
                    incumbent=incumbent,
                )
            charge("run planning", marker)
            note_host_memory(plan_log, f"planned {gib(budget)}")
            plan_report = training.plan_report
            print_breakdown(plan_report, tokens_per_step)

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
                    if step == 1 and plan_report.initial_pressurefit_result is not None:
                        note = "   (first-step plan)"
                    plan_log.write(
                        f"  step {step:>3}   {timing.cycle_seconds:7.3f} s"
                        f"   {tokens_per_step / timing.cycle_seconds:>10,.0f} tok/s"
                        f"   opening {timing.opening_delay_seconds:6.3f} s"
                        f"   loss {losses[step]:.4f}{note}\n"
                    )

            def run_step(step: int, *, traced: bool) -> Any:
                started = time.perf_counter()
                result = training(
                    step_microbatches(*geometry),
                    hyperparams={"lr": LEARNING_RATE},
                    runtime_trace=traced,
                )
                losses[step] = statistics.fmean(
                    float(value) for value in result.objectives
                )
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
            print()
            assert result.diagnostics is not None
            diagnostics = result.diagnostics.result()
            print()
            print_epilogue(diagnostics)
            trace_path = run_root / "steps" / f"{budget / _GIB:g}gib.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(diagnostics.as_dict(), indent=2, sort_keys=True)
            )
            print(f"  step diagnostics: {trace_path}")
            ledger["steps execution"] = ledger.get("steps execution", 0.0) + sum(
                hosts.values()
            )
            walls = [cycles[step] for step in sorted(cycles)]
            # The first step pays the plan's reconciliation of its initial
            # state; the measured step is the median of the ones after it.
            measured = walls[1:] if len(walls) > 1 else walls
            # The final StepResult's public outputs are caller-owned device
            # tensors; the runtime refuses to close while they are alive.
            del result
            gc.collect()
            training.close()
            step_summary = diagnostics.summary
            return RunBudgetOutcome(
                execution_budget_bytes=budget,
                simulated_step_seconds=plan_report.summary.simulated_step_seconds,
                measured_step_seconds=statistics.median(measured),
                step_seconds=tuple(walls),
                profiled_task_seconds=step_summary.profiled_task_seconds,
                real_task_seconds=step_summary.real_task_event_seconds,
                simulated_idle_seconds=(step_summary.simulated_inter_task_idle_seconds),
                real_idle_seconds=step_summary.real_inter_task_idle_seconds,
                prologue_seconds=(step_summary.real_initial_readiness_wait_seconds),
                terminal_tail_seconds=(step_summary.simulator_terminal_tail_seconds),
            )

        run_entries: list[RunBudgetOutcome] = []
        for budget in run_budgets:
            incumbent: AnnotatedProgramPlan | None = None
            if manual is not None:
                geometry = (manual, sequences_per_step // manual)
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
            run_entries.append(run_one_budget(budget, geometry, incumbent))
            # The frame that owned the closed plan is gone; collect what its
            # internals hold in cycles, so the host memory that plan still
            # occupies is free before the next budget plans.
            gc.collect()
            note_host_memory(plan_log, f"closed the {gib(budget)} plan")
        log_handle.close()
        if arguments.plots and run_entries:
            plot_dir = run_root / "figures"
            marker = time.perf_counter()
            written_run = plot_step_run(
                run_entries,
                plot_dir,
                tokens_per_step=tokens_per_step,
            )
            charge("figures", marker)
            for path in written_run:
                print(f"  figure: {path}")
            print()
        release_case_model(case, runtime=runtime)
    runtime.close()
    total = time.perf_counter() - command_started
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
