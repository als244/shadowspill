"""Human-readable progress and diagnostics for generic quickstart experiments."""

from __future__ import annotations

import resource
import statistics
import time
from pathlib import Path
from typing import Any

from shadowspill.diagnostics.step import TaskRecord, TransferRecord
from shadowspill.pytorch import StepSearchReport

_GIB = 1 << 30
_GB = 1_000_000_000


def rule(title: str) -> str:
    head = f"── {title} "
    return head + "─" * max(0, 68 - len(head))


def bar(fraction: float) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * 16)
    return "█" * filled + "░" * (16 - filled)


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


def print_search(
    report: StepSearchReport, units_per_step: float, unit_label: str = "updates"
) -> None:
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
                f"{point.candidate} x {point.accumulation_count} {point.ordering.label}"
            )
            if point.status == "succeeded" and point.makespan_seconds is not None:
                outcome = (
                    f"{point.makespan_seconds:8.3f} s"
                    f"   {units_per_step / point.makespan_seconds:>10,.0f}"
                    f" {unit_label}/s"
                )
                if point.incumbent_budget_bytes is not None:
                    outcome += f"   plan from {gib(point.incumbent_budget_bytes)}"
            else:
                outcome = point.status
            print(f"  {mark} {shape:>16}   {outcome}")
    for sequences, accumulation, reason in report.metadata.get("skipped", ()):
        print(f"    {sequences} x {accumulation:<4} skipped: {reason}")
    print(
        f"  builds {report.total_build_seconds:.1f} s across"
        f" {len(report.geometries)} geometries"
        f"   searches {report.total_search_seconds:.1f} s"
    )
    print()


def print_breakdown(
    report: Any, units_per_step: float, unit_label: str = "updates"
) -> None:
    summary = report.summary
    simulated = summary.simulated_step_seconds
    print(rule("The chosen plan's breakdown"))
    print(
        f"  simulated step   {simulated:8.3f} s"
        f"   {units_per_step / simulated:>10,.0f} {unit_label}/s"
    )
    print(
        f"  unconstrained    {summary.unconstrained_step_seconds:8.3f} s"
        f"   {units_per_step / summary.unconstrained_step_seconds:>10,.0f}"
        f" {unit_label}/s"
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
        print("  traced invocation: unknown (incomplete transfer timestamps)")
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
        measured_text = "unknown" if real is None else f"{real:.3f} s"
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
        missing = lane_summary.transfers - len(measured)
        if missing:
            print(
                f"    timestamps unknown for {missing} transfers;"
                " deltas use timed transfers"
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
