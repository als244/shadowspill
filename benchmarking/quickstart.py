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
        --search-budget-gib 10,12,16,20,24,28 \\
        --run-budget-gib 10,12,16,20,24,28 --spill-gib 112 --steps 5 \\
        --plots --plot-dir benchmarking/quickstart_plots/mlops_llama3

Every flag defaults to the model's retained qualification value, so
`python -m benchmarking.quickstart mlops_olmoe` searches and runs the
known cell.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.plots import plot_step_run, plot_step_search
from shadowspill.pytorch import Runtime, StepSearchReport, plan_step, plan_step_search
from tools.qualification.model_state import import_case_model, release_case_model
from workloads.full_model import build_case, manifest_for
from workloads.providers import ModelImplementation


def _budget_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


IDENTITIES = (
    "mlops_llama3",
    "mlops_qwen35",
    "mlops_olmoe",
    "pytorch_llama3",
    "pytorch_qwen35",
)
_GIB = 1 << 30


def rule(title: str) -> str:
    head = f"── {title} "
    return head + "─" * max(0, 68 - len(head))


def bar(fraction: float) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * 16)
    return "█" * filled + "░" * (16 - filled)


def gib(value: float) -> str:
    return f"{value / _GIB:.2f} GiB"


def gib_s(value: float) -> str:
    return f"{value / _GIB:.1f} GiB/s"


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
    print("  seqs/microbatch x accumulation, fastest simulated step wins")
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
            shape = f"{point.sequences_per_microbatch} x {point.accumulation_count}"
            if point.status == "succeeded" and point.makespan_seconds is not None:
                outcome = (
                    f"{point.makespan_seconds:8.3f} s"
                    f"   {tokens_per_step / point.makespan_seconds:>10,.0f} tok/s"
                )
            else:
                outcome = point.status
            print(f"  {mark} {shape:>9}   {outcome}")
    for sequences, accumulation, reason in report.skipped:
        print(f"    {sequences} x {accumulation:<4} skipped: {reason}")
    print(
        f"  builds {report.total_build_seconds:.1f} s across"
        f" {len(report.geometries)} geometries"
        f"   searches {report.total_search_seconds:.1f} s"
    )
    print()


def print_promise(report: Any, tokens: int) -> None:
    summary = report.summary
    simulated = summary.simulated_step_seconds
    extra = simulated - summary.unconstrained_step_seconds
    print(rule("The chosen plan's promise"))
    print(f"  simulated step   {simulated:8.3f} s   {tokens / simulated:>10,.0f} tok/s")
    print(
        f"  unconstrained    {summary.unconstrained_step_seconds:8.3f} s"
        f"   {tokens / summary.unconstrained_step_seconds:>10,.0f} tok/s"
        "   (cheapest graphs, no waiting)"
    )
    print()
    print(f"  where the extra {extra:.3f} s goes")
    for label, value in (
        ("extra recomputation", summary.recomputation_overhead_seconds),
        ("waiting between tasks", summary.idle_seconds),
        ("terminal writeback", summary.terminal_writeback_seconds),
    ):
        share = value / extra if extra > 0 else 0.0
        print(f"    {label:<22}{value:+8.3f} s  {bar(share)}  {share:6.1%}")
    print(
        f"  recomputation chosen for {summary.recompute_selection_count}"
        f" of {summary.selection_count} groups"
        f" ({summary.recompute_selection_fraction:.0%})"
    )
    chosen = summary.selected_candidate
    if chosen:
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
    print("                     (execution is the budget after fixed reservations)")
    fetch, evict = report.fetch_profile, report.evict_profile
    print(
        f"  fetch bandwidth    {gib_s(fetch.bandwidth_bytes_per_second)} assumed"
        f"   ({gib_s(fetch.solo_bandwidth_bytes_per_second)} solo)"
    )
    print(
        f"  evict bandwidth    {gib_s(evict.bandwidth_bytes_per_second)} assumed"
        f"   ({gib_s(evict.solo_bandwidth_bytes_per_second)} solo)"
    )
    print()


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
        "  between tasks    real"
        f" {summary.real_inter_task_readiness_wait_seconds:.3f} s waiting"
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
    worst = max(compute, key=lambda item: abs(item.duration_delta_seconds))
    print(f"  compute lane ({len(compute)} tasks), real minus simulated")
    print(deltas("starts   ", [item.start_delta_seconds for item in compute]))
    print(
        deltas(
            "durations",
            [item.duration_delta_seconds for item in compute],
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
            f" lane busy real {lane_summary.stream_busy_seconds:.3f} s"
            f" simulated {lane_summary.simulated_busy_seconds:.3f} s"
            + (
                f"; effective {effective / 2**30:.1f} GiB/s"
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
                    [item.duration_delta_seconds or 0.0 for item in measured],
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
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="where the search report JSON is saved for post-hoc analysis;"
        " defaults to benchmarking/quickstart_reports/<model>.json",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="defaults to benchmarking/quickstart_plots/<model>",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--artifact-store",
        type=Path,
        default=None,
        help="compile/profile cache; defaults to benchmarking/quickstart_store/<model>",
    )
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument(
        "--search-log",
        type=Path,
        default=None,
        help="plan-style progress log (planner phase lines + search"
        " progress, wall-clock stamped); defaults to"
        " benchmarking/quickstart_reports/<model>.search.log",
    )
    arguments = parser.parse_args()
    if arguments.steps < 1:
        parser.error("--steps must be at least 1")

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
    store = arguments.artifact_store or (
        Path("benchmarking/quickstart_store") / arguments.model
    )

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
    print()

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
    marker = time.perf_counter()
    case = build_case(manifest, seed=arguments.seed)
    charge("model construction", marker)
    vocabulary = int(manifest.model_config.vocab_size)

    def example_microbatches(
        sequences: int, accumulation: int
    ) -> tuple[tuple[object, ...], ...]:
        shape = (1, sequences * sequence_length)
        lengths = (sequence_length,) * sequences
        return tuple(
            (
                torch.randint(vocabulary, shape),
                torch.randint(vocabulary, shape),
                lengths,
            )
            for _ in range(accumulation)
        )

    with case.implementations():
        marker = time.perf_counter()
        case = import_case_model(case, runtime=runtime)
        charge("model import into the spill pool", marker)
        report = None
        plan_log: PlanLog | None = None
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
            search_log = arguments.search_log or (
                Path("benchmarking/quickstart_reports")
                / f"{arguments.model}.search.log"
            )
            search_log.parent.mkdir(parents=True, exist_ok=True)
            log_handle = search_log.open("w")
            plan_log = PlanLog(log_handle, sys.stdout)

            def progress(message: str) -> None:
                plan_log.note(message)

            print(f"  progress log: {search_log}   (tail -f it to follow)")
            with contextlib.redirect_stdout(plan_log):
                report = plan_step_search(
                    case.model,
                    objective=case.objective,
                    opt=case.optimizer,
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
                    verbose=True,
                    progress=progress,
                    force_fresh=arguments.force_fresh,
                )
            print()
            print_search(report, tokens_per_step)
            report_path = arguments.report_path or (
                Path("benchmarking/quickstart_reports") / f"{arguments.model}.json"
            )
            print(f"  search report: {report.save(report_path)}")
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
                plot_dir = arguments.plot_dir or (
                    Path("benchmarking/quickstart_plots") / arguments.model
                )
                marker = time.perf_counter()
                written = plot_step_search(report, plot_dir)
                charge("figures", marker)
                print(rule("Figures"))
                for path in written:
                    print(f"  {path}")
                print()

        run_entries: list[tuple[int, float, float]] = []
        for budget in run_budgets:
            if manual is not None:
                geometry = (manual, sequences_per_step // manual)
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
            print(rule(f"Run at execution {gib(budget)}"))
            print(
                f"  geometry {geometry[0]} sequences per microbatch"
                f" x {geometry[1]} accumulation rounds"
            )
            print()
            microbatches = example_microbatches(*geometry)
            walls: list[float] = []
            marker = time.perf_counter()
            plan_sink: Any = (
                contextlib.redirect_stdout(plan_log)
                if plan_log is not None
                else contextlib.nullcontext()
            )
            if plan_log is not None:
                plan_log.note(f"run planning at execution {gib(budget)}")
            with plan_sink:
                training = plan_step(
                    case.model,
                    objective=case.objective,
                    opt=case.optimizer,
                    example_inputs=microbatches,
                    runtime=runtime,
                    execution="execution",
                    spill="spill",
                    execution_budget=budget,
                    optimizer_ordering="stage_interleaved",
                    artifact_store_dir=store,
                    save_plan=True,
                    force_fresh=False,
                    overwrite_plan=False,
                )
            charge("run planning", marker)
            plan_report = training.plan_report
            print_promise(plan_report, tokens_per_step)

            def run_step(
                step: int,
                *,
                traced: bool,
                training: Any = training,
                microbatches: Any = microbatches,
                plan_report: Any = plan_report,
                walls: list[float] = walls,
            ) -> Any:
                started = time.perf_counter()
                result = training(microbatches, runtime_trace=traced)
                loss = float(result.objectives[-1])
                wall = time.perf_counter() - started
                walls.append(wall)
                note = ""
                if step == 1 and plan_report.initial_pressurefit_result is not None:
                    note = "   (first-step plan)"
                print(
                    f"  step {step:>3}   {wall:7.3f} s"
                    f"   {tokens_per_step / wall:>10,.0f} tok/s"
                    f"   loss {loss:.4f}{note}"
                )
                return result

            if arguments.steps > 1:
                print(rule("Steps"))
                for step in range(1, arguments.steps):
                    result = run_step(step, traced=False)
                print()
            print(rule("Traced step versus simulation"))
            result = run_step(arguments.steps, traced=True)
            assert result.diagnostics is not None
            diagnostics = result.diagnostics.result()
            print()
            print_epilogue(diagnostics)
            run_entries.append(
                (
                    budget,
                    plan_report.summary.simulated_step_seconds,
                    statistics.median(walls),
                )
            )
            ledger["steps execution"] = ledger.get("steps execution", 0.0) + sum(walls)
            # The final StepResult's public outputs are caller-owned device
            # tensors; the runtime refuses to close while they are alive.
            del result
            gc.collect()
            training.close()
        if plan_log is not None:
            log_handle.close()
        if arguments.plots and run_entries:
            plot_dir = arguments.plot_dir or (
                Path("benchmarking/quickstart_plots") / arguments.model
            )
            marker = time.perf_counter()
            written_run = plot_step_run(
                run_entries, plot_dir, tokens_per_step=tokens_per_step
            )
            charge("figures", marker)
            print(f"  figure: {written_run}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
