"""Model-independent search, admission, measured execution and diagnostics."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import statistics
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch

from shadowspill.diagnostics.occupancy import write_run_timelines
from shadowspill.memory import device, pinned_host, transfer_route
from shadowspill.planner import StepDataOrdering
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.plots import (
    RunBudgetOutcome,
    plot_step_run,
    plot_step_search,
    write_run_tables,
)
from shadowspill.pytorch import (
    Runtime,
    StepSearchReport,
    import_model_state,
    plan_step,
    plan_step_search,
    release_model_state,
)
from shadowspill.runtime.failures import RuntimeExecutionError
from shadowspill.training._model import initialize_model
from shadowspill.training.observations import StepObservations

from .options import profiling_policy, search_policy
from .reporting import (
    PlanLog,
    bar,
    gib,
    host_memory,
    host_memory_ceiling,
    note_host_memory,
    print_breakdown,
    print_epilogue,
    print_search,
    rule,
)
from .storage import Ledger, RunPaths

_GIB = 1 << 30


@dataclass(frozen=True)
class Request:
    label: str
    search_budgets: list[int]
    run_budgets: list[int]
    physical_capacity: int
    spill_budget: int
    device: torch.device
    manual: str | None = None
    remote_spill: tuple[str, int] | None = None
    external_headroom: int = 512 << 20


def open_runtime(
    request: Request,
    ledger: Ledger,
    *,
    control_group=None,
    host_headroom_gib=2,
    preparation_timeout=1800,
) -> Runtime:
    """The runtime, calibrated once here and reused by every geometry."""

    marker = time.perf_counter()
    capacity = request.spill_budget
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
        control_group=control_group,
        host_headroom_bytes=int(host_headroom_gib * _GIB),
        preparation_timeout=preparation_timeout,
        pools={
            "execution": device(
                physical_capacity=request.physical_capacity,
                device=request.device.index,
                external_headroom=request.external_headroom,
            ),
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
    fill is what is left after the accelerator problem and the external-memory
    headroom. Planning resolves that, so both phases are given the resolved
    figure -- the search used to take its budgets literally and rank against a
    slab the cap cannot hold, which made it promise plans the run could not
    reproduce.
    """

    execution_pool = runtime.pools["execution"]
    # CLI budgets are complete physical caps. Translate each by the same
    # runtime carve-out before passing logical slab limits to planning.
    fixed = (
        execution_pool.physical_capacity or execution_pool.capacity
    ) - execution_pool.capacity
    planned = {
        item: item - fixed
        for item in dict.fromkeys([*request.search_budgets, *request.run_budgets])
    }
    if any(value <= 0 for value in planned.values()):
        raise ValueError(
            f"requested execution budget leaves no slab after {fixed} fixed bytes"
        )
    return Budgets(request.search_budgets, request.run_budgets, planned)


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
        experiment: Mapping[str, Any],
    ) -> None:
        self.arguments = arguments
        self.request = request
        self.paths = paths
        self.budgets = budgets
        self.runtime = runtime
        self.ledger = ledger
        self.experiment = dict(experiment)
        self.candidates = dict(experiment["candidates"])
        if not self.candidates or any(
            not inputs for inputs in self.candidates.values()
        ):
            raise ValueError("each named candidate must contain microbatches")
        self.metadata = dict(experiment.get("metadata", {}))
        self.units_per_step = float(experiment.get("units_per_step", 1))
        self.unit_label = str(experiment.get("unit_label", "updates"))
        self.metadata.update(
            units_per_step=self.units_per_step, unit_label=self.unit_label
        )
        if self.units_per_step <= 0:
            raise ValueError("units_per_step must be positive")
        self.hyperparams = dict(experiment.get("hyperparams", {}))
        self.objective = experiment["objective"]
        self.optimizer = experiment["optimizer"]
        self.plan_options = dict(experiment.get("plan_options", {}))
        self.distributed = experiment.get("distributed")
        if self.distributed is None:
            self.distributed = self.plan_options.pop("distributed", None)
        elif "distributed" in self.plan_options:
            raise ValueError("specify distributed once, outside plan_options")
        if arguments.distributed and self.distributed is None:
            raise ValueError(
                "distributed quickstart factory must return a Distributed specification"
            )
        if self.distributed is not None and "model_factory" not in experiment:
            raise ValueError(
                "distributed quickstart needs model_factory to rebuild each budget"
            )
        self.policy = search_policy(arguments)
        self.profiling = profiling_policy(arguments)
        marker = time.perf_counter()
        self.model = self._build_model()
        ledger.charge("model construction and import", marker)
        note_host_memory(None, "model imported into the spill pool")
        self.trained = False
        self.closed = False
        self.report: StepSearchReport | None = None
        # Opened before the branch: a run that chose its geometry by hand still
        # plans once per budget, and that is the same granular output a search
        # produces. Only the search is optional; the log is not.
        progress_log = paths.root / "progress.log"
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = progress_log.open("w")
        self.plan_log = PlanLog(self.log_handle, sys.stdout)
        print(f"  progress log: {progress_log}   (tail -f it to follow)")

    def _build_model(self):
        factory = self.experiment.get("model_factory")
        model = (
            factory()
            if factory is not None
            else copy.deepcopy(self.experiment["model"])
        )
        model = initialize_model(model, initialize=self.experiment.get("initialize"))
        specification = (
            self.distributed(model) if callable(self.distributed) else self.distributed
        )
        symmetric = getattr(self.arguments, "symmetric_planning", None)
        if symmetric is not None:
            if specification is None and symmetric:
                raise ValueError(
                    "symmetric_planning requires a Distributed specification"
                )
            if specification is not None:
                specification = copy.copy(specification)
                specification.symmetric_planning = symmetric
        return import_model_state(
            model,
            runtime=self.runtime,
            pool="spill",
            release_source=True,
            distributed=specification,
        )

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
        plan_log, ledger = self.plan_log, self.ledger
        runtime = self.runtime
        if request.manual is not None:
            if request.manual not in self.candidates:
                raise ValueError(f"unknown candidate {request.manual!r}")
            print(f"  candidate chosen manually: {request.manual}")
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
                    self.model,
                    objective=self.objective,
                    optimizer=self.optimizer,
                    hyperparams=tuple(self.hyperparams),
                    candidates=self.candidates,
                    metadata=self.metadata,
                    execution_device=request.device,
                    **self.plan_options,
                    budgets=[
                        (budget, request.spill_budget) for budget in self.budgets.search
                    ],
                    runtime=runtime,
                    execution="execution",
                    spill="spill",
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
            print_search(report, self.units_per_step, self.unit_label)
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
        self, training: Any, plan_report: Any, candidate: str, *, budget: int
    ) -> _Steps:
        """Run the steps on one plan and say what each one took."""

        arguments, plan_log = self.arguments, self.plan_log
        units_per_step = self.units_per_step
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
                    f"   {units_per_step / timing.cycle_seconds:>10,.0f}"
                    f" {self.unit_label}/s"
                    f"   loss {losses[step]:.4f}{note}\n"
                )

        def run_step(step: int, *, traced: bool) -> Any:
            started = time.perf_counter()
            result = training(
                self.candidates[candidate],
                hyperparams=self.hyperparams,
                runtime_trace=traced,
            )
            # The objective supplies each microbatch's normalized contribution;
            # no data-unit or replica-group normalization is inferred here.
            # Host collection happens after completion, never inside a task.
            training.synchronize()
            observed = StepObservations.collect(result.objectives, result.metrics)
            record = {
                "step": step,
                "objective_loss": sum(observed.losses),
                "loss": sum(observed.losses),
            }
            reducer = self.experiment.get("metric_reducer")
            if reducer is not None:
                record.update(reducer(observed))
            losses[step] = float(record["loss"])
            with (self.paths.root / "step_metrics.jsonl").open("a") as output:
                output.write(json.dumps({"budget_bytes": budget, **record}) + "\n")
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
            f"   {len(cycles) * units_per_step / elapsed:>10,.0f} {self.unit_label}/s"
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
        candidate: str,
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
        runtime, ledger = self.runtime, self.ledger
        plan_log = self.plan_log
        if self.trained:
            # Every budget starts from the same weights and a fresh
            # optimizer, on the same tokens per step, so its losses agree
            # with every other budget's bar reduction order: the run is a
            # correctness check as well as a measurement.
            marker = time.perf_counter()
            self._release_model()
            self.model = self._build_model()
            ledger.charge("model construction", marker)
            plan_log.note("model and optimizer state reset for a comparable run")
        self.trained = True
        microbatches = self.candidates[candidate]
        marker = time.perf_counter()
        plan_sink: Any = contextlib.redirect_stdout(plan_log)
        plan_log.note(f"run planning at execution {gib(budget)}")
        with plan_sink:
            training = plan_step(
                self.model,
                objective=self.objective,
                optimizer=self.optimizer,
                hyperparams=tuple(self.hyperparams),
                example_inputs=microbatches,
                execution_device=request.device,
                **self.plan_options,
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
        try:
            ledger.charge("run planning", marker)
            note_host_memory(plan_log, f"planned {gib(budget)}")
            plan_report = training.plan_report
            print_breakdown(plan_report, self.units_per_step, self.unit_label)

            steps = self._run_steps(training, plan_report, candidate, budget=budget)
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
                recomputation_seconds=(
                    plan_report.summary.recomputation_overhead_seconds
                ),
                simulated_entry_delay_seconds=step_summary.simulated_entry_delay_seconds,
                real_entry_delay_seconds=step_summary.entry_delay_seconds,
                terminal_tail_seconds=step_summary.simulator_terminal_tail_seconds,
                real_terminal_tail_seconds=step_summary.real_terminal_tail_seconds,
            )
        finally:
            training.close()

    def run(self) -> list[RunBudgetOutcome]:
        """Every run budget, on the search's winner or the manual geometry."""

        request, plan_log = self.request, self.plan_log
        report = self.report
        run_entries: list[RunBudgetOutcome] = []
        for budget in self.budgets.run:
            incumbent: AnnotatedProgramPlan | None = None
            if request.manual is not None:
                candidate = request.manual
                ordering = StepDataOrdering.depth_first(len(self.candidates[candidate]))
            else:
                assert report is not None
                winner = report.winner(budget, request.spill_budget)
                if winner is None:
                    print(
                        f"  no geometry planned successfully at {gib(budget)};"
                        " skipping this run budget"
                    )
                    continue
                candidate = winner.candidate
                ordering = winner.ordering
                incumbent = report.winner_plans.get((budget, request.spill_budget))
            print(rule(f"Run at execution {gib(budget)}"))
            print(
                f"  candidate {candidate}, "
                f"{len(self.candidates[candidate])} microbatches; "
                f"{ordering.label}"
            )
            print()
            # A budget whose plan could not be admitted is a result about that
            # budget, not about the tour: an infeasible *plan* already skips with
            # a message, and a refused layout should read the same way rather
            # than discarding every budget after it. The figures for the budgets
            # that did run are worth more than a stack trace.
            try:
                run_entries.append(
                    self.run_one_budget(budget, candidate, ordering, incumbent)
                )
            except RuntimeExecutionError as error:
                if self.distributed is not None:
                    raise
                print(f"  {gib(budget)} could not be admitted: {error}")
                plan_log.note(f"{gib(budget)} refused admission; skipping")
                gc.collect()
            if run_entries:
                # The record is kept current rather than written once at the
                # end, so a run that stops early still leaves what it measured
                # and its figures can be redrawn from the tables.
                write_run_tables(
                    run_entries,
                    self.paths.root / "figures" / "raw_data",
                    units_per_step=self.units_per_step,
                    unit_label=self.unit_label,
                )
            # The frame that owned the closed plan is gone; collect what its
            # internals hold in cycles, so the host memory that plan still
            # occupies is free before the next budget plans.
            gc.collect()
            note_host_memory(plan_log, f"closed the {gib(budget)} plan")
        return run_entries

    def plot_run(self, run_entries: list[RunBudgetOutcome]) -> None:
        """The run's figures, then the model's state back to the pool."""

        arguments = self.arguments
        if arguments.plots and run_entries:
            plot_dir = self.paths.root / "figures"
            marker = time.perf_counter()
            written_run = plot_step_run(
                run_entries,
                plot_dir,
                units_per_step=self.units_per_step,
                unit_label=self.unit_label,
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

    def _release_model(self) -> None:
        release_model_state(self.model, runtime=self.runtime)
        cleanup = self.experiment.get("cleanup_model")
        if cleanup is not None:
            cleanup(self.model)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.log_handle.close()
            self._release_model()


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
    capacity = request.spill_budget
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
