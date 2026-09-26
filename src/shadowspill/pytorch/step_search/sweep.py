"""One sweep: build each geometry, then plan every ordering at every budget.

The sweep is one object because the points, builds and best-so-far plans
accumulate across geometries, and because the budget loop hands each point
the winner of the budget below it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from os import PathLike
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import OutOfMemoryError, nn

from shadowspill.planner import StepDataOrdering
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.diagnostics import GraphPairOutcome
from shadowspill.pytorch.api import build_step_programs
from shadowspill.pytorch.runtime import Runtime
from shadowspill.search.planner import _Best, _Carried, _Planner
from shadowspill.search.refusals import _EXHAUSTED, _INFEASIBLE
from shadowspill.search.report import StepSearchGeometryBuild, StepSearchPoint
from shadowspill.step import StepProgram
from shadowspill.store import StoreMode

#: A point the planner refuses, for whatever reason it gives, is recorded and
#: the sweep goes on; ProblemPreparationError is one such RuntimeError.
_REJECTED = (RuntimeError,)


@dataclass(frozen=True, slots=True)
class _Build:
    """Everything one geometry's capture, lowering and profiling needs."""

    model: nn.Module
    objective: Any
    optimizer: Any
    hyperparams: Sequence[str]
    example_microbatches: Callable[[int, int], Sequence[Sequence[Any]]]
    runtime: Runtime
    execution: str
    spill: str
    optimizer_ordering: Literal["stage_interleaved", "tail"]
    verbose: bool
    artifact_store: str | PathLike[str] | None
    build_store: str | PathLike[str] | None
    build_store_mode: StoreMode
    export_bypass_key: str | None
    master_dtype: torch.dtype | None
    grad_dtype: torch.dtype | None
    round_accumulation_once: bool

    def programs(
        self,
        sequences: int,
        accumulation: int,
        orderings: Sequence[StepDataOrdering],
    ) -> tuple[tuple[StepProgram, ...], Exception | None]:
        """Build one program per ordering, or report what exhausted the device."""

        try:
            examples = self.example_microbatches(sequences, accumulation)
            return (
                build_step_programs(
                    self.model,
                    objective=self.objective,
                    optimizer=self.optimizer,
                    hyperparams=self.hyperparams,
                    example_inputs=examples,
                    runtime=self.runtime,
                    execution=self.execution,
                    spill=self.spill,
                    optimizer_ordering=self.optimizer_ordering,
                    orderings=orderings,
                    verbose=self.verbose,
                    artifact_store=self.artifact_store,
                    build_store=self.build_store,
                    build_store_mode=self.build_store_mode,
                    export_bypass_key=self.export_bypass_key,
                    master_dtype=self.master_dtype,
                    grad_dtype=self.grad_dtype,
                    round_accumulation_once=self.round_accumulation_once,
                ),
                None,
            )
        except Exception as error:
            if not _device_exhausted(error):
                raise
            # Exhaustion happens while profiling, which every ordering of
            # the geometry shares, so every ordering is infeasible.
            return (), error


@dataclass(slots=True)
class _Sweep:
    """The sweep's running state: what it has built, planned and found best."""

    ask: _Planner
    budgets: tuple[tuple[int, int], ...]
    incumbents: bool
    announce: Callable[[str], None]
    point_total: int
    builds: list[StepSearchGeometryBuild] = field(default_factory=list)
    points: list[StepSearchPoint] = field(default_factory=list)
    best_by_budget: dict[tuple[int, int], _Best] = field(default_factory=dict)
    point_index: int = 0

    def run(
        self,
        geometries: Sequence[tuple[int, int]],
        per_geometry: Sequence[tuple[StepDataOrdering, ...]],
        build: _Build,
    ) -> None:
        """Walk every geometry, and within it every ordering and budget."""

        for index, (sequences, accumulation) in enumerate(geometries, 1):
            shape = f"{sequences} x {accumulation}"
            orderings = per_geometry[index - 1]
            self.announce(
                f"geometry {index}/{len(geometries)}: building {shape}"
                f" ({len(orderings)} orderings)"
            )
            started = time.perf_counter()
            steps, exhausted = build.programs(sequences, accumulation, orderings)
            build_seconds = time.perf_counter() - started
            if exhausted is not None:
                self.announce(
                    f"geometry {index}/{len(geometries)}: {shape}"
                    " exhausted the device after"
                    f" {build_seconds:.1f} s;"
                    " every budget of every ordering is infeasible"
                )
            for position, ordering in enumerate(orderings, 1):
                name = f"{shape} {ordering.label}"
                if exhausted is not None:
                    self._refuse(sequences, accumulation, ordering, name, exhausted)
                    continue
                self.announce(
                    f"geometry {index}/{len(geometries)};"
                    f" ordering {position}/{len(orderings)}: built {name}"
                )
                self._record_build(
                    sequences,
                    accumulation,
                    ordering,
                    steps[position - 1],
                    # The geometry's wall clock is charged to its first
                    # ordering; the phases say how it split, the shared
                    # capture and profiling on the first program and each
                    # lowering on its own.
                    build_seconds if position == 1 else 0.0,
                )
                self._search(
                    steps[position - 1], sequences, accumulation, ordering, name
                )

    def _refuse(
        self,
        sequences: int,
        accumulation: int,
        ordering: StepDataOrdering,
        name: str,
        exhausted: Exception,
    ) -> None:
        """Record every budget of an ordering whose geometry never built."""

        for execution_budget, spill_budget in self.budgets:
            self.point_index += 1
            self.announce(
                f"point {self.point_index}/{self.point_total}: {name} @"
                f" {execution_budget >> 30} GiB -> infeasible"
            )
            self.points.append(
                StepSearchPoint(
                    sequences_per_microbatch=sequences,
                    accumulation_count=accumulation,
                    ordering=ordering,
                    execution_budget_bytes=execution_budget,
                    spill_budget_bytes=spill_budget,
                    status="infeasible",
                    makespan_seconds=None,
                    summary=None,
                    error=str(exhausted),
                    search_seconds=0.0,
                )
            )

    def _record_build(
        self,
        sequences: int,
        accumulation: int,
        ordering: StepDataOrdering,
        step: StepProgram,
        build_seconds: float,
    ) -> None:
        """Record what building this ordering's program cost and produced."""

        self.builds.append(
            StepSearchGeometryBuild(
                sequences_per_microbatch=sequences,
                accumulation_count=accumulation,
                ordering=ordering,
                step_program_digest=step.digest,
                build_seconds=build_seconds,
                phase_seconds=MappingProxyType(
                    {name: duration / 1e9 for name, duration in step.phase_timings_ns}
                ),
                transfer_bandwidths=step.problem.transfer_bandwidths,
            )
        )

    def _search(
        self,
        step: StepProgram,
        sequences: int,
        accumulation: int,
        ordering: StepDataOrdering,
        name: str,
    ) -> None:
        """Plan one program at every budget, ascending, carrying the plan to beat.

        Budgets ascend so the best plan found at a smaller budget is in hand
        for every larger one: a plan that fits in less memory fits in more,
        and the search answers with it unless it does strictly better.
        """

        carried: _Carried | None = None
        for execution_budget, spill_budget in sorted(self.budgets):
            self.point_index += 1
            started = time.perf_counter()
            status, makespan, summary, failure = "succeeded", None, None, None
            outcomes: tuple[GraphPairOutcome, ...] = ()
            inherited: int | None = None
            try:
                answer = self.ask.answer(
                    step.problem,
                    execution_budget,
                    spill_budget,
                    carried if self.incumbents else None,
                )
            except _EXHAUSTED as error:
                status, failure = "search_exhausted", str(error)
            except _INFEASIBLE as error:
                status, failure = "infeasible", str(error)
            except _REJECTED as error:
                status, failure = "rejected", str(error)
            else:
                makespan = answer.makespan_ns / 1e9
                summary = answer.summary
                outcomes = answer.outcomes
                found_at = (execution_budget, spill_budget)
                if carried is not None and answer.answered_with_incumbent:
                    inherited = carried.execution_budget_bytes
                    found_at = (inherited, carried.spill_budget_bytes)
                if carried is None or answer.makespan_ns < carried.makespan_ns:
                    carried = _Carried(*found_at, answer.makespan_ns, answer.plan)
                held = self.best_by_budget.get((execution_budget, spill_budget))
                if held is None or answer.makespan_ns < held.makespan_ns:
                    self.best_by_budget[(execution_budget, spill_budget)] = _Best(
                        answer.makespan_ns, step.problem, answer.plan
                    )
            self.announce(
                f"point {self.point_index}/{self.point_total}: {name} @"
                f" {execution_budget >> 30} GiB -> {status}"
                + (f" {makespan:.3f} s" if makespan is not None else "")
            )
            self.points.append(
                StepSearchPoint(
                    sequences_per_microbatch=sequences,
                    accumulation_count=accumulation,
                    ordering=ordering,
                    execution_budget_bytes=execution_budget,
                    spill_budget_bytes=spill_budget,
                    status=status,
                    makespan_seconds=makespan,
                    summary=summary,
                    error=failure,
                    search_seconds=time.perf_counter() - started,
                    graph_pair_selections=outcomes,
                    incumbent_budget_bytes=inherited,
                )
            )

    def winner_plans(self) -> dict[tuple[int, int], AnnotatedProgramPlan]:
        """Read back whole the plans a summary answered, for the run that follows.

        Every other plan the search touched stays in the store.
        """

        winners: dict[tuple[int, int], AnnotatedProgramPlan] = {}
        for budget, held in sorted(self.best_by_budget.items()):
            plan = held.plan
            if plan is None:
                started = time.perf_counter()
                plan = self.ask.plan(held.problem, *budget)
                self.announce(
                    f"winner @ {budget[0] >> 30} GiB: plan read back in"
                    f" {time.perf_counter() - started:.1f} s"
                )
            winners[budget] = plan
        return winners


def _device_exhausted(error: BaseException) -> bool:
    """Whether a build failed because the device ran out of memory.

    Profiling runs a task's real kernels, so the largest geometries can
    exhaust the device before any plan exists. The frontend wraps what a
    phase raised, chaining the original, so the exhaustion is found by
    walking the chain rather than by matching the outermost type.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OutOfMemoryError):
            return True
        current = current.__cause__ or current.__context__
    return False
