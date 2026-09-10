"""Search every microbatch-by-accumulation split of one step, across budgets.

A caller who knows how many sequences one optimizer step must consume
rarely knows which split of that total into microbatches and accumulation
rounds plans best. :func:`plan_step_search` answers by planning all of
them: it captures, profiles, and lowers one :class:`StepProgram` per
distinct geometry — expensive work the artifact store deduplicates by
structural digest, so each unique microbatch shape compiles and profiles
once — then runs the search for every geometry under every
requested budget pair. It executes nothing and returns reports only;
running a winner afterward is one ordinary :func:`plan_step` call at the
chosen geometry, warm against the same store.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import OutOfMemoryError, nn

from shadowspill.errors import (
    PlanInfeasibleError,
    PlanSearchExhaustedError,
)
from shadowspill.planner import (
    SearchOptions,
    StepDataOrdering,
    plan_program,
)
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.diagnostics import INCUMBENT_CANDIDATE_ID
from shadowspill.planner.diagnostics.plan import (
    PlanSummary,
    summarize_selected_plan,
)
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.planner.result import ProgramPlanResult
from shadowspill.pytorch.api import build_step_program
from shadowspill.pytorch.runtime_adapter.runtime import Runtime
from shadowspill.schema import artifact_schema
from shadowspill.simulator import SimulationInfeasibleError
from shadowspill.store import StoreMode

_INFEASIBLE = (PlanInfeasibleError, SimulationInfeasibleError)
_EXHAUSTED = (PlanSearchExhaustedError,)
# a point the planner refuses, for whatever reason it gives, is recorded and
# the sweep goes on; ProblemPreparationError is one such RuntimeError
_REJECTED = (RuntimeError,)


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


def search_geometries(
    total_sequences_per_step: int,
    *,
    sequence_length: int,
    min_tokens_per_microbatch: int | None = None,
    max_tokens_per_microbatch: int | None = None,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, str], ...]]:
    """Split a step's sequence total into every microbatch/accumulation pair.

    Returns the admitted ``(sequences_per_microbatch, accumulation)`` pairs,
    largest microbatch first, and the pairs the optional token bounds
    skipped, each with its reason. Bounds are in tokens per microbatch, so
    they mean the same thing at every sequence length.
    """

    if total_sequences_per_step < 1:
        raise ValueError("total_sequences_per_step must be positive")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    admitted: list[tuple[int, int]] = []
    skipped: list[tuple[int, int, str]] = []
    for sequences in range(total_sequences_per_step, 0, -1):
        if total_sequences_per_step % sequences:
            continue
        accumulation = total_sequences_per_step // sequences
        tokens = sequences * sequence_length
        if min_tokens_per_microbatch is not None and tokens < min_tokens_per_microbatch:
            skipped.append(
                (
                    sequences,
                    accumulation,
                    f"{tokens} tokens per microbatch is below the minimum"
                    f" of {min_tokens_per_microbatch}",
                )
            )
            continue
        if max_tokens_per_microbatch is not None and tokens > max_tokens_per_microbatch:
            skipped.append(
                (
                    sequences,
                    accumulation,
                    f"{tokens} tokens per microbatch is above the maximum"
                    f" of {max_tokens_per_microbatch}",
                )
            )
            continue
        admitted.append((sequences, accumulation))
    return tuple(admitted), tuple(skipped)


@dataclass(frozen=True, slots=True)
class GraphPairOutcome:
    """The best plan the search found under one graph-pair selection.

    A search settles one selection at a time -- one choice of graph-pair
    option per graph-pair group -- and answers with the best plan across
    all of them. Keeping only that answer hides what the choice cost: whether
    the winner beat the alternatives by a hair or by a factor, and whether
    the others were slower or simply would not fit.

    Everything here is derivable without a second simulation.
    ``selected_compute_seconds`` and ``unconstrained_seconds`` come from the
    program's task profiles and this selection's own choices, so the split
    below needs only the makespan the search already recorded.
    """

    selection_id: str
    #: Groups this selection asked to recompute rather than save, out of the
    #: graph-pair groups the program has. This is the "level" the ladder
    #: of selections walks.
    recompute_groups: int
    group_count: int
    #: The makespan of this selection's best plan, or `None` when no policy
    #: produced a plan that fits.
    makespan_seconds: float | None
    #: Compute this selection asks for, and the cheapest any selection could
    #: ask for, both as the sum of the selected tasks' profiles.
    selected_compute_seconds: float
    unconstrained_seconds: float
    valid_candidate_count: int
    candidate_count: int
    #: What this selection's own best plan moves. Zero when it placed
    #: nothing, and on a plan read back from a store written before these
    #: were recorded.
    fetched_bytes: int
    evicted_bytes: int

    @property
    def recomputation_overhead_seconds(self) -> float:
        """Compute this selection spends above the cheapest possible."""

        return self.selected_compute_seconds - self.unconstrained_seconds

    @property
    def waiting_seconds(self) -> float | None:
        """Everything in the step that is not compute.

        Waiting between tasks and the terminal writeback together, because
        separating them needs the span of this selection's plan and the
        search records only its makespan.
        """

        if self.makespan_seconds is None:
            return None
        return self.makespan_seconds - self.selected_compute_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "selection_id": self.selection_id,
            "recompute_groups": self.recompute_groups,
            "group_count": self.group_count,
            "makespan_seconds": self.makespan_seconds,
            "selected_compute_seconds": self.selected_compute_seconds,
            "unconstrained_seconds": self.unconstrained_seconds,
            "valid_candidate_count": self.valid_candidate_count,
            "candidate_count": self.candidate_count,
            "fetched_bytes": self.fetched_bytes,
            "evicted_bytes": self.evicted_bytes,
        }


def _graph_pair_outcomes(
    result: ProgramPlanResult,
) -> tuple[GraphPairOutcome, ...]:
    """One record per graph-pair selection the search evaluated.

    The costs are read off the program rather than the simulator: a group's
    option names the tasks it activates, and a task names its profile, so
    both the cheapest total and this selection's total are sums over the
    same table.
    """

    program = result.program
    runtime_ns = {item.profile_id: item.runtime_ns for item in program.profiles}
    task_ns = {item.task_id: runtime_ns[item.profile_id] for item in program.tasks}
    variant_tasks: set[str] = set()
    option_cost: dict[tuple[str, str], int] = {}
    cheapest: dict[str, int] = {}
    for group in program.task_alternative_groups:
        for option in group.options:
            variant_tasks.update(option.active_task_ids)
            cost = sum(task_ns[task_id] for task_id in option.active_task_ids)
            option_cost[(group.group_id, option.option_id)] = cost
        cheapest[group.group_id] = min(
            option_cost[(group.group_id, option.option_id)] for option in group.options
        )
    fixed_ns = sum(
        task_ns[item.task_id]
        for item in program.tasks
        if item.task_id not in variant_tasks
    )
    floor_ns = fixed_ns + sum(cheapest.values())

    outcomes = []
    for problem in result.diagnostics.resolved_programs:
        chosen = {item.group_id: item.option_id for item in problem.choices}
        selected_ns = fixed_ns
        recomputing = 0
        for group_id, option_id in chosen.items():
            cost = option_cost[(group_id, option_id)]
            selected_ns += cost
            if cost > cheapest[group_id]:
                recomputing += 1
        statuses = [item.status for item in problem.candidate_evaluations]
        outcomes.append(
            GraphPairOutcome(
                selection_id=problem.selection_id,
                recompute_groups=recomputing,
                group_count=len(chosen),
                makespan_seconds=(
                    None
                    if problem.selected_makespan_ns is None
                    else problem.selected_makespan_ns / 1e9
                ),
                selected_compute_seconds=selected_ns / 1e9,
                unconstrained_seconds=floor_ns / 1e9,
                valid_candidate_count=sum(1 for item in statuses if item == "valid"),
                candidate_count=len(statuses),
                fetched_bytes=problem.fetched_bytes,
                evicted_bytes=problem.evicted_bytes,
            )
        )
    return tuple(sorted(outcomes, key=lambda item: item.recompute_groups))


@dataclass(frozen=True, slots=True)
class StepSearchPoint:
    """One geometry under one budget pair, with its search outcome."""

    sequences_per_microbatch: int
    accumulation_count: int
    #: How this point's program walked its microbatches.
    ordering: StepDataOrdering
    execution_budget_bytes: int
    spill_budget_bytes: int
    status: str
    makespan_seconds: float | None
    summary: PlanSummary | None
    error: str | None
    search_seconds: float
    #: Every graph-pair selection the search evaluated at this point, not
    #: only the one it answered with, ordered by how many groups recompute.
    #: Named for the graph-pair choices it makes rather than "selections",
    #: which in this codebase also names a candidate policy.
    graph_pair_selections: tuple[GraphPairOutcome, ...] = ()
    #: The smaller budget whose plan this point answered with, when the
    #: search was handed it and did not beat it; `None` when this point's
    #: own search won, or when no plan was handed in.
    incumbent_budget_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class StepSearchGeometryBuild:
    """The shared capture/profile/lowering work behind one geometry.

    ``phase_seconds`` breaks ``build_seconds`` down by frontend phase, in
    phase order — the geometry-search counterpart of
    ``PlanSummary.planning_phase_seconds``, which stays empty on a step-search
    point because a point runs only the search this build already paid
    everything else for.
    """

    sequences_per_microbatch: int
    accumulation_count: int
    ordering: StepDataOrdering
    step_program_digest: str
    build_seconds: float
    phase_seconds: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    #: The transfer calibration this build's program embeds, which every
    #: point of the geometry planned against unless the search overrode it.
    transfer_bandwidths: TransferBandwidths | None = None


@dataclass(frozen=True, slots=True)
class StepSearchReport:
    """Every geometry-by-budget outcome of one geometry search."""

    total_sequences_per_step: int
    sequence_length: int
    budgets: tuple[tuple[int, int], ...]
    geometries: tuple[StepSearchGeometryBuild, ...]
    points: tuple[StepSearchPoint, ...]
    skipped: tuple[tuple[int, int, str], ...]
    #: The resolution options every point was searched over, as exact
    #: fractions of the flexible groups recomputing.
    search_options: SearchOptions | None = None
    #: The calibration every point planned against instead of its program's
    #: own, or `None` when each program's embedded calibration was used; the
    #: per-geometry record says what that was.
    transfer_bandwidths: TransferBandwidths | None = None
    #: Each budget pair's winning plan, for a caller that will run the winner
    #: and wants the replan to start from it as the plan to beat. Held in
    #: memory only: the report on disk names the winner, the store holds it.
    winner_plans: Mapping[tuple[int, int], AnnotatedProgramPlan] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def tokens_per_step(self) -> int:
        return self.total_sequences_per_step * self.sequence_length

    @property
    def total_build_seconds(self) -> float:
        """Wall time spent capturing, profiling, and lowering geometries."""

        return sum(item.build_seconds for item in self.geometries)

    @property
    def total_search_seconds(self) -> float:
        """Wall time spent searching across every point."""

        return sum(item.search_seconds for item in self.points)

    def winner(
        self, execution_budget_bytes: int, spill_budget_bytes: int
    ) -> StepSearchPoint | None:
        """The fastest succeeded point under one budget pair, if any."""

        candidates = [
            point
            for point in self.points
            if point.execution_budget_bytes == execution_budget_bytes
            and point.spill_budget_bytes == spill_budget_bytes
            and point.status == "succeeded"
            and point.makespan_seconds is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda point: point.makespan_seconds or 0.0)

    @property
    def winners(self) -> tuple[StepSearchPoint, ...]:
        """One winner per requested budget pair, omitting budgets nobody won."""

        found = (self.winner(*budget) for budget in self.budgets)
        return tuple(point for point in found if point is not None)

    def to_dict(self) -> dict[str, object]:
        """The whole search as one JSON-ready record for post-hoc analysis."""

        return {
            "schema": artifact_schema("step_search_report"),
            "total_sequences_per_step": self.total_sequences_per_step,
            "sequence_length": self.sequence_length,
            "budgets": [list(item) for item in self.budgets],
            "search_options": (
                None if self.search_options is None else self.search_options.to_dict()
            ),
            "transfer_bandwidths": (
                None
                if self.transfer_bandwidths is None
                else self.transfer_bandwidths.to_dict()
            ),
            "geometries": [
                {
                    "sequences_per_microbatch": item.sequences_per_microbatch,
                    "accumulation_count": item.accumulation_count,
                    "ordering": item.ordering.to_dict(),
                    "ordering_label": item.ordering.label,
                    "step_program_digest": item.step_program_digest,
                    "build_seconds": item.build_seconds,
                    "phase_seconds": dict(item.phase_seconds),
                    "transfer_bandwidths": (
                        None
                        if item.transfer_bandwidths is None
                        else item.transfer_bandwidths.to_dict()
                    ),
                }
                for item in self.geometries
            ],
            "points": [
                {
                    "sequences_per_microbatch": item.sequences_per_microbatch,
                    "accumulation_count": item.accumulation_count,
                    "ordering": item.ordering.to_dict(),
                    "ordering_label": item.ordering.label,
                    "execution_budget_bytes": item.execution_budget_bytes,
                    "spill_budget_bytes": item.spill_budget_bytes,
                    "status": item.status,
                    "makespan_seconds": item.makespan_seconds,
                    "summary": (
                        None if item.summary is None else item.summary.as_dict()
                    ),
                    "error": item.error,
                    "search_seconds": item.search_seconds,
                    "graph_pair_selections": [
                        outcome.as_dict() for outcome in item.graph_pair_selections
                    ],
                    "incumbent_budget_bytes": item.incumbent_budget_bytes,
                }
                for item in self.points
            ],
            "skipped": [list(item) for item in self.skipped],
        }

    def save(self, path: str | PathLike[str]) -> Path:
        """Write the report as JSON and return the path."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return target


def default_orderings(accumulation: int) -> tuple[StepDataOrdering, ...]:
    """Every ``depth x breadth`` factor pair of the accumulation count.

    Depth-first first, so the order every step ran in before there was a
    choice is the first program built and the bound the rest are searched
    against; then wider and wider passes, down to one pass over every
    microbatch. The flags stay at their defaults throughout: the search does
    not toggle them, because pairing the loss won every cell it was measured
    in and the reversed walk cost nothing.
    """
    return tuple(
        StepDataOrdering(accumulation // breadth, breadth)
        for breadth in range(1, accumulation + 1)
        if accumulation % breadth == 0
    )


def plan_step_search(
    model: nn.Module,
    *,
    objective: Any,
    optimizer: Any,
    optimizer_state_init: Callable[[str, torch.Tensor, torch.nn.Parameter], None]
    | None = None,
    hyperparams: Sequence[str] = (),
    example_microbatches: Callable[[int, int], Sequence[Sequence[Any]]],
    total_sequences_per_step: int,
    sequence_length: int,
    budgets: Sequence[tuple[int, int]],
    runtime: Runtime,
    execution: str,
    spill: str,
    transfer_bandwidths: TransferBandwidths | None = None,
    min_tokens_per_microbatch: int | None = None,
    max_tokens_per_microbatch: int | None = None,
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
    orderings: Callable[[int], Sequence[StepDataOrdering]] | None = None,
    search_options: SearchOptions | None = None,
    incumbents: bool = True,
    artifact_store: str | PathLike[str] | None = None,
    build_store: str | PathLike[str] | None = None,
    plan_store: str | PathLike[str] | None = None,
    build_store_mode: StoreMode = "contribute",
    plan_store_mode: StoreMode = "contribute",
    verbose: bool = False,
    progress: Callable[[str], None] | None = None,
    implementation_revision: str | None = None,
) -> StepSearchReport:
    """Plan every admitted geometry under every budget; execute nothing.

    ``example_microbatches(sequences, accumulation)`` supplies the example
    inputs for one geometry — structure is what matters, values are not.
    ``transfer_bandwidths`` overrides the calibration each step program
    embeds from the runtime; leave it unset to plan against the measured
    routes. Either way the report records the calibration each geometry's
    program embeds, and the override when there was one, so two searches
    can be compared or one pinned to another's. ``search_options`` reaches
    every point unchanged, so a value set here is the value searched under.
    Failures are outcomes, not errors: a geometry-budget point that proves
    infeasible or exhausts its search budget is reported with that status
    while the search continues. A geometry whose build exhausts the
    device -- profiling runs real kernels, so the largest microbatch can --
    reports every one of its budgets ``infeasible`` with the exhaustion as
    the point's error, and the search moves to the next geometry; that
    geometry contributes no build to the report, because it produced no
    program. ``progress`` is called with a short line at every geometry and
    point boundary; ``verbose`` additionally forwards each planning call's
    own phase reporting.

    ``orderings`` maps a geometry's accumulation count to the
    :class:`StepDataOrdering` values to try for it; each ordering is lowered
    into its own program, sharing the geometry's capture and profiles, and
    planned under every budget. The default, :func:`default_orderings`, is
    every ``depth x breadth`` factor pair with the flags at their defaults.

    ``search_options`` names the resolutions every point is searched
    over, with the meaning it has for :func:`plan_step`; ``None`` is the
    library's default of every quarter. Options that are not valid are
    rejected before any geometry is built.

    ``incumbents`` hands each point the best plan found at a smaller budget
    of the same program, as the plan to beat: budgets are planned ascending,
    a plan that fits in less memory fits in more, and the search answers
    with it unless it does strictly better, so no program plans worse with
    more memory. A point that answered with a handed-in plan records the
    budget it came from as ``incumbent_budget_bytes``. ``False`` searches
    every point alone, which is how the two are compared.

    ``plan_store`` keeps every point's plan records apart from the
    artifact store, so a search can reuse another run's captures, profiles
    and lowering and still plan every point itself; ``None`` keeps them in
    the store, where a matching plan would be read back instead of planned.
    """

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    if not budgets:
        raise ValueError("at least one (execution, spill) budget is required")
    # The best plan seen under each budget pair, across every geometry and
    # ordering: what a run of the winner starts from.
    best_by_budget: dict[tuple[int, int], tuple[int, AnnotatedProgramPlan]] = {}
    chosen = search_options
    geometries, skipped = search_geometries(
        total_sequences_per_step,
        sequence_length=sequence_length,
        min_tokens_per_microbatch=min_tokens_per_microbatch,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
    )
    orderings_for = default_orderings if orderings is None else orderings
    per_geometry = [
        tuple(orderings_for(accumulation)) for _sequences, accumulation in geometries
    ]
    builds: list[StepSearchGeometryBuild] = []
    points: list[StepSearchPoint] = []
    point_total = sum(len(item) for item in per_geometry) * len(budgets)
    point_index = 0
    for geometry_index, (sequences, accumulation) in enumerate(geometries, 1):
        shape = f"{sequences} x {accumulation}"
        exhausted: Exception | None = None
        orderings_here = per_geometry[geometry_index - 1]
        for ordering_index, ordering in enumerate(orderings_here, 1):
            name = f"{shape} {ordering.label}"
            where = (
                f"geometry {geometry_index}/{len(geometries)};"
                f" ordering {ordering_index}/{len(orderings_here)}"
            )
            if exhausted is None:
                announce(f"{where}: building {name}")
                build_started = time.perf_counter()
                try:
                    examples = example_microbatches(sequences, accumulation)
                    step = build_step_program(
                        model,
                        objective=objective,
                        optimizer=optimizer,
                        optimizer_state_init=optimizer_state_init,
                        hyperparams=hyperparams,
                        example_inputs=examples,
                        runtime=runtime,
                        execution=execution,
                        spill=spill,
                        optimizer_ordering=optimizer_ordering,
                        depth=ordering.depth,
                        breadth=ordering.breadth,
                        reverse_breadth=ordering.reverse_breadth,
                        pair_loss=ordering.pair_loss,
                        verbose=verbose,
                        artifact_store=artifact_store,
                        build_store=build_store,
                        build_store_mode=build_store_mode,
                                    implementation_revision=implementation_revision,
                    )
                except Exception as error:
                    if not _device_exhausted(error):
                        raise
                    # Exhaustion happens while profiling, which every ordering
                    # of the geometry shares, so the rest would only repeat it.
                    exhausted = error
                    announce(
                        f"geometry {geometry_index}/{len(geometries)}: {shape}"
                        " exhausted the device after"
                        f" {time.perf_counter() - build_started:.1f} s;"
                        " every budget of every ordering is infeasible"
                    )
            if exhausted is not None:
                for execution_budget, spill_budget in budgets:
                    point_index += 1
                    announce(
                        f"point {point_index}/{point_total}: {name} @"
                        f" {execution_budget >> 30} GiB -> infeasible"
                    )
                    points.append(
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
                continue
            announce(
                f"{where}: built {name} in"
                f" {time.perf_counter() - build_started:.1f} s"
            )
            builds.append(
                StepSearchGeometryBuild(
                    sequences_per_microbatch=sequences,
                    accumulation_count=accumulation,
                    ordering=ordering,
                    step_program_digest=step.digest,
                    build_seconds=time.perf_counter() - build_started,
                    phase_seconds=MappingProxyType(
                        {
                            name_: duration / 1e9
                            for name_, duration in step.phase_timings_ns
                        }
                    ),
                    transfer_bandwidths=step.recurrent.transfer_bandwidths,
                )
            )
            # Budgets ascending, so the best plan found at a smaller budget
            # is in hand for every larger one: a plan that fits in less
            # memory fits in more, and the search answers with it unless it
            # does strictly better. `carried` is that plan and the budget it
            # was found at, which a point that answers with it records.
            carried: tuple[int, AnnotatedProgramPlan] | None = None
            for execution_budget, spill_budget in sorted(budgets):
                point_index += 1
                search_started = time.perf_counter()
                status, makespan, summary, failure = "succeeded", None, None, None
                outcomes: tuple[GraphPairOutcome, ...] = ()
                inherited: int | None = None
                try:
                    plan = plan_program(
                        step.recurrent,
                        execution_budget=execution_budget,
                        spill_budget=spill_budget,
                        transfer_bandwidths=transfer_bandwidths,
                        search_options=chosen,
                        incumbent=(
                            carried[1] if incumbents and carried is not None else None
                        ),
                        artifact_store=artifact_store,
                        plan_store=plan_store,
                        plan_store_mode=plan_store_mode,
                        verbose=verbose,
                                )
                except _EXHAUSTED as error:
                    status, failure = "search_exhausted", str(error)
                except _INFEASIBLE as error:
                    status, failure = "infeasible", str(error)
                except _REJECTED as error:
                    status, failure = "rejected", str(error)
                else:
                    makespan = plan.simulation.makespan_ns / 1e9
                    summary = summarize_selected_plan(plan.result)
                    outcomes = _graph_pair_outcomes(plan.result)
                    if (
                        carried is not None
                        and plan.result.diagnostics.selected_candidate_id
                        == INCUMBENT_CANDIDATE_ID
                    ):
                        inherited = carried[0]
                    if (
                        carried is None
                        or plan.simulation.makespan_ns
                        < carried[1].simulation.makespan_ns
                    ):
                        carried = (
                            execution_budget if inherited is None else inherited,
                            plan,
                        )
                    held = best_by_budget.get((execution_budget, spill_budget))
                    if held is None or plan.simulation.makespan_ns < held[0]:
                        best_by_budget[(execution_budget, spill_budget)] = (
                            plan.simulation.makespan_ns,
                            plan,
                        )
                announce(
                    f"point {point_index}/{point_total}: {name} @"
                    f" {execution_budget >> 30} GiB -> {status}"
                    + (f" {makespan:.3f} s" if makespan is not None else "")
                )
                points.append(
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
                        search_seconds=time.perf_counter() - search_started,
                        graph_pair_selections=outcomes,
                        incumbent_budget_bytes=inherited,
                    )
                )
    return StepSearchReport(
        total_sequences_per_step=total_sequences_per_step,
        sequence_length=sequence_length,
        budgets=tuple(budgets),
        geometries=tuple(builds),
        points=tuple(points),
        skipped=skipped,
        search_options=chosen,
        transfer_bandwidths=transfer_bandwidths,
        winner_plans=MappingProxyType(
            {budget: held[1] for budget, held in best_by_budget.items()}
        ),
    )
