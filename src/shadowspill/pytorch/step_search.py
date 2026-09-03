"""Search every microbatch-by-accumulation split of one step, across budgets.

A caller who knows how many sequences one optimizer step must consume
rarely knows which split of that total into microbatches and accumulation
rounds plans best. :func:`plan_step_search` answers by planning all of
them: it captures, profiles, and lowers one :class:`StepProgram` per
distinct geometry — expensive work the artifact store deduplicates by
structural digest, so each unique microbatch shape compiles and profiles
once — then runs the PressureFit search for every geometry under every
requested budget pair. It executes nothing and returns reports only;
running a winner afterward is one ordinary :func:`plan_step` call at the
chosen geometry, warm against the same store.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from torch import OutOfMemoryError, nn

from shadowspill.errors import (
    PlanInfeasibleError,
    PlanSearchExhaustedError,
)
from shadowspill.planner import (
    PressureFitInfeasibleError,
    PressureFitOptions,
    PressureFitSearchExhaustedError,
    pressurefit_program,
)
from shadowspill.planner.diagnostics.plan import (
    PlanSummary,
    summarize_selected_plan,
)
from shadowspill.planner.program_inputs import TransferBandwidths
from shadowspill.pytorch.api import make_step_program
from shadowspill.pytorch.runtime_adapter.runtime import Runtime
from shadowspill.schema import artifact_schema
from shadowspill.simulator import SimulationInfeasibleError

_INFEASIBLE = (
    PressureFitInfeasibleError,
    PlanInfeasibleError,
    SimulationInfeasibleError,
)
_EXHAUSTED = (PressureFitSearchExhaustedError, PlanSearchExhaustedError)


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
class StepSearchPoint:
    """One geometry under one budget pair, with its search outcome."""

    sequences_per_microbatch: int
    accumulation_count: int
    execution_budget_bytes: int
    spill_budget_bytes: int
    status: str
    makespan_seconds: float | None
    summary: PlanSummary | None
    error: str | None
    search_seconds: float


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
    step_program_digest: str
    build_seconds: float
    phase_seconds: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class StepSearchReport:
    """Every geometry-by-budget outcome of one geometry search."""

    total_sequences_per_step: int
    sequence_length: int
    budgets: tuple[tuple[int, int], ...]
    geometries: tuple[StepSearchGeometryBuild, ...]
    points: tuple[StepSearchPoint, ...]
    skipped: tuple[tuple[int, int, str], ...]

    @property
    def tokens_per_step(self) -> int:
        return self.total_sequences_per_step * self.sequence_length

    @property
    def total_build_seconds(self) -> float:
        """Wall time spent capturing, profiling, and lowering geometries."""

        return sum(item.build_seconds for item in self.geometries)

    @property
    def total_search_seconds(self) -> float:
        """Wall time spent in the PressureFit search across every point."""

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
            "geometries": [
                {
                    "sequences_per_microbatch": item.sequences_per_microbatch,
                    "accumulation_count": item.accumulation_count,
                    "step_program_digest": item.step_program_digest,
                    "build_seconds": item.build_seconds,
                    "phase_seconds": dict(item.phase_seconds),
                }
                for item in self.geometries
            ],
            "points": [
                {
                    "sequences_per_microbatch": item.sequences_per_microbatch,
                    "accumulation_count": item.accumulation_count,
                    "execution_budget_bytes": item.execution_budget_bytes,
                    "spill_budget_bytes": item.spill_budget_bytes,
                    "status": item.status,
                    "makespan_seconds": item.makespan_seconds,
                    "summary": (
                        None if item.summary is None else item.summary.as_dict()
                    ),
                    "error": item.error,
                    "search_seconds": item.search_seconds,
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


def plan_step_search(
    model: nn.Module,
    *,
    objective: Any,
    opt: Any,
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
    options: PressureFitOptions | None = None,
    minimum_object_bytes_evict_eligible: int = 1 << 20,
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
    artifact_store_dir: str | PathLike[str] | None = None,
    verbose: bool = False,
    progress: Callable[[str], None] | None = None,
    force_fresh: bool = False,
    implementation_revision: str | None = None,
) -> StepSearchReport:
    """Plan every admitted geometry under every budget; execute nothing.

    ``example_microbatches(sequences, accumulation)`` supplies the example
    inputs for one geometry — structure is what matters, values are not.
    ``transfer_bandwidths`` overrides the calibration each step program
    embeds from the runtime; leave it unset to plan against the measured
    routes. ``options`` selects the search policy;
    ``minimum_object_bytes_evict_eligible`` applies on top of it with the same
    meaning and default as :func:`plan_step`. Failures are outcomes, not
    errors: a geometry-budget point that
    proves infeasible or exhausts its search budget is reported with that
    status while the search continues. A geometry whose build exhausts the
    device -- profiling runs real kernels, so the largest microbatch can --
    reports every one of its budgets ``infeasible`` with the exhaustion as
    the point's error, and the search moves to the next geometry; that
    geometry contributes no build to the report, because it produced no
    program. ``progress`` is called with a short line at every geometry and
    point boundary; ``verbose`` additionally forwards each planning call's
    own phase reporting.
    """

    def announce(message: str) -> None:
        if progress is not None:
            progress(message)

    if not budgets:
        raise ValueError("at least one (execution, spill) budget is required")
    options = replace(
        options or PressureFitOptions(),
        minimum_object_bytes_evict_eligible=minimum_object_bytes_evict_eligible,
    )
    geometries, skipped = search_geometries(
        total_sequences_per_step,
        sequence_length=sequence_length,
        min_tokens_per_microbatch=min_tokens_per_microbatch,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
    )
    builds: list[StepSearchGeometryBuild] = []
    points: list[StepSearchPoint] = []
    point_total = len(geometries) * len(budgets)
    point_index = 0
    for geometry_index, (sequences, accumulation) in enumerate(geometries, 1):
        announce(
            f"geometry {geometry_index}/{len(geometries)}: building"
            f" {sequences} x {accumulation}"
        )
        build_started = time.perf_counter()
        try:
            examples = example_microbatches(sequences, accumulation)
            step = make_step_program(
                model,
                objective=objective,
                opt=opt,
                example_inputs=examples,
                runtime=runtime,
                execution=execution,
                spill=spill,
                optimizer_ordering=optimizer_ordering,
                verbose=verbose,
                artifact_store_dir=artifact_store_dir,
                save_plan=True,
                force_fresh=force_fresh,
                implementation_revision=implementation_revision,
            )
        except Exception as error:
            if not _device_exhausted(error):
                raise
            announce(
                f"geometry {geometry_index}/{len(geometries)}: {sequences} x"
                f" {accumulation} exhausted the device after"
                f" {time.perf_counter() - build_started:.1f} s;"
                " every budget is infeasible"
            )
            for execution_budget, spill_budget in budgets:
                point_index += 1
                announce(
                    f"point {point_index}/{point_total}: {sequences} x"
                    f" {accumulation} @ {execution_budget >> 30} GiB ->"
                    " infeasible"
                )
                points.append(
                    StepSearchPoint(
                        sequences_per_microbatch=sequences,
                        accumulation_count=accumulation,
                        execution_budget_bytes=execution_budget,
                        spill_budget_bytes=spill_budget,
                        status="infeasible",
                        makespan_seconds=None,
                        summary=None,
                        error=str(error),
                        search_seconds=0.0,
                    )
                )
            continue
        announce(
            f"geometry {geometry_index}/{len(geometries)}: built"
            f" {sequences} x {accumulation} in"
            f" {time.perf_counter() - build_started:.1f} s"
        )
        builds.append(
            StepSearchGeometryBuild(
                sequences_per_microbatch=sequences,
                accumulation_count=accumulation,
                step_program_digest=step.digest,
                build_seconds=time.perf_counter() - build_started,
                phase_seconds=MappingProxyType(
                    {name: duration / 1e9 for name, duration in step.phase_timings_ns}
                ),
            )
        )
        for execution_budget, spill_budget in budgets:
            point_index += 1
            search_started = time.perf_counter()
            status, makespan, summary, failure = "succeeded", None, None, None
            try:
                plan = pressurefit_program(
                    step.recurrent,
                    execution_budget=execution_budget,
                    spill_budget=spill_budget,
                    transfer_bandwidths=transfer_bandwidths,
                    options=options,
                    artifact_store_dir=artifact_store_dir,
                    verbose=verbose,
                    save_plan=True,
                    force_fresh=force_fresh,
                    overwrite_plan=False,
                )
            except _EXHAUSTED as error:
                status, failure = "search_exhausted", str(error)
            except _INFEASIBLE as error:
                status, failure = "infeasible", str(error)
            else:
                makespan = plan.simulation.makespan_ns / 1e9
                summary = summarize_selected_plan(plan.result)
            announce(
                f"point {point_index}/{point_total}: {sequences} x"
                f" {accumulation} @ {execution_budget >> 30} GiB -> {status}"
                + (f" {makespan:.3f} s" if makespan is not None else "")
            )
            points.append(
                StepSearchPoint(
                    sequences_per_microbatch=sequences,
                    accumulation_count=accumulation,
                    execution_budget_bytes=execution_budget,
                    spill_budget_bytes=spill_budget,
                    status=status,
                    makespan_seconds=makespan,
                    summary=summary,
                    error=failure,
                    search_seconds=time.perf_counter() - search_started,
                )
            )
    return StepSearchReport(
        total_sequences_per_step=total_sequences_per_step,
        sequence_length=sequence_length,
        budgets=tuple(budgets),
        geometries=tuple(builds),
        points=tuple(points),
        skipped=skipped,
    )
