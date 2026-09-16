"""What a step search answered: one point per geometry, budget and walk."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from types import MappingProxyType

from shadowspill.planner import (
    SearchOptions,
    StepDataOrdering,
)
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from shadowspill.planner.diagnostics import (
    GraphPairOutcome,
)
from shadowspill.planner.diagnostics.plan import (
    PlanSummary,
)
from shadowspill.planner.program_inputs import (
    TransferBandwidths,
)
from shadowspill.planner.serialization import (
    _integer,
    _list,
    _mapping,
    _number,
    _optional_integer,
    _optional_number,
    _optional_string,
    _string,
)
from shadowspill.schema import artifact_schema

# a point the planner refuses, for whatever reason it gives, is recorded and
# the sweep goes on; ProblemPreparationError is one such RuntimeError
_REJECTED = (RuntimeError,)


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

    @classmethod
    def from_dict(cls, value: object, path: str) -> StepSearchPoint:
        """Read back one point as :meth:`StepSearchReport.to_dict` wrote it."""

        record = _mapping(value, path)
        summary = record.get("summary")
        selections = _list(
            record.get("graph_pair_selections", []),
            f"{path}.graph_pair_selections",
        )
        return cls(
            sequences_per_microbatch=_integer(
                record["sequences_per_microbatch"],
                f"{path}.sequences_per_microbatch",
            ),
            accumulation_count=_integer(
                record["accumulation_count"], f"{path}.accumulation_count"
            ),
            ordering=StepDataOrdering.from_dict(record["ordering"], f"{path}.ordering"),
            execution_budget_bytes=_integer(
                record["execution_budget_bytes"],
                f"{path}.execution_budget_bytes",
            ),
            spill_budget_bytes=_integer(
                record["spill_budget_bytes"], f"{path}.spill_budget_bytes"
            ),
            status=_string(record["status"], f"{path}.status"),
            makespan_seconds=_optional_number(
                record.get("makespan_seconds"), f"{path}.makespan_seconds"
            ),
            summary=(
                None
                if summary is None
                else PlanSummary.from_dict(summary, f"{path}.summary")
            ),
            error=_optional_string(record.get("error"), f"{path}.error"),
            search_seconds=_number(
                record.get("search_seconds", 0.0), f"{path}.search_seconds"
            ),
            graph_pair_selections=tuple(
                GraphPairOutcome.from_dict(
                    item, f"{path}.graph_pair_selections[{index}]"
                )
                for index, item in enumerate(selections)
            ),
            incumbent_budget_bytes=_optional_integer(
                record.get("incumbent_budget_bytes"),
                f"{path}.incumbent_budget_bytes",
            ),
        )


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

    @classmethod
    def from_dict(cls, value: object, path: str) -> StepSearchGeometryBuild:
        """Read back one geometry as :meth:`StepSearchReport.to_dict` wrote it."""

        record = _mapping(value, path)
        rates = record.get("transfer_bandwidths")
        phases = _mapping(record.get("phase_seconds", {}), f"{path}.phase_seconds")
        return cls(
            sequences_per_microbatch=_integer(
                record["sequences_per_microbatch"],
                f"{path}.sequences_per_microbatch",
            ),
            accumulation_count=_integer(
                record["accumulation_count"], f"{path}.accumulation_count"
            ),
            ordering=StepDataOrdering.from_dict(record["ordering"], f"{path}.ordering"),
            step_program_digest=_string(
                record["step_program_digest"], f"{path}.step_program_digest"
            ),
            build_seconds=_number(record["build_seconds"], f"{path}.build_seconds"),
            phase_seconds=MappingProxyType(
                {
                    key: _number(item, f"{path}.phase_seconds.{key}")
                    for key, item in phases.items()
                }
            ),
            transfer_bandwidths=(
                None
                if rates is None
                else TransferBandwidths.from_value(rates, f"{path}.transfer_bandwidths")
            ),
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

    @property
    def planned_lanes(self) -> TransferBandwidths | None:
        """The lanes every point was priced against.

        The pinned override when there was one, else what the first built
        geometry planned with, which is the calibration the runtime measured
        once for the run; `None` when nothing was built. A caller that runs a
        winner plans against this, so its plan is the one the search chose.
        """

        if self.transfer_bandwidths is not None:
            return self.transfer_bandwidths
        return next(
            (
                item.transfer_bandwidths
                for item in self.geometries
                if item.transfer_bandwidths is not None
            ),
            None,
        )

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

    @classmethod
    def from_dict(
        cls, value: object, path: str = "step_search_report"
    ) -> StepSearchReport:
        """Read back what :meth:`to_dict` wrote.

        The inverse makes a saved search a record rather than a write-only log:
        its figures can be redrawn, narrowed to a few budgets or geometries, or
        set beside another run's, without planning anything again.

        ``winner_plans`` is not carried. It is held in memory only -- the report
        on disk names each winner and the plan store holds the plan itself -- so
        a report read back answers every question about what was searched, and
        none about the plan objects a caller would run.
        """

        record = _mapping(value, path)
        options = record.get("search_options")
        rates = record.get("transfer_bandwidths")
        budgets = _list(record.get("budgets", []), f"{path}.budgets")
        geometries = _list(record.get("geometries", []), f"{path}.geometries")
        points = _list(record.get("points", []), f"{path}.points")
        skipped = _list(record.get("skipped", []), f"{path}.skipped")
        return cls(
            total_sequences_per_step=_integer(
                record["total_sequences_per_step"],
                f"{path}.total_sequences_per_step",
            ),
            sequence_length=_integer(
                record["sequence_length"], f"{path}.sequence_length"
            ),
            budgets=tuple(
                (
                    _integer(item[0], f"{path}.budgets[{index}][0]"),
                    _integer(item[1], f"{path}.budgets[{index}][1]"),
                )
                for index, item in enumerate(budgets)
            ),
            geometries=tuple(
                StepSearchGeometryBuild.from_dict(item, f"{path}.geometries[{index}]")
                for index, item in enumerate(geometries)
            ),
            points=tuple(
                StepSearchPoint.from_dict(item, f"{path}.points[{index}]")
                for index, item in enumerate(points)
            ),
            skipped=tuple(
                (
                    _integer(item[0], f"{path}.skipped[{index}][0]"),
                    _integer(item[1], f"{path}.skipped[{index}][1]"),
                    _string(item[2], f"{path}.skipped[{index}][2]"),
                )
                for index, item in enumerate(skipped)
            ),
            search_options=(
                None
                if options is None
                else SearchOptions.from_dict(options, f"{path}.search_options")
            ),
            transfer_bandwidths=(
                None
                if rates is None
                else TransferBandwidths.from_value(rates, f"{path}.transfer_bandwidths")
            ),
        )

    @classmethod
    def load(cls, path: str | PathLike[str]) -> StepSearchReport:
        """Read a report back from the JSON :meth:`save` wrote."""

        return cls.from_dict(json.loads(Path(path).read_text()), str(path))

    def save(self, path: str | PathLike[str]) -> Path:
        """Write the report as JSON and return the path."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return target
