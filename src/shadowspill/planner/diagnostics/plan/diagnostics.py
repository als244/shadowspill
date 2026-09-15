"""What the search did, per resolved program, as the report keeps it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shadowspill.planner.diagnostics import PlanningDiagnostics
from shadowspill.schema import artifact_schema

from .compilation import (
    PlanCacheArtifact,
    PlanCompilerProfile,
    PlanPhaseTiming,
    PlanProfilingMetadata,
)
from .graphs import PlanUniqueStage
from .layout import PlanPhysicalLayout
from .stages import PlanTaskStage


@dataclass(frozen=True, slots=True)
class PlanDiagnostics:
    """Structured evidence describing one frontend planning call.

    Phase intervals are mutually exclusive, and ``measured_wall_time_ns``
    adds them up. The difference from ``total_wall_time_ns`` is the small
    remainder spent between measured intervals and constructing the immutable
    report, which is a subtraction rather than a stored field so that the two
    can never disagree.
    """

    phases: tuple[PlanPhaseTiming, ...]
    total_wall_time_ns: int
    profile_unique_keys: int
    profile_cache_hits: int
    profile_cache_misses: int
    allocation_probe_seeds: int
    allocation_probe_repetitions: int
    captured_stage_count: int
    aot_unique_stage_contracts: int
    aot_graph_pair_cache_hits: int
    aot_graph_pair_cache_misses: int
    planned_program_cache_hits: int
    planned_program_cache_misses: int
    task_stage_map: tuple[PlanTaskStage, ...] = ()
    unique_stages: tuple[PlanUniqueStage, ...] = ()
    compiler_phase_timings_ns: tuple[tuple[str, int], ...] = ()
    compiler_profiles: tuple[PlanCompilerProfile, ...] = ()
    store_directories: tuple[tuple[str, str], ...] = ()
    cache_artifacts: tuple[PlanCacheArtifact, ...] = ()
    profiling_metadata: tuple[PlanProfilingMetadata, ...] = ()
    search_runs: tuple[PlanningDiagnostics, ...] = ()
    physical_layouts: tuple[PlanPhysicalLayout, ...] = ()

    @property
    def measured_wall_time_ns(self) -> int:
        return sum(item.duration_ns for item in self.phases)

    def as_dict(self) -> dict[str, object]:
        selected_tasks = self.tasks
        return {
            "schema": artifact_schema("plan_diagnostics"),
            "phases": [item.as_dict() for item in self.phases],
            "measured_wall_time_ns": self.measured_wall_time_ns,
            "total_wall_time_ns": self.total_wall_time_ns,
            "profile": {
                "unique_keys": self.profile_unique_keys,
                "cache_hits": self.profile_cache_hits,
                "cache_misses": self.profile_cache_misses,
                "allocation_probe_seeds": self.allocation_probe_seeds,
                "allocation_probe_repetitions": (self.allocation_probe_repetitions),
            },
            "compiler": {
                "phases": [
                    {"name": name, "duration_ns": duration}
                    for name, duration in self.compiler_phase_timings_ns
                ],
                "measured_wall_time_ns": sum(
                    duration for _name, duration in self.compiler_phase_timings_ns
                ),
                "structural_contracts": {
                    item.structural_contract_key: item.as_dict()
                    for item in self.compiler_profiles
                },
            },
            "store_directories": dict(self.store_directories),
            "cache_artifacts": [item.as_dict() for item in self.cache_artifacts],
            "profiling_metadata": [item.as_dict() for item in self.profiling_metadata],
            "capture": {
                "stage_count": self.captured_stage_count,
                "aot_unique_stage_contracts": self.aot_unique_stage_contracts,
                "aot_graph_pair_cache_hits": self.aot_graph_pair_cache_hits,
                "aot_graph_pair_cache_misses": self.aot_graph_pair_cache_misses,
            },
            "recomputation": {
                "cache_hits": self.planned_program_cache_hits,
                "cache_misses": self.planned_program_cache_misses,
            },
            "search": [
                {
                    "run_index": index,
                    **item.to_dict(),
                }
                for index, item in enumerate(self.search_runs)
            ],
            "physical_layouts": [item.as_dict() for item in self.physical_layouts],
            "tasks": {
                execution_task_id: item.as_dict()
                for execution_task_id, item in selected_tasks.items()
            },
            "task_variants_by_ir_id": {
                item.task_id: item.as_dict() for item in self.task_stage_map
            },
            "unique_stages": [item.as_dict() for item in self.unique_stages],
        }

    @property
    def tasks(self) -> Mapping[str, PlanTaskStage]:
        """Selected tasks keyed by contiguous chronological execution identity."""

        return MappingProxyType(
            {
                item.execution_task_id: item
                for item in self.task_stage_map
                if item.selected and item.execution_task_id is not None
            }
        )

    def task(self, execution_task_id: str) -> PlanTaskStage:
        """Return selected-task information for ``execution_task_id``."""

        try:
            return self.tasks[execution_task_id]
        except KeyError:
            raise KeyError(execution_task_id) from None

    def task_by_ir_id(self, task_id: str) -> PlanTaskStage:
        """Return variant information by stable canonical IR task identity."""

        for item in self.task_stage_map:
            if item.task_id == task_id:
                return item
        raise KeyError(task_id)
