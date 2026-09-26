"""The whole record of one planning call."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.ir import (
    AliasGroupSpec,
    ExecutionPlan,
    MemoryAction,
    ShadowSpillProgram,
    TaskProfile,
    shared_residency_footprint,
)
from shadowspill.planner.result import ProgramPlanResult
from shadowspill.runtime.topology import TransferCapabilities, TransferProfile
from shadowspill.step import StepDataOrdering

from ...search import SearchOptions
from .diagnostics import PlanDiagnostics
from .summary import PlanSummary, summarize_selected_plan


@dataclass(frozen=True, slots=True)
class PlanReport:
    """Immutable planning, profiling, schedule, and physical-admission evidence."""

    mode: str
    capture_identity: str
    execution_plan: ExecutionPlan
    task_profiles: tuple[TaskProfile, ...]
    transfer_actions: tuple[MemoryAction, ...]
    transfer_bytes_evicted: int
    transfer_bytes_fetched: int
    profile_unique_keys: int
    profile_cache_hits: int
    profile_cache_misses: int
    allocation_probe_seeds: int
    allocation_probe_repetitions: int
    profiling_provenance: tuple[str, ...]
    phase_timings_ns: tuple[tuple[str, int], ...]
    diagnostics: PlanDiagnostics
    execution_pool: str
    spill_pool: str
    execution_budget_bytes: int
    spill_budget_bytes: int
    requested_dynamic_scratch_reserve_bytes: int
    execution_device: int
    transfer_capabilities: TransferCapabilities
    optimizer_ordering: str | None = None
    #: How the step walked its microbatches, or `None` for a forward plan.
    data_ordering: StepDataOrdering | None = None
    #: The resolution options the plan was searched over, as exact fractions
    #: of the flexible groups recomputing, or `None` for a forward plan.
    search_options: SearchOptions | None = None
    planned_program_cache_hits: int = 0
    planned_program_cache_misses: int = 0
    fixed_slab_bytes: int = 0
    captured_stage_count: int = 0
    aot_unique_stage_contracts: int = 0
    aot_graph_pair_cache_hits: int = 0
    aot_graph_pair_cache_misses: int = 0
    search_results: tuple[ProgramPlanResult, ...] = ()

    @property
    def program(self) -> ShadowSpillProgram:
        """Canonical ShadowSpillProgram handed straight to the search."""

        return self.execution_plan.program

    @property
    def search_result(self) -> ProgramPlanResult:
        """The search call boundary and selected result for the plan."""

        if not self.search_results:
            raise RuntimeError("PlanReport does not contain search evidence")
        return self.search_results[-1]

    @property
    def predicted_device_peak_bytes(self) -> int:
        return self.execution_plan.prediction.device_peak_bytes

    @property
    def predicted_spill_peak_bytes(self) -> int:
        return self.execution_plan.prediction.spill_peak_bytes

    @property
    def predicted_makespan_ns(self) -> int:
        return self.execution_plan.prediction.makespan_ns

    @property
    def summary(self) -> PlanSummary:
        """The selected plan's promise as one derived object."""

        return summarize_selected_plan(
            self.search_result, phase_timings_ns=self.phase_timings_ns
        )

    @property
    def shared_aliases(self) -> tuple[AliasGroupSpec, ...]:
        """Runtime-global aliases charged outside this callable's schedule."""

        return tuple(
            item for item in self.program.alias_groups if item.shared_residency
        )

    @property
    def shared_execution_bytes(self) -> int:
        """Execution-pool bytes retained by shared runtime objects."""

        footprint = shared_residency_footprint(self.program)
        return footprint.for_device(self.program.devices[0].device_id)

    @property
    def shared_spill_bytes(self) -> int:
        """Spill-pool bytes retained by shared runtime objects."""

        return shared_residency_footprint(self.program).spill_bytes

    @property
    def callable_execution_budget_bytes(self) -> int:
        """Execution capacity remaining for this callable's admitted layout."""

        return self.execution_budget_bytes - self.shared_execution_bytes

    @property
    def callable_spill_budget_bytes(self) -> int:
        """Spill capacity remaining for this callable's movable objects."""

        return self.spill_budget_bytes - self.shared_spill_bytes

    @property
    def fetch_profile(self) -> TransferProfile:
        """Measured spill-to-execution route consumed by this plan."""

        return self.transfer_capabilities.route(self.spill_pool, self.execution_pool)

    @property
    def evict_profile(self) -> TransferProfile:
        """Measured execution-to-spill route consumed by this plan."""

        return self.transfer_capabilities.route(self.execution_pool, self.spill_pool)
