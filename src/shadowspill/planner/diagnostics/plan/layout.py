"""Where the plan put its leases, and what a refused attempt reported."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.planner.diagnostics import PlanningDiagnostics

from .stages import PlanTaskMemoryEnvelope


@dataclass(frozen=True, slots=True)
class PlanFixedLayoutAttempt:
    """One capacity/layout trial made during admission."""

    requested_object_capacity_bytes: int
    effective_object_capacity_bytes: int
    required_bytes: int
    pool_capacity_bytes: int
    accepted: bool
    search_wall_time_ns: int
    physical_admission_wall_time_ns: int
    search_diagnostics: PlanningDiagnostics | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_object_capacity_bytes": self.requested_object_capacity_bytes,
            "effective_object_capacity_bytes": self.effective_object_capacity_bytes,
            "required_bytes": self.required_bytes,
            "pool_capacity_bytes": self.pool_capacity_bytes,
            "accepted": self.accepted,
            "search_wall_time_ns": self.search_wall_time_ns,
            "physical_admission_wall_time_ns": (self.physical_admission_wall_time_ns),
            "search_diagnostics": (
                None
                if self.search_diagnostics is None
                else self.search_diagnostics.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class PlanPhysicalLayout:
    """Complete fixed-layout admission summary for one execution phase."""

    plan_role: str
    strategy: str
    layout_digest: str
    program_digest: str
    schedule_digest: str
    facts_digest: str
    pool_capacity_bytes: int
    original_object_capacity_bytes: int
    effective_object_capacity_bytes: int
    fixed_slice_bytes: int
    resident_slice_bytes: int
    dynamic_reserve_bytes: int
    scratch_reserve_bytes: int
    required_bytes: int
    placement_count: int
    dynamic_lifetime_count: int
    reuse_dependency_count: int
    placements_by_purpose: tuple[tuple[str, int], ...]
    attempts: tuple[PlanFixedLayoutAttempt, ...]
    task_memory_envelopes: tuple[PlanTaskMemoryEnvelope, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "plan_role": self.plan_role,
            "strategy": self.strategy,
            "layout_digest": self.layout_digest,
            "program_digest": self.program_digest,
            "schedule_digest": self.schedule_digest,
            "facts_digest": self.facts_digest,
            "pool_capacity_bytes": self.pool_capacity_bytes,
            "original_object_capacity_bytes": self.original_object_capacity_bytes,
            "effective_object_capacity_bytes": self.effective_object_capacity_bytes,
            "fixed_slice_bytes": self.fixed_slice_bytes,
            "resident_slice_bytes": self.resident_slice_bytes,
            "dynamic_reserve_bytes": self.dynamic_reserve_bytes,
            "scratch_reserve_bytes": self.scratch_reserve_bytes,
            "required_bytes": self.required_bytes,
            "placement_count": self.placement_count,
            "dynamic_lifetime_count": self.dynamic_lifetime_count,
            "reuse_dependency_count": self.reuse_dependency_count,
            "placements_by_purpose": dict(self.placements_by_purpose),
            "attempts": [item.as_dict() for item in self.attempts],
            "task_memory_envelopes": [
                item.as_dict() for item in self.task_memory_envelopes
            ],
        }
