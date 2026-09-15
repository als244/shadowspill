"""A task stage in the plan, and the memory one task asks of the pools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlanTaskStage:
    """Direct task-to-stage and selected-variant lookup record."""

    task_id: str
    execution_ordinal: int | None
    execution_task_id: str | None
    semantic_name: str
    phase: str
    microbatch: int | None
    stage_occurrence_id: str | None
    unique_stage_id: str
    structural_contract_key: str
    semantic_contract_digest: str | None
    executable_contract_digest: str | None
    compiled_layout_digest: str | None
    graph_pair_variant: str | None
    chosen_graph_pair_variant: str | None
    selected: bool
    profile_compatibility_digest: str | None = None
    profiling_metadata_digest: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "execution_ordinal": self.execution_ordinal,
            "execution_task_id": self.execution_task_id,
            "semantic_name": self.semantic_name,
            "phase": self.phase,
            "microbatch": self.microbatch,
            "stage_occurrence_id": self.stage_occurrence_id,
            "unique_stage_id": self.unique_stage_id,
            "structural_contract_key": self.structural_contract_key,
            "semantic_contract_digest": self.semantic_contract_digest,
            "executable_contract_digest": self.executable_contract_digest,
            "compiled_layout_digest": self.compiled_layout_digest,
            "graph_pair_variant": self.graph_pair_variant,
            "chosen_graph_pair_variant": self.chosen_graph_pair_variant,
            "selected": self.selected,
            "profile_compatibility_digest": self.profile_compatibility_digest,
            "profiling_metadata_digest": self.profiling_metadata_digest,
        }


@dataclass(frozen=True, slots=True)
class PlanTaskMemoryEnvelope:
    """Fail-closed allocator limits admitted for one selected task."""

    task_id: str
    maximum_requested_allocation_bytes: int
    maximum_charged_allocation_bytes: int
    live_requested_allocation_limit_bytes: int
    live_charged_allocation_limit_bytes: int
    dynamic_scratch_maximum_allocation_bytes: int
    dynamic_scratch_live_limit_bytes: int
    allocation_contract_digest: str | None
    allocation_contract_operation_count: int
    allocation_path_digests: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "maximum_requested_allocation_bytes": (
                self.maximum_requested_allocation_bytes
            ),
            "maximum_charged_allocation_bytes": (self.maximum_charged_allocation_bytes),
            "live_requested_allocation_limit_bytes": (
                self.live_requested_allocation_limit_bytes
            ),
            "live_charged_allocation_limit_bytes": (
                self.live_charged_allocation_limit_bytes
            ),
            "dynamic_scratch_maximum_allocation_bytes": (
                self.dynamic_scratch_maximum_allocation_bytes
            ),
            "dynamic_scratch_live_limit_bytes": (self.dynamic_scratch_live_limit_bytes),
            "allocation_contract_digest": self.allocation_contract_digest,
            "allocation_contract_operation_count": (
                self.allocation_contract_operation_count
            ),
            "allocation_path_digests": list(self.allocation_path_digests),
        }
