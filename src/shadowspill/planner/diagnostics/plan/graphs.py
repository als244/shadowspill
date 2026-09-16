"""One captured graph as the report describes it: its storage, its allocations,
its representative inputs, and the pair it belongs to."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlanObjectFootprint:
    """One logical object view and the allocator extent containing it."""

    object_id: str
    alias_group_id: str
    role: str
    logical_size_bytes: int
    allocation_size_bytes: int
    offset_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "alias_group_id": self.alias_group_id,
            "role": self.role,
            "logical_size_bytes": self.logical_size_bytes,
            "allocation_size_bytes": self.allocation_size_bytes,
            "offset_bytes": self.offset_bytes,
        }


@dataclass(frozen=True, slots=True)
class PlanAllocationEvent:
    """One allocation/free point in a profiled graph's local timeline."""

    allocation_ordinal: int
    operation: str
    requested_bytes: int
    charged_bytes: int
    output_leaf_indices: tuple[int, ...]
    output_view_offsets: tuple[int, ...]
    reuses_ordinal: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "allocation_ordinal": self.allocation_ordinal,
            "operation": self.operation,
            "requested_bytes": self.requested_bytes,
            "charged_bytes": self.charged_bytes,
            "output_leaf_indices": list(self.output_leaf_indices),
            "output_view_offsets": list(self.output_view_offsets),
            "reuses_ordinal": self.reuses_ordinal,
        }


@dataclass(frozen=True, slots=True)
class PlanAllocationABIStep:
    """One pointer-free allocator operation required by a compiled task."""

    operation_index: int
    allocation_ordinal: int
    operation: str
    requested_bytes: int
    charged_bytes: int
    alignment_bytes: int
    output_leaf_indices: tuple[int, ...]
    mutation_input_positions: tuple[int, ...]
    persistent_after_task: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "operation_index": self.operation_index,
            "allocation_ordinal": self.allocation_ordinal,
            "operation": self.operation,
            "requested_bytes": self.requested_bytes,
            "charged_bytes": self.charged_bytes,
            "alignment_bytes": self.alignment_bytes,
            "output_leaf_indices": list(self.output_leaf_indices),
            "mutation_input_positions": list(self.mutation_input_positions),
            "persistent_after_task": self.persistent_after_task,
        }


@dataclass(frozen=True, slots=True)
class PlanStorageRoot:
    """Offline semantic root exposed in planning diagnostics."""

    root_id: int
    kind: str
    source_input: int | None
    producer_node: str | None
    producer_target: str | None
    producer_result: int | None
    minimum_span_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "root_id": self.root_id,
            "kind": self.kind,
            "source_input": self.source_input,
            "producer_node": self.producer_node,
            "producer_target": self.producer_target,
            "producer_result": self.producer_result,
            "minimum_span_bytes": self.minimum_span_bytes,
        }


@dataclass(frozen=True, slots=True)
class PlanOutputView:
    """Semantic output-view geometry exposed in planning diagnostics."""

    leaf_index: int
    root_id: int
    offset_bytes: int
    span_bytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    layout: str

    def as_dict(self) -> dict[str, object]:
        return {
            "leaf_index": self.leaf_index,
            "root_id": self.root_id,
            "offset_bytes": self.offset_bytes,
            "span_bytes": self.span_bytes,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "layout": self.layout,
        }


@dataclass(frozen=True, slots=True)
class PlanMutationBinding:
    """Schema- or Export-derived task-input mutation diagnostics."""

    input_position: int
    replacement_output_leaf: int | None
    producer_node: str
    producer_target: str
    argument_name: str

    def as_dict(self) -> dict[str, object]:
        return {
            "input_position": self.input_position,
            "replacement_output_leaf": self.replacement_output_leaf,
            "producer_node": self.producer_node,
            "producer_target": self.producer_target,
            "argument_name": self.argument_name,
        }


@dataclass(frozen=True, slots=True)
class PlanCompiledRoot:
    """Observed physical allocation for one semantic root."""

    root_id: int
    allocation_ordinal: int | None
    requested_bytes: int
    charged_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "root_id": self.root_id,
            "allocation_ordinal": self.allocation_ordinal,
            "requested_bytes": self.requested_bytes,
            "charged_bytes": self.charged_bytes,
        }


@dataclass(frozen=True, slots=True)
class PlanCompiledOutputView:
    """Observed physical binding for one returned object leaf."""

    leaf_index: int
    root_id: int
    allocation_ordinal: int | None
    offset_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "leaf_index": self.leaf_index,
            "root_id": self.root_id,
            "allocation_ordinal": self.allocation_ordinal,
            "offset_bytes": self.offset_bytes,
        }


@dataclass(frozen=True, slots=True)
class PlanRepresentativeInput:
    """Content-free value provenance for one independently profiled input."""

    position: int
    role: str
    source: str | None
    value_policy: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    alias_group: int
    consumer_targets: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "role": self.role,
            "source": self.source,
            "value_policy": self.value_policy,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "alias_group": self.alias_group,
            "consumer_targets": list(self.consumer_targets),
        }


@dataclass(frozen=True, slots=True)
class PlanGraphProfile:
    """Measured cost and memory geometry for one executable graph contract."""

    direction: str
    structural_contract_key: str
    semantic_contract_digest: str
    semantic_contract_capture_ns: int
    semantic_roots: tuple[PlanStorageRoot, ...]
    semantic_output_views: tuple[PlanOutputView, ...]
    semantic_mutations: tuple[PlanMutationBinding, ...]
    executable_contract_digest: str
    executable_contract_capture_ns: int
    executable_roots: tuple[PlanStorageRoot, ...]
    executable_output_views: tuple[PlanOutputView, ...]
    executable_mutations: tuple[PlanMutationBinding, ...]
    compiled_layout_digest: str
    compiled_roots: tuple[PlanCompiledRoot, ...]
    compiled_output_views: tuple[PlanCompiledOutputView, ...]
    physical_profile_wall_time_ns: int
    representative_task_id: str
    runtime_ns: int
    samples_ns: tuple[int, ...]
    provenance: str
    representative_inputs: tuple[PlanRepresentativeInput, ...]
    profile_phase_timings_ns: tuple[tuple[str, int], ...]
    timing_relative_mad: float
    timing_half_drift: float
    timing_unstable: bool
    inputs: tuple[PlanObjectFootprint, ...]
    mutations: tuple[PlanObjectFootprint, ...]
    outputs: tuple[PlanObjectFootprint, ...]
    input_logical_bytes: int
    input_allocation_bytes: int
    mutation_logical_bytes: int
    mutation_allocation_bytes: int
    output_logical_bytes: int
    output_allocation_bytes: int
    workspace_requested_bytes: int
    workspace_charged_bytes: int
    replacement_transition_bytes: int
    task_workspace_bytes: int
    workspace_extent_bytes: tuple[int, ...]
    persistent_extent_bytes: tuple[int, ...]
    allocation_contract_digest: str | None
    allocation_contract: tuple[PlanAllocationABIStep, ...]
    allocation_timeline: tuple[PlanAllocationEvent, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "structural_contract_key": self.structural_contract_key,
            "semantic_contract_digest": self.semantic_contract_digest,
            "semantic_contract_capture_ns": self.semantic_contract_capture_ns,
            "semantic_roots": [item.as_dict() for item in self.semantic_roots],
            "semantic_output_views": [
                item.as_dict() for item in self.semantic_output_views
            ],
            "semantic_mutations": [item.as_dict() for item in self.semantic_mutations],
            "executable_contract_digest": self.executable_contract_digest,
            "executable_contract_capture_ns": (self.executable_contract_capture_ns),
            "executable_roots": [item.as_dict() for item in self.executable_roots],
            "executable_output_views": [
                item.as_dict() for item in self.executable_output_views
            ],
            "executable_mutations": [
                item.as_dict() for item in self.executable_mutations
            ],
            "compiled_layout_digest": self.compiled_layout_digest,
            "compiled_roots": [item.as_dict() for item in self.compiled_roots],
            "compiled_output_views": [
                item.as_dict() for item in self.compiled_output_views
            ],
            "physical_profile_wall_time_ns": self.physical_profile_wall_time_ns,
            "representative_task_id": self.representative_task_id,
            "runtime_ns": self.runtime_ns,
            "samples_ns": list(self.samples_ns),
            "provenance": self.provenance,
            "representative_inputs": [
                item.as_dict() for item in self.representative_inputs
            ],
            "profile_phase_timings_ns": [
                list(item) for item in self.profile_phase_timings_ns
            ],
            "timing_relative_mad": self.timing_relative_mad,
            "timing_half_drift": self.timing_half_drift,
            "timing_unstable": self.timing_unstable,
            "inputs": [item.as_dict() for item in self.inputs],
            "mutations": [item.as_dict() for item in self.mutations],
            "outputs": [item.as_dict() for item in self.outputs],
            "input_logical_bytes": self.input_logical_bytes,
            "input_allocation_bytes": self.input_allocation_bytes,
            "mutation_logical_bytes": self.mutation_logical_bytes,
            "mutation_allocation_bytes": self.mutation_allocation_bytes,
            "output_logical_bytes": self.output_logical_bytes,
            "output_allocation_bytes": self.output_allocation_bytes,
            "workspace_requested_bytes": self.workspace_requested_bytes,
            "workspace_charged_bytes": self.workspace_charged_bytes,
            "replacement_transition_bytes": self.replacement_transition_bytes,
            "task_workspace_bytes": self.task_workspace_bytes,
            "workspace_extent_bytes": list(self.workspace_extent_bytes),
            "persistent_extent_bytes": list(self.persistent_extent_bytes),
            "allocation_contract_digest": self.allocation_contract_digest,
            "allocation_contract": [
                item.as_dict() for item in self.allocation_contract
            ],
            "allocation_timeline": [
                item.as_dict() for item in self.allocation_timeline
            ],
        }


@dataclass(frozen=True, slots=True)
class PlanGraphPair:
    """One legal stage choice; forward-only choices omit ``backward``."""

    variant: str
    memory_budget: float | None
    recomputation: bool
    saved_value_count: int
    specialized_unit_tangent_count: int
    saved_input_root_count: int
    saved_boundary_root_count: int
    saved_internal_root_count: int
    saved_input_minimum_bytes: int
    saved_boundary_minimum_bytes: int
    saved_internal_minimum_bytes: int
    forward: PlanGraphProfile
    backward: PlanGraphProfile | None

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "memory_budget": self.memory_budget,
            "recomputation": self.recomputation,
            "saved_value_count": self.saved_value_count,
            "specialized_unit_tangent_count": self.specialized_unit_tangent_count,
            "saved_input_root_count": self.saved_input_root_count,
            "saved_boundary_root_count": self.saved_boundary_root_count,
            "saved_internal_root_count": self.saved_internal_root_count,
            "saved_input_minimum_bytes": self.saved_input_minimum_bytes,
            "saved_boundary_minimum_bytes": self.saved_boundary_minimum_bytes,
            "saved_internal_minimum_bytes": self.saved_internal_minimum_bytes,
            "forward": self.forward.as_dict(),
            "backward": None if self.backward is None else self.backward.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PlanUniqueStage:
    """Deduplicated structural stage and every legal graph-pair choice."""

    unique_stage_id: str
    structural_key: str
    module_targets: tuple[str, ...]
    occurrence_count: int
    graph_pairs: tuple[PlanGraphPair, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "unique_stage_id": self.unique_stage_id,
            "structural_key": self.structural_key,
            "module_targets": list(self.module_targets),
            "occurrence_count": self.occurrence_count,
            "graph_pairs": [item.as_dict() for item in self.graph_pairs],
        }
