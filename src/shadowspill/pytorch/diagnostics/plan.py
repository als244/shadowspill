"""Immutable diagnostics describing one completed planning call."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shadowspill.ir import ExecutionPlan, MemoryAction, Program, TaskProfile
from shadowspill.planner import PressureFitDiagnostics, PressureFitResult
from shadowspill.pytorch.runtime_adapter import TransferCapabilities, TransferProfile


@dataclass(frozen=True, slots=True)
class PlanPhaseTiming:
    """One non-overlapping interval measured during frontend planning."""

    name: str
    duration_ns: int

    @property
    def duration_seconds(self) -> float:
        return self.duration_ns / 1e9

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "duration_ns": self.duration_ns,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class PlanCompilerProfile:
    """Non-overlapping compiler phases for one structural task ABI."""

    structural_abi_key: str
    phases: tuple[PlanPhaseTiming, ...]

    @property
    def total_wall_time_ns(self) -> int:
        return sum(item.duration_ns for item in self.phases)

    def as_dict(self) -> dict[str, object]:
        return {
            "structural_abi_key": self.structural_abi_key,
            "phases": [item.as_dict() for item in self.phases],
            "total_wall_time_ns": self.total_wall_time_ns,
        }


@dataclass(frozen=True, slots=True)
class PlanCacheArtifact:
    """One persistent planning artifact touched by this planning call.

    ``access`` distinguishes bytes actually read or written from an existing
    artifact that merely matched a freshly produced in-memory result.  PyTorch
    Inductor's implementation-private directory is reported as ``managed``.
    """

    category: str
    kind: str
    digest: str | None
    path: str
    access: str
    schema: str | None = None
    dependencies: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "kind": self.kind,
            "digest": self.digest,
            "path": self.path,
            "access": self.access,
            "schema": self.schema,
            "dependencies": list(self.dependencies),
        }


@dataclass(frozen=True, slots=True)
class PlanProfilingMetadata:
    """Canonical planning-only workload metadata for one input position."""

    position: int
    digest: str
    canonical_json: str

    def as_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "digest": self.digest,
            "canonical_json": self.canonical_json,
        }


@dataclass(frozen=True, slots=True)
class PlanObjectFootprint:
    """One logical tensor view and the allocator extent containing it."""

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
    """Observed physical binding for one returned tensor leaf."""

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
    """Measured cost and memory geometry for one executable graph ABI."""

    direction: str
    structural_abi_key: str
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
            "structural_abi_key": self.structural_abi_key,
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
    structural_abi_key: str
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
            "structural_abi_key": self.structural_abi_key,
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


@dataclass(frozen=True, slots=True)
class PlanFixedLayoutAttempt:
    """One PressureFit-capacity/layout trial made during admission."""

    requested_object_capacity_bytes: int
    effective_object_capacity_bytes: int
    required_bytes: int
    pool_capacity_bytes: int
    accepted: bool
    pressurefit_wall_time_ns: int
    physical_admission_wall_time_ns: int
    pressurefit_diagnostics: PressureFitDiagnostics | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_object_capacity_bytes": self.requested_object_capacity_bytes,
            "effective_object_capacity_bytes": self.effective_object_capacity_bytes,
            "required_bytes": self.required_bytes,
            "pool_capacity_bytes": self.pool_capacity_bytes,
            "accepted": self.accepted,
            "pressurefit_wall_time_ns": self.pressurefit_wall_time_ns,
            "physical_admission_wall_time_ns": (self.physical_admission_wall_time_ns),
            "pressurefit_diagnostics": (
                None
                if self.pressurefit_diagnostics is None
                else self.pressurefit_diagnostics.to_dict()
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
    topology_digest: str
    pool_capacity_bytes: int
    original_object_capacity_bytes: int
    effective_object_capacity_bytes: int
    object_capacity_reduction_bytes: int
    fixed_slice_bytes: int
    dynamic_reserve_bytes: int
    scratch_reserve_bytes: int
    required_bytes: int
    slack_bytes: int
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
            "topology_digest": self.topology_digest,
            "pool_capacity_bytes": self.pool_capacity_bytes,
            "original_object_capacity_bytes": self.original_object_capacity_bytes,
            "effective_object_capacity_bytes": self.effective_object_capacity_bytes,
            "object_capacity_reduction_bytes": (self.object_capacity_reduction_bytes),
            "fixed_slice_bytes": self.fixed_slice_bytes,
            "dynamic_reserve_bytes": self.dynamic_reserve_bytes,
            "scratch_reserve_bytes": self.scratch_reserve_bytes,
            "required_bytes": self.required_bytes,
            "slack_bytes": self.slack_bytes,
            "placement_count": self.placement_count,
            "dynamic_lifetime_count": self.dynamic_lifetime_count,
            "reuse_dependency_count": self.reuse_dependency_count,
            "placements_by_purpose": dict(self.placements_by_purpose),
            "attempts": [item.as_dict() for item in self.attempts],
            "task_memory_envelopes": [
                item.as_dict() for item in self.task_memory_envelopes
            ],
        }


@dataclass(frozen=True, slots=True)
class PlanDiagnostics:
    """Structured evidence describing one frontend planning call.

    Phase intervals are mutually exclusive. ``unattributed_overhead_ns`` is
    the small remainder spent between measured intervals and constructing the
    immutable report; phases plus that remainder equal ``total_wall_time_ns``.
    """

    phases: tuple[PlanPhaseTiming, ...]
    total_wall_time_ns: int
    unattributed_overhead_ns: int
    profile_unique_keys: int
    profile_cache_hits: int
    profile_cache_misses: int
    allocation_probe_seeds: int
    allocation_probe_repetitions: int
    captured_stage_count: int
    aot_unique_stage_abis: int
    aot_graph_pair_cache_hits: int
    aot_graph_pair_cache_misses: int
    recomputation_cache_hits: int
    recomputation_cache_misses: int
    task_stage_map: tuple[PlanTaskStage, ...] = ()
    unique_stages: tuple[PlanUniqueStage, ...] = ()
    compiler_phase_timings_ns: tuple[tuple[str, int], ...] = ()
    compiler_profiles: tuple[PlanCompilerProfile, ...] = ()
    cache_directories: tuple[tuple[str, str], ...] = ()
    cache_artifacts: tuple[PlanCacheArtifact, ...] = ()
    profiling_metadata: tuple[PlanProfilingMetadata, ...] = ()
    pressurefit_runs: tuple[PressureFitDiagnostics, ...] = ()
    physical_layouts: tuple[PlanPhysicalLayout, ...] = ()

    @property
    def measured_wall_time_ns(self) -> int:
        return sum(item.duration_ns for item in self.phases)

    def as_dict(self) -> dict[str, object]:
        selected_tasks = self.tasks
        return {
            "schema": "shadowspill.plan_diagnostics/v1",
            "phases": [item.as_dict() for item in self.phases],
            "measured_wall_time_ns": self.measured_wall_time_ns,
            "unattributed_overhead_ns": self.unattributed_overhead_ns,
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
                "structural_abis": {
                    item.structural_abi_key: item.as_dict()
                    for item in self.compiler_profiles
                },
            },
            "cache_directories": dict(self.cache_directories),
            "cache_artifacts": [item.as_dict() for item in self.cache_artifacts],
            "profiling_metadata": [item.as_dict() for item in self.profiling_metadata],
            "capture": {
                "stage_count": self.captured_stage_count,
                "aot_unique_stage_abis": self.aot_unique_stage_abis,
                "aot_graph_pair_cache_hits": self.aot_graph_pair_cache_hits,
                "aot_graph_pair_cache_misses": self.aot_graph_pair_cache_misses,
            },
            "recomputation": {
                "cache_hits": self.recomputation_cache_hits,
                "cache_misses": self.recomputation_cache_misses,
            },
            "pressurefit": [
                {
                    "run_index": index,
                    **item.to_dict(),
                }
                for index, item in enumerate(self.pressurefit_runs)
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
    initial_execution_plan: ExecutionPlan | None = None
    recomputation_cache_hits: int = 0
    recomputation_cache_misses: int = 0
    fixed_slab_bytes: int = 0
    captured_stage_count: int = 0
    aot_unique_stage_abis: int = 0
    aot_graph_pair_cache_hits: int = 0
    aot_graph_pair_cache_misses: int = 0
    pressurefit_results: tuple[PressureFitResult, ...] = ()

    @property
    def program(self) -> Program:
        """Canonical recurrent Program supplied directly to PressureFit.

        Forward plans have one Program.  Training plans expose the recurrent
        step here; :attr:`initial_program` names the optional lazy-state first
        step separately.
        """

        return self.execution_plan.program

    @property
    def initial_program(self) -> Program | None:
        """Canonical first-step Program, when lazy optimizer state requires one."""

        if self.initial_execution_plan is None:
            return None
        return self.initial_execution_plan.program

    @property
    def pressurefit_result(self) -> PressureFitResult:
        """PressureFit call boundary and selected result for the recurrent plan."""

        if not self.pressurefit_results:
            raise RuntimeError("PlanReport does not contain PressureFit evidence")
        return self.pressurefit_results[-1]

    @property
    def initial_pressurefit_result(self) -> PressureFitResult | None:
        """Selected first-step PressureFit result, when one was planned."""

        if self.initial_execution_plan is None:
            return None
        if len(self.pressurefit_results) < 2:
            raise RuntimeError("PlanReport is missing first-step PressureFit evidence")
        return self.pressurefit_results[0]

    @property
    def predicted_device_peak_bytes(self) -> int:
        return self.execution_plan.prediction.device_peak_bytes

    @property
    def predicted_host_peak_bytes(self) -> int:
        return self.execution_plan.prediction.host_peak_bytes

    @property
    def predicted_makespan_ns(self) -> int:
        return self.execution_plan.prediction.makespan_ns

    @property
    def fetch_profile(self) -> TransferProfile:
        """Measured spill-to-execution route consumed by this plan."""

        return self.transfer_capabilities.route(self.spill_pool, self.execution_pool)

    @property
    def evict_profile(self) -> TransferProfile:
        """Measured execution-to-spill route consumed by this plan."""

        return self.transfer_capabilities.route(self.execution_pool, self.spill_pool)
