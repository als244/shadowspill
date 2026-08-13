"""Documented forward/training result values and planned callable types."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
import torch.nn as nn

from shadowspill.ir import ExecutionPlan, MemoryAction, Program, TaskProfile
from shadowspill.planner import PressureFitResult

from ._planning_cache import PlanningCache
from .executor import ForwardExecutor
from .guards import InputSignature, validate_training_inputs
from .materialization import MaterializedForwardState
from .runtime import Runtime, TransferCapabilities, TransferProfile
from .training_executor import ExecutionTiming, StepDiagnostics, TrainingExecutor
from .training_materialization import TrainingMaterializedState


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
            "allocation_timeline": [
                item.as_dict() for item in self.allocation_timeline
            ],
        }


@dataclass(frozen=True, slots=True)
class PlanGraphPair:
    """One legal stage choice; forward-only choices omit ``backward``."""

    variant: str
    recomputation: bool
    saved_value_count: int
    specialized_unit_tangent_count: int
    forward: PlanGraphProfile
    backward: PlanGraphProfile | None

    def as_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "recomputation": self.recomputation,
            "saved_value_count": self.saved_value_count,
            "specialized_unit_tangent_count": self.specialized_unit_tangent_count,
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
        """Selected tasks keyed by dense chronological execution identity."""

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
    profiling_provenance: tuple[str, ...]
    phase_timings_ns: tuple[tuple[str, int], ...]
    diagnostics: PlanDiagnostics
    execution_pool: str
    spill_pool: str
    execution_budget_bytes: int
    spill_budget_bytes: int
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


class PlannedForward:
    """Forward-only callable returned by :func:`plan_forward`.

    The original model is runtime-owned until `close()`. Calls validate the
    complete fixed input signature before writing an input slot or launching a
    task. Returned tensors are ordinary caller-owned allocator records.
    """

    def __init__(
        self,
        model: nn.Module,
        signature: InputSignature,
        executor: ForwardExecutor,
        state: MaterializedForwardState,
        report: PlanReport,
        runtime: Runtime,
    ) -> None:
        self._model = model
        self._signature = signature
        self._executor = executor
        self._state = state
        self.plan_report = report
        self._runtime = runtime
        self._runtime._adopt_plan()
        self._closed = False
        self._profiler_annotations_active = False

    def __call__(
        self,
        inputs: Sequence[Any],
        *,
        profiler_annotations: bool = False,
    ) -> object:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
        if not isinstance(profiler_annotations, bool):
            raise TypeError("profiler_annotations must be a bool")
        if self._profiler_annotations_active and not profiler_annotations:
            self._executor.finish_profiler_annotations()
            self._profiler_annotations_active = False
        elif profiler_annotations and not self._profiler_annotations_active:
            self._executor.set_profiler_annotations(True)
            self._profiler_annotations_active = True
        self._signature.validate(inputs)
        return self._executor(inputs)

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Synchronously snapshot model state into ordinary CPU tensors."""

        if self._closed:
            return OrderedDict(self._model.state_dict())
        return self._state.state_dict()

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Synchronously replace the current host-authoritative model state."""

        if self._closed:
            self._model.load_state_dict(state)
            return
        self._state.load_state_dict(state)

    def close(self) -> None:
        """Synchronize, restore the original model to CPU, and release the plan."""

        if self._closed:
            return
        if self._profiler_annotations_active:
            self._executor.finish_profiler_annotations()
            self._profiler_annotations_active = False
        self._state.restore_cpu_and_unregister()
        self._closed = True
        self._runtime._release_plan()

    def __enter__(self) -> PlannedForward:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()


@dataclass(frozen=True, slots=True)
class StepResult:
    """Detached device results for every microbatch in one logical step."""

    objectives: tuple[torch.Tensor, ...]
    metrics: tuple[Any, ...]
    step_number: int
    diagnostics: DiagnosticsHandle | None = None


class DiagnosticsHandle:
    """Deferred collection of an explicitly enabled detailed execution trace."""

    def __init__(self, collector: Callable[[], StepDiagnostics]) -> None:
        self._collector = collector
        self._result: StepDiagnostics | None = None

    @property
    def resolved(self) -> bool:
        return self._result is not None

    def result(self) -> StepDiagnostics:
        """Synchronize trace completion and return immutable step evidence."""

        if self._result is None:
            self._result = self._collector()
        return self._result

    wait = result


class PlannedTrainStep:
    """Accumulated training callable returned by :func:`plan_step`."""

    def __init__(
        self,
        model: nn.Module,
        signatures: tuple[InputSignature, ...],
        executor: TrainingExecutor,
        state: TrainingMaterializedState,
        optimizer: torch.optim.Optimizer,
        report: PlanReport,
        runtime: Runtime,
    ) -> None:
        self._model = model
        self._signatures = signatures
        self._executor = executor
        self._state = state
        self._optimizer = optimizer
        self.plan_report = report
        self._runtime = runtime
        self._runtime._adopt_plan()
        self._step = 0
        self._closed = False
        self._trace_prepared = False
        self._pending_diagnostics: DiagnosticsHandle | None = None
        self._profiler_annotations_active = False

    def __call__(
        self,
        inputs: Sequence[Sequence[Any]],
        *,
        runtime_trace: bool = False,
        profiler_annotations: bool = False,
    ) -> StepResult:
        if self._closed:
            raise RuntimeError("planned training callable is closed")
        if (
            self._pending_diagnostics is not None
            and not self._pending_diagnostics.resolved
        ):
            raise RuntimeError(
                "resolve the preceding traced StepResult diagnostics before "
                "launching another traced step"
            )
        if not isinstance(runtime_trace, bool):
            raise TypeError("runtime_trace must be a bool")
        if not isinstance(profiler_annotations, bool):
            raise TypeError("profiler_annotations must be a bool")
        if self._profiler_annotations_active and not profiler_annotations:
            self._executor.finish_profiler_annotations()
            self._profiler_annotations_active = False
        elif profiler_annotations and not self._profiler_annotations_active:
            self._executor.set_profiler_annotations(True)
            self._profiler_annotations_active = True
        validate_training_inputs(inputs, self._signatures)
        trace_setup_ns = 0
        if runtime_trace:
            if not self._trace_prepared:
                started_ns = time.perf_counter_ns()
                self._executor.prepare_execution_tracing()
                trace_setup_ns = time.perf_counter_ns() - started_ns
                self._trace_prepared = True
            self._executor.arm_compute_timing(trace_setup_ns=trace_setup_ns)
        try:
            objectives, metrics = self._executor(inputs)
        except BaseException:
            if runtime_trace:
                self._executor.cancel_execution_timing()
            raise
        self._step += 1
        diagnostics = (
            DiagnosticsHandle(self._executor.collect_step_diagnostics)
            if runtime_trace
            else None
        )
        self._pending_diagnostics = diagnostics
        return StepResult(objectives, metrics, self._step, diagnostics)

    def _arm_compute_timing(self) -> None:
        """Arm qualification-only first-task-to-final-optimizer timing."""

        self._executor.arm_compute_timing()

    def _collect_compute_seconds(self) -> float:
        """Collect a previously armed qualification timing interval."""

        return self._executor.collect_compute_seconds()

    def _collect_execution_timing(self) -> ExecutionTiming:
        """Collect qualification-only per-task and per-phase timings."""

        return self._executor.collect_execution_timing()

    def _arm_selected_span_timing(self) -> None:
        """Arm production-like two-event task-span timing."""

        self._executor.arm_selected_span_timing()

    def _collect_selected_span_seconds(self) -> float:
        """Collect production-like two-event task-span timing."""

        return self._executor.collect_selected_span_seconds()

    def state_dict(self) -> dict[str, object]:
        """Synchronously snapshot model, optimizer, and logical step state."""

        if self._closed:
            model_state = OrderedDict(self._model.state_dict())
        else:
            model_state = self._state.state_dict()
        return {
            "model": model_state,
            "optimizer": self._optimizer.state_dict()
            if self._closed
            else self._executor.optimizer_state_dict(),
            "step": self._step,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore an exact state produced by :meth:`state_dict`."""

        if set(state) != {"model", "optimizer", "step"}:
            raise RuntimeError("training state_dict keys differ")
        model_state = state["model"]
        optimizer_state = state["optimizer"]
        step = state["step"]
        if not isinstance(model_state, Mapping) or not isinstance(
            optimizer_state, Mapping
        ):
            raise TypeError("training checkpoint model/optimizer must be mappings")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise TypeError("training checkpoint step must be non-negative")
        self._state.load_model_state(model_state)
        initialized = self._executor.load_optimizer_state(optimizer_state)
        self._executor.set_optimizer_state_initialized(initialized)
        self._step = step

    def close(self) -> None:
        if self._closed:
            return
        if (
            self._pending_diagnostics is not None
            and not self._pending_diagnostics.resolved
        ):
            self._pending_diagnostics.result()
        if self._profiler_annotations_active:
            self._executor.finish_profiler_annotations()
            self._profiler_annotations_active = False
        for parameter in self._model.parameters():
            parameter.grad = None
        self._executor.restore_optimizer_cpu()
        self._state.restore_cpu_and_unregister()
        self._closed = True
        self._runtime._release_plan()

    def __enter__(self) -> PlannedTrainStep:
        if self._closed:
            raise RuntimeError("planned training callable is closed")
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()


def plan_forward(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    runtime: Runtime,
    execution: str,
    spill: str,
    execution_budget: int | None = None,
    spill_budget: int | None = None,
    execution_device: int | str | torch.device | None = None,
    partition: str = "auto",
    verbose: bool = True,
    planning_cachedir: str | os.PathLike[str] | None = None,
    profiling_metadata: object = None,
    save_plan: bool = True,
    force_fresh: bool = False,
    overwrite_plan: bool = False,
    implementation_revision: str | None = None,
) -> PlannedForward:
    """Plan one fixed-shape forward program around ordinary PyTorch tasks.

    The runtime and pool roles are explicit. The original model remains
    runtime-owned until the returned callable is closed. ``profiling_metadata``
    is a JSON-compatible, key-only description of value-sensitive profiling
    behavior. It is not passed to the model or returned callable.

    ``planning_cachedir`` selects the shared artifact store. ``force_fresh``
    disables cache reads; ``save_plan`` controls writes; and
    ``overwrite_plan`` replaces an existing identity only during a saved fresh
    run. ``implementation_revision`` invalidates compiler/profile artifacts
    when a lower-level custom implementation changes without changing its
    exported graph.
    """

    from .planning.forward import build_forward

    memory = runtime._resolve_plan(
        execution=execution,
        spill=spill,
        execution_budget=execution_budget,
        spill_budget=spill_budget,
        execution_device=execution_device,
    )
    cache = PlanningCache.resolve(
        planning_cachedir,
        save_plan=save_plan,
        force_fresh=force_fresh,
        overwrite_plan=overwrite_plan,
        implementation_revision=implementation_revision,
    )
    try:
        with cache.activate_pytorch():
            return build_forward(
                model,
                example_inputs=example_inputs,
                memory=memory,
                partition=partition,
                verbose=verbose,
                planning_cache=cache,
                profiling_metadata=profiling_metadata,
            )
    except BaseException as error:
        try:
            runtime._abort_plan()
        except BaseException as cleanup_error:
            error.add_note(f"Runtime planning cleanup also failed: {cleanup_error}")
        raise


def plan_step(
    model: nn.Module,
    *,
    objective: Any,
    opt: Any,
    example_inputs: Sequence[Sequence[Any]],
    runtime: Runtime,
    execution: str,
    spill: str,
    execution_budget: int | None = None,
    spill_budget: int | None = None,
    execution_device: int | str | torch.device | None = None,
    partition: str = "auto",
    optimizer_ordering: Literal["stage_interleaved", "tail"] = "stage_interleaved",
    verbose: bool = True,
    planning_cachedir: str | os.PathLike[str] | None = None,
    profiling_metadata: Sequence[object] | None = None,
    save_plan: bool = True,
    force_fresh: bool = False,
    overwrite_plan: bool = False,
    implementation_revision: str | None = None,
) -> PlannedTrainStep:
    """Plan a fixed accumulated forward/objective/backward/update program.

    ``verbose=True`` reports each planning phase and unique structural ABI as
    it starts. Set it to ``False`` for silent embedding; diagnostics are still
    retained in :attr:`PlannedTrainStep.plan_report` either way.

    ``profiling_metadata`` has one JSON-compatible entry per example
    microbatch. It only distinguishes value-sensitive task measurements and
    their downstream plans; it is never passed to the objective or runtime.
    Cache policy arguments have the same meaning as :func:`plan_forward`.
    """

    from .planning.training import build_training

    memory = runtime._resolve_plan(
        execution=execution,
        spill=spill,
        execution_budget=execution_budget,
        spill_budget=spill_budget,
        execution_device=execution_device,
    )
    cache = PlanningCache.resolve(
        planning_cachedir,
        save_plan=save_plan,
        force_fresh=force_fresh,
        overwrite_plan=overwrite_plan,
        implementation_revision=implementation_revision,
    )
    try:
        with cache.activate_pytorch():
            return build_training(
                model,
                objective=objective,
                opt=opt,
                example_inputs=example_inputs,
                memory=memory,
                partition=partition,
                optimizer_ordering=optimizer_ordering,
                verbose=verbose,
                planning_cache=cache,
                profiling_metadata=profiling_metadata,
            )
    except BaseException as error:
        try:
            runtime._abort_plan()
        except BaseException as cleanup_error:
            error.add_note(f"Runtime planning cleanup also failed: {cleanup_error}")
        raise


__all__ = [
    "DiagnosticsHandle",
    "ExecutionTiming",
    "PlanAllocationEvent",
    "PlanCacheArtifact",
    "PlanCompiledOutputView",
    "PlanCompiledRoot",
    "PlanCompilerProfile",
    "PlanDiagnostics",
    "PlanGraphPair",
    "PlanGraphProfile",
    "PlanMutationBinding",
    "PlanObjectFootprint",
    "PlanOutputView",
    "PlanPhaseTiming",
    "PlanProfilingMetadata",
    "PlanReport",
    "PlanStorageRoot",
    "PlanTaskStage",
    "PlanUniqueStage",
    "PlannedForward",
    "PlannedTrainStep",
    "StepDiagnostics",
    "StepResult",
    "plan_forward",
    "plan_step",
]
