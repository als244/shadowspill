"""Documented forward/training result values and planned callable types."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn as nn

from shadowspill.ir import ExecutionPlan, MemoryAction, TaskProfile
from shadowspill.planner import PressureFitResult

from .executor import ForwardExecutor
from .guards import InputSignature, validate_training_inputs
from .materialization import MaterializedForwardState
from .training_executor import ExecutionTiming, StepDiagnostics, TrainingExecutor
from .training_materialization import TrainingMaterializedState


@dataclass(frozen=True, slots=True)
class PlanPhaseTiming:
    """One non-overlapping interval measured during ``plan()``."""

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
    reuses_ordinal: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "allocation_ordinal": self.allocation_ordinal,
            "operation": self.operation,
            "requested_bytes": self.requested_bytes,
            "charged_bytes": self.charged_bytes,
            "output_leaf_indices": list(self.output_leaf_indices),
            "reuses_ordinal": self.reuses_ordinal,
        }


@dataclass(frozen=True, slots=True)
class PlanGraphProfile:
    """Measured cost and memory geometry for one executable graph ABI."""

    direction: str
    structural_abi_key: str
    representative_task_id: str
    runtime_ns: int
    samples_ns: tuple[int, ...]
    provenance: str
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
    task_workspace_bytes: int
    workspace_extent_bytes: tuple[int, ...]
    persistent_extent_bytes: tuple[int, ...]
    allocation_timeline: tuple[PlanAllocationEvent, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "structural_abi_key": self.structural_abi_key,
            "representative_task_id": self.representative_task_id,
            "runtime_ns": self.runtime_ns,
            "samples_ns": list(self.samples_ns),
            "provenance": self.provenance,
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
    graph_pair_variant: str | None
    chosen_graph_pair_variant: str | None
    selected: bool

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
            "graph_pair_variant": self.graph_pair_variant,
            "chosen_graph_pair_variant": self.chosen_graph_pair_variant,
            "selected": self.selected,
        }


@dataclass(frozen=True, slots=True)
class PlanDiagnostics:
    """Structured evidence describing work performed by ``plan()``.

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
    transfer_bytes_to_host: int
    transfer_bytes_to_device: int
    profile_unique_keys: int
    profile_cache_hits: int
    profile_cache_misses: int
    profiling_provenance: tuple[str, ...]
    phase_timings_ns: tuple[tuple[str, int], ...]
    diagnostics: PlanDiagnostics
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
    def predicted_device_peak_bytes(self) -> int:
        return self.execution_plan.prediction.device_peak_bytes

    @property
    def predicted_host_peak_bytes(self) -> int:
        return self.execution_plan.prediction.host_peak_bytes

    @property
    def predicted_makespan_ns(self) -> int:
        return self.execution_plan.prediction.makespan_ns


class PlannedForward:
    """Forward-only callable returned by :func:`forward_pass`.

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
    ) -> None:
        self._model = model
        self._signature = signature
        self._executor = executor
        self._state = state
        self.plan_report = report
        self._closed = False

    def __call__(self, inputs: Sequence[Any]) -> object:
        if self._closed:
            raise RuntimeError("planned forward callable is closed")
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
        self._state.restore_cpu_and_unregister()
        self._closed = True

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
    """Accumulated training callable returned by :func:`plan`."""

    def __init__(
        self,
        model: nn.Module,
        signatures: tuple[InputSignature, ...],
        executor: TrainingExecutor,
        state: TrainingMaterializedState,
        optimizer: torch.optim.Optimizer,
        report: PlanReport,
    ) -> None:
        self._model = model
        self._signatures = signatures
        self._executor = executor
        self._state = state
        self._optimizer = optimizer
        self.plan_report = report
        self._step = 0
        self._closed = False
        self._trace_prepared = False
        self._pending_diagnostics: DiagnosticsHandle | None = None

    def __call__(
        self, inputs: Sequence[Sequence[Any]], *, trace: bool = False
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
        if not isinstance(trace, bool):
            raise TypeError("trace must be a bool")
        validate_training_inputs(inputs, self._signatures)
        trace_setup_ns = 0
        if trace:
            if not self._trace_prepared:
                started_ns = time.perf_counter_ns()
                self._executor.prepare_execution_tracing()
                trace_setup_ns = time.perf_counter_ns() - started_ns
                self._trace_prepared = True
            self._executor.arm_compute_timing(trace_setup_ns=trace_setup_ns)
        try:
            objectives, metrics = self._executor(inputs)
        except BaseException:
            if trace:
                self._executor.cancel_execution_timing()
            raise
        self._step += 1
        diagnostics = (
            DiagnosticsHandle(self._executor.collect_step_diagnostics)
            if trace
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
        for parameter in self._model.parameters():
            parameter.grad = None
        self._executor.restore_optimizer_cpu()
        self._state.restore_cpu_and_unregister()
        self._closed = True

    def __enter__(self) -> PlannedTrainStep:
        if self._closed:
            raise RuntimeError("planned training callable is closed")
        return self

    def __exit__(self, *exception: object) -> None:
        del exception
        self.close()


def forward_pass(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    device_budget: int,
    host_budget: int,
    partition: str = "auto",
    verbose: bool = True,
) -> PlannedForward:
    """Plan one fixed-shape forward program around ordinary PyTorch tasks.

    Planning installs ShadowSpill's process-global CUDA allocator. The original
    model remains runtime-owned until the returned callable is closed.
    """

    from .session import build_forward

    return build_forward(
        model,
        example_inputs=example_inputs,
        device_budget=device_budget,
        host_budget=host_budget,
        partition=partition,
        verbose=verbose,
    )


def plan(
    model: nn.Module,
    *,
    objective: Any,
    opt: Any,
    example_inputs: Sequence[Sequence[Any]],
    device_budget: int,
    host_budget: int,
    partition: str = "auto",
    verbose: bool = True,
) -> PlannedTrainStep:
    """Plan a fixed accumulated forward/objective/backward/update program.

    ``verbose=True`` reports each planning phase and unique structural ABI as
    it starts. Set it to ``False`` for silent embedding; diagnostics are still
    retained in :attr:`PlannedTrainStep.plan_report` either way.
    """

    from .training_session import build_training

    return build_training(
        model,
        objective=objective,
        opt=opt,
        example_inputs=example_inputs,
        device_budget=device_budget,
        host_budget=host_budget,
        partition=partition,
        verbose=verbose,
    )


__all__ = [
    "DiagnosticsHandle",
    "ExecutionTiming",
    "PlanAllocationEvent",
    "PlanDiagnostics",
    "PlanGraphPair",
    "PlanGraphProfile",
    "PlanObjectFootprint",
    "PlanPhaseTiming",
    "PlanReport",
    "PlanTaskStage",
    "PlanUniqueStage",
    "PlannedForward",
    "PlannedTrainStep",
    "StepDiagnostics",
    "StepResult",
    "forward_pass",
    "plan",
]
