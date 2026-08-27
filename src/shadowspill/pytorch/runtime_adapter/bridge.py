"""Narrow high-level bridge over the private PyTorch adapter ABI."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from shadowspill.errors import PlanningError
from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MutationSpec,
    Program,
    TaskSpec,
)
from shadowspill.pytorch.runtime_adapter.abi import (
    AdapterStatistics,
    FixedDependencyDescription,
    FixedLayoutDescription,
    FixedPlacementDescription,
    ObjectBinding,
    ObjectSnapshot,
    ObjectUpdate,
    RuntimeAction,
    TaskDescription,
    TaskPublicationDescription,
    runtime_library,
)
from shadowspill.pytorch.runtime_adapter.abi import (
    TaskAllocationContractStep as CTaskAllocationContractStep,
)
from shadowspill.pytorch.runtime_adapter.failures import (
    RuntimeExecutionError,
    generic_runtime_error,
    raise_if_allocator_failed,
    read_allocator_failure,
)
from shadowspill.pytorch.runtime_adapter.fixed_layout import RuntimeFixedLayout
from shadowspill.pytorch.runtime_adapter.runtime import Runtime
from shadowspill.runtime import ObjectConsistency, ObjectRef
from shadowspill.status import ABI_VERSION

if TYPE_CHECKING:
    from shadowspill.pytorch.materialization.replacement import (
        ReplacementStorageViews,
    )
    from shadowspill.pytorch.profiling.allocation_contract import TaskAllocationContract
from shadowspill.pytorch.runtime_adapter.trace import (
    CapturedRuntimeTrace,
    begin_runtime_trace,
    end_runtime_trace,
    prepare_runtime_trace,
    read_runtime_trace,
)

_ACTION_KIND = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.OFFLOAD: 1,
    MemoryActionKind.PREFETCH: 2,
}


@dataclass(frozen=True, slots=True)
class TaskMemoryEnvelope:
    """Conservative bounds on one task's anonymous allocator behavior."""

    maximum_requested_allocation_bytes: int = 0
    maximum_charged_allocation_bytes: int = 0
    live_requested_allocation_limit_bytes: int = 0
    live_charged_allocation_limit_bytes: int = 0
    dynamic_scratch_maximum_allocation_bytes: int = 0
    dynamic_scratch_live_limit_bytes: int = 0
    allocation_path_digests: tuple[str, ...] = ()
    allocation_contract: TaskAllocationContract | None = None

    def __post_init__(self) -> None:
        values = (
            self.maximum_requested_allocation_bytes,
            self.maximum_charged_allocation_bytes,
            self.live_requested_allocation_limit_bytes,
            self.live_charged_allocation_limit_bytes,
            self.dynamic_scratch_maximum_allocation_bytes,
            self.dynamic_scratch_live_limit_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("task memory envelope bounds must be non-negative")
        if (
            self.live_requested_allocation_limit_bytes
            and self.maximum_requested_allocation_bytes
            > self.live_requested_allocation_limit_bytes
        ):
            raise ValueError("requested allocation maximum exceeds live limit")
        if (
            self.live_charged_allocation_limit_bytes
            and self.maximum_charged_allocation_bytes
            > self.live_charged_allocation_limit_bytes
        ):
            raise ValueError("charged allocation maximum exceeds live limit")
        if (
            self.dynamic_scratch_live_limit_bytes
            and self.dynamic_scratch_maximum_allocation_bytes
            > self.dynamic_scratch_live_limit_bytes
        ):
            raise ValueError("scratch allocation maximum exceeds scratch live limit")
        if any(len(value) != 64 for value in self.allocation_path_digests):
            raise ValueError("allocation path digests must be SHA-256")


@dataclass(frozen=True, slots=True)
class TaskPublication:
    """One stable logical object that an admitted task may publish."""

    alias_id: str
    replace_lease: bool = False


@dataclass(frozen=True, slots=True)
class PublishedStorage:
    """One concrete task result matched to a predecoded publication."""

    tensor: torch.Tensor
    alias_id: str
    publication_ordinal: int


@dataclass(slots=True)
class _ExecutionBuffers:
    description: TaskDescription
    encoded_task_label: bytes
    input_ids: Any
    updates: Any
    publications: Any
    actions: Any
    allocation_contract_steps: Any
    encoded_labels: tuple[bytes | None, ...]


def _plan_local_id(value: str, prefix: str) -> int:
    if not value.startswith(prefix):
        raise PlanningError(
            f"plan-local identity {value!r} does not start with {prefix!r}"
        )
    suffix = value.removeprefix(prefix)
    if not suffix.isdigit():
        raise PlanningError(f"plan-local identity {value!r} has a nonnumeric suffix")
    return int(suffix)


def _action_labels(
    actions: tuple[MemoryAction, ...],
    labels: tuple[str, ...] | None,
) -> tuple[str, ...]:
    resolved = ("",) * len(actions) if labels is None else labels
    if len(resolved) != len(actions):
        raise ValueError("action trace labels must align with ordered actions")
    return resolved


def _runtime_action(
    object_id: int,
    kind: int,
    *,
    trace_label: bytes | None = None,
) -> RuntimeAction:
    """Construct one ABI action without relying on ctypes field ordering."""

    return RuntimeAction(
        object_id=object_id,
        kind=kind,
        trace_label=trace_label,
    )


class RuntimeBridge:
    """Bind one Program's local identities to shared runtime objects."""

    def __init__(
        self,
        runtime: Runtime,
        program: Program,
        plan_handle: int,
        *,
        execution_pool_id: int,
        spill_pool_id: int,
    ) -> None:
        if execution_pool_id < 0 or spill_pool_id < 0:
            raise ValueError("plan pool IDs must be non-negative")
        if execution_pool_id == spill_pool_id:
            raise ValueError("execution and spill pools must be distinct")
        self.runtime = runtime
        self.library = runtime._installed.library
        # Plan admission is the neutral runtime's own API, so the bridge
        # calls it there rather than through an adapter that only cast
        # the handle and passed it along.
        self.runtime_library = runtime_library()
        self.plan_handle = plan_handle
        self.execution_pool_id = execution_pool_id
        self.spill_pool_id = spill_pool_id
        self._alias_by_object: dict[str, str] = {
            item.object_id: item.alias_group_id for item in program.objects
        }
        self._size_by_alias: dict[str, int] = {
            item.alias_group_id: item.size_bytes for item in program.alias_groups
        }
        self._zero_generations: dict[str, int] = {}
        self._registered: set[str] = set()
        self._borrowed: set[str] = set()
        self._binding_consistency: dict[str, int] = {}
        self._runtime_object_ids: dict[str, int] = {}
        self._admitted_task_handles: set[int] = set()
        self._admitted_action_batches: dict[
            int, tuple[int, tuple[tuple[str, MemoryActionKind], ...]]
        ] = {}
        self._admitted_acquisitions: dict[tuple[str, ...], int] = {}
        self._fixed_layout_installed = False

    def _runtime_object_id(self, alias_id: str) -> int:
        try:
            return self._runtime_object_ids[alias_id]
        except KeyError as exc:
            raise RuntimeExecutionError(
                f"plan object {alias_id!r} is not bound to a runtime object"
            ) from exc

    def runtime_object_id(self, alias_id: str) -> int:
        """Return the shared runtime identity bound to one plan-local alias."""

        return self._runtime_object_id(alias_id)

    def acquire_object_reference(self, alias_id: str) -> ObjectRef:
        """Retain one registered alias independently of this plan's lifetime."""

        if alias_id not in self._registered:
            raise RuntimeExecutionError(
                f"cannot retain unregistered plan object {alias_id!r}"
            )
        if not self.requires_storage(alias_id):
            raise RuntimeExecutionError(
                "zero-byte shared objects are not supported by the public reference API"
            )
        return self.runtime._acquire_object_reference(
            object_id=self._runtime_object_id(alias_id),
            size_bytes=self._size(alias_id),
        )

    def release_object_generation(
        self,
        alias_id: str,
        *,
        expected_generation: int,
    ) -> None:
        """Release a closed public slot's residency, preserving identity."""

        if alias_id not in self._registered:
            raise RuntimeExecutionError(
                f"cannot release unregistered plan object {alias_id!r}"
            )
        if not self.requires_storage(alias_id):
            return
        self.runtime._release_object_generation(
            object_id=self._runtime_object_id(alias_id),
            expected_generation=expected_generation,
        )

    def _record_runtime_object(self, alias_id: str, runtime_object_id: int) -> None:
        existing = self._runtime_object_ids.get(alias_id)
        if existing is not None and existing != runtime_object_id:
            raise RuntimeExecutionError(
                f"plan object {alias_id!r} changed runtime identity: "
                f"{existing} -> {runtime_object_id}"
            )
        self._runtime_object_ids[alias_id] = runtime_object_id

    def _allocate_runtime_object_id(self, alias_id: str) -> int:
        existing = self._runtime_object_ids.get(alias_id)
        if existing is not None:
            return existing
        runtime_object_id = self.runtime._reserve_runtime_object_ids(1)[0]
        self._record_runtime_object(alias_id, runtime_object_id)
        return runtime_object_id

    def _bind_plan_object(
        self,
        alias_id: str,
        *,
        consistency: int | None = None,
    ) -> None:
        if not self.requires_storage(alias_id):
            return
        resolved_consistency = self._binding_consistency.get(alias_id, 0)
        if consistency is not None:
            existing = self._binding_consistency.get(alias_id)
            if existing is not None and existing != consistency:
                raise RuntimeExecutionError(
                    f"plan object {alias_id!r} changed consistency policy"
                )
            resolved_consistency = consistency
            self._binding_consistency[alias_id] = consistency
        handle = ctypes.c_size_t()
        self._require(
            self.runtime_library.shadowspill_object_handle_acquire(
                self.runtime._runtime_handle,
                self._runtime_object_id(alias_id),
                ctypes.byref(handle),
            ),
            f"acquire runtime object {alias_id}",
        )
        if handle.value == 0:
            raise RuntimeExecutionError(
                f"runtime returned an empty object handle for {alias_id!r}"
            )
        try:
            self._require(
                self.runtime_library.shadowspill_plan_bind_object(
                    self.plan_handle,
                    _plan_local_id(alias_id, "alias_"),
                    handle.value,
                    resolved_consistency,
                ),
                f"bind plan object {alias_id}",
            )
        finally:
            self._require(
                self.runtime_library.shadowspill_object_handle_release(handle.value),
                f"release runtime object handle {alias_id}",
            )

    def _bind_plan_objects(self, alias_ids: Iterable[str]) -> None:
        for alias_id in dict.fromkeys(alias_ids):
            self._bind_plan_object(alias_id)

    def profile_range_begin(self, name: str) -> int:
        """Open one optional provider-backed profiling range."""

        return int(
            self.library.shadowspill_pytorch_profile_range_begin(name.encode("utf-8"))
        )

    def profile_range_end(self, range_id: int) -> None:
        """Close a range returned by :meth:`profile_range_begin`."""

        self.library.shadowspill_pytorch_profile_range_end(range_id)

    def set_profiler_annotations(self, enabled: bool) -> None:
        """Toggle provider annotations without changing runtime tracing."""

        self._require(
            self.library.shadowspill_pytorch_profiler_annotations_set(enabled),
            f"{'enable' if enabled else 'disable'} profiler annotations",
        )

    def admit_task(
        self,
        task: TaskSpec,
        input_alias_ids: tuple[str, ...],
        actions: tuple[MemoryAction, ...],
        action_trace_labels: tuple[str, ...] | None = None,
        memory_envelope: TaskMemoryEnvelope | None = None,
        *,
        trace_label: str,
        publications: tuple[TaskPublication, ...] = (),
    ) -> int:
        """Resolve one immutable task topology in the neutral runtime."""

        if memory_envelope is None:
            memory_envelope = TaskMemoryEnvelope()
        labels = _action_labels(actions, action_trace_labels)
        runtime_inputs = self._runtime_inputs(input_alias_ids)
        mutations = self._runtime_mutations(task.mutations)
        action_pairs = self._runtime_actions(actions, labels)
        runtime_publications = tuple(
            item for item in publications if self.requires_storage(item.alias_id)
        )
        for alias_id in self._referenced_aliases(
            runtime_inputs,
            mutations,
            runtime_publications,
            action_pairs,
        ):
            self.register_placeholder(alias_id)
        self._bind_plan_objects(
            self._referenced_aliases(
                runtime_inputs, mutations, runtime_publications, action_pairs
            )
        )
        buffers = self._execution_buffers(
            task,
            runtime_inputs,
            mutations,
            runtime_publications,
            action_pairs,
            memory_envelope,
            trace_label,
        )
        task_handle = ctypes.c_size_t()
        self._require(
            self.runtime_library.shadowspill_plan_admit_task(
                self.plan_handle,
                ctypes.byref(buffers.description),
                ctypes.byref(task_handle),
            ),
            f"admit task {task.task_id}",
        )
        if task_handle.value == 0:
            raise RuntimeExecutionError(
                f"task {task.task_id} admitted with a null handle"
            )
        resolved = int(task_handle.value)
        self._admitted_task_handles.add(resolved)
        return resolved

    def admit_fixed_layout(self, layout: RuntimeFixedLayout) -> None:
        """Copy one indexed physical-layout certificate into the C runtime."""

        if self._admitted_task_handles or self._admitted_action_batches:
            raise RuntimeExecutionError(
                "fixed layout must be admitted before execution tasks"
            )
        placements = (FixedPlacementDescription * len(layout.placements))(
            *(
                FixedPlacementDescription(
                    task_id=item.task_id,
                    ordinal=item.ordinal,
                    object_id=item.object_id,
                    offset=item.offset,
                    bytes=item.bytes,
                    alignment_bytes=item.alignment,
                    kind=int(item.kind),
                )
                for item in layout.placements
            )
        )
        dependencies = (FixedDependencyDescription * len(layout.dependencies))(
            *(
                FixedDependencyDescription(
                    predecessor_task_id=item.predecessor_task_id,
                    predecessor_action_ordinal=(item.predecessor_action_ordinal),
                    successor_task_id=item.successor_task_id,
                    successor_ordinal=item.successor_ordinal,
                    successor_kind=int(item.successor_kind),
                )
                for item in layout.dependencies
            )
        )
        description = FixedLayoutDescription(
            abi_version=ABI_VERSION,
            slice_bytes=layout.slice_bytes,
            placements=placements if layout.placements else None,
            placement_count=len(layout.placements),
            dependencies=dependencies if layout.dependencies else None,
            dependency_count=len(layout.dependencies),
        )
        status = int(
            self.runtime_library.shadowspill_plan_admit_fixed_layout(
                self.plan_handle, ctypes.byref(description)
            )
        )
        if status != 0:
            runtime = self.statistics().runtime
            raise RuntimeExecutionError(
                "admit fixed physical layout failed: "
                f"status={status}, requested_slice={layout.slice_bytes}, "
                f"allocated={int(runtime.allocated_bytes)}, "
                f"free={int(runtime.free_bytes)}, "
                f"free_prefix={int(runtime.free_prefix_bytes)}, "
                "largest_free_range="
                f"{int(runtime.largest_free_range_bytes)}, "
                "external_fragmentation="
                f"{int(runtime.external_fragmentation_bytes)}, "
                f"live_allocations={int(runtime.live_allocations)}, "
                f"pool_capacity={int(runtime.execution_pool_bytes)}"
            )
        self._fixed_layout_installed = True

    def admit_initial_actions(
        self,
        actions: tuple[MemoryAction, ...],
        *,
        task_number: int,
        action_trace_labels: tuple[str, ...] | None = None,
    ) -> int:
        """Admit one reusable initial-placement action batch."""

        labels = _action_labels(actions, action_trace_labels)
        action_pairs = self._runtime_actions(actions, labels)
        expected = tuple(
            (action.alias_group_id, action.kind) for action, _label in action_pairs
        )
        existing = self._admitted_action_batches.get(task_number)
        if existing is not None:
            existing_handle, admitted = existing
            if admitted != expected:
                raise RuntimeExecutionError(
                    "initial action admission changed for an existing task: "
                    f"task={task_number}, expected={admitted}, observed={expected}"
                )
            return existing_handle
        for action, _label in action_pairs:
            self.register_placeholder(action.alias_group_id)
        self._bind_plan_objects(
            action.alias_group_id for action, _label in action_pairs
        )
        encoded_labels = tuple(
            label.encode("utf-8") if label else None for _action, label in action_pairs
        )
        runtime_actions = (RuntimeAction * len(action_pairs))(
            *(
                _runtime_action(
                    _plan_local_id(action.alias_group_id, "alias_"),
                    _ACTION_KIND[action.kind],
                    trace_label=encoded,
                )
                for (action, _label), encoded in zip(
                    action_pairs, encoded_labels, strict=True
                )
            )
        )
        action_batch_handle = ctypes.c_size_t()
        self._require(
            self.runtime_library.shadowspill_plan_admit_action_batch(
                self.plan_handle,
                task_number,
                runtime_actions if action_pairs else None,
                len(action_pairs),
                ctypes.byref(action_batch_handle),
            ),
            f"admit initial action batch {task_number}",
        )
        if action_batch_handle.value == 0:
            raise RuntimeExecutionError(
                "runtime returned a null initial action batch handle"
            )
        resolved = int(action_batch_handle.value)
        self._admitted_action_batches[task_number] = (resolved, expected)
        return resolved

    def seal_fixed_layout(self) -> None:
        """Resolve the installed certificate after execution admission."""

        if not self._fixed_layout_installed:
            raise RuntimeExecutionError("no fixed physical layout was admitted")
        self._require(
            self.runtime_library.shadowspill_plan_seal_fixed_layout(self.plan_handle),
            "seal fixed physical layout",
        )

    def _runtime_inputs(self, input_alias_ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            alias_id for alias_id in input_alias_ids if self.requires_storage(alias_id)
        )

    def _runtime_mutations(
        self,
        mutations: tuple[MutationSpec, ...],
    ) -> tuple[MutationSpec, ...]:
        return tuple(
            mutation
            for mutation in mutations
            if self.requires_storage(self.alias_for_object(mutation.object_id))
        )

    def _runtime_actions(
        self,
        actions: tuple[MemoryAction, ...],
        labels: tuple[str, ...],
    ) -> tuple[tuple[MemoryAction, str], ...]:
        return tuple(
            (action, label)
            for action, label in zip(actions, labels, strict=True)
            if self.requires_storage(action.alias_group_id)
        )

    def _referenced_aliases(
        self,
        inputs: tuple[str, ...],
        mutations: tuple[MutationSpec, ...],
        publications: tuple[TaskPublication, ...],
        actions: tuple[tuple[MemoryAction, str], ...],
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *inputs,
                    *(self.alias_for_object(item.object_id) for item in mutations),
                    *(item.alias_id for item in publications),
                    *(action.alias_group_id for action, _label in actions),
                )
            )
        )

    def _execution_buffers(
        self,
        task: TaskSpec,
        inputs: tuple[str, ...],
        mutations: tuple[MutationSpec, ...],
        publications: tuple[TaskPublication, ...],
        actions: tuple[tuple[MemoryAction, str], ...],
        memory_envelope: TaskMemoryEnvelope,
        trace_label: str,
    ) -> _ExecutionBuffers:
        input_ids = (ctypes.c_uint64 * len(inputs))(
            *(_plan_local_id(value, "alias_") for value in inputs)
        )
        updates = (ObjectUpdate * len(mutations))(
            *(
                ObjectUpdate(
                    _plan_local_id(self.alias_for_object(item.object_id), "alias_"),
                    item.version_delta,
                )
                for item in mutations
            )
        )
        publication_values = (TaskPublicationDescription * len(publications))(
            *(
                TaskPublicationDescription(
                    object_id=_plan_local_id(item.alias_id, "alias_"),
                    kind=1 if item.replace_lease else 0,
                )
                for item in publications
            )
        )
        labels = tuple(label.encode("utf-8") if label else None for _, label in actions)
        action_values = (RuntimeAction * len(actions))(
            *(
                _runtime_action(
                    _plan_local_id(action.alias_group_id, "alias_"),
                    _ACTION_KIND[action.kind],
                    trace_label=label,
                )
                for (action, _text), label in zip(actions, labels, strict=True)
            )
        )
        allocation_contract = memory_envelope.allocation_contract
        contract_steps = (
            () if allocation_contract is None else allocation_contract.steps
        )
        contract_values = (CTaskAllocationContractStep * len(contract_steps))(
            *(
                CTaskAllocationContractStep(
                    allocation_ordinal=step.allocation_ordinal,
                    requested_bytes=step.requested_bytes,
                    charged_bytes=step.charged_bytes,
                    alignment_bytes=step.alignment_bytes,
                    operation=0 if step.operation.value == "allocate" else 1,
                    required=step.required,
                )
                for step in contract_steps
            )
        )
        encoded_task_label = trace_label.encode("utf-8")
        description = TaskDescription(
            task_id=_plan_local_id(task.task_id, "task_"),
            trace_label=encoded_task_label,
            input_object_ids=input_ids if inputs else None,
            input_count=len(inputs),
            updates=updates if mutations else None,
            update_count=len(mutations),
            publications=publication_values if publications else None,
            publication_count=len(publications),
            actions=action_values if actions else None,
            action_count=len(actions),
            allocation_contract_steps=contract_values if contract_steps else None,
            allocation_contract_step_count=len(contract_steps),
            enforce_allocation_contract=allocation_contract is not None,
            maximum_requested_allocation_bytes=(
                memory_envelope.maximum_requested_allocation_bytes
            ),
            maximum_charged_allocation_bytes=(
                memory_envelope.maximum_charged_allocation_bytes
            ),
            live_requested_allocation_limit_bytes=(
                memory_envelope.live_requested_allocation_limit_bytes
            ),
            live_charged_allocation_limit_bytes=(
                memory_envelope.live_charged_allocation_limit_bytes
            ),
            dynamic_scratch_maximum_allocation_bytes=(
                memory_envelope.dynamic_scratch_maximum_allocation_bytes
            ),
            dynamic_scratch_live_limit_bytes=(
                memory_envelope.dynamic_scratch_live_limit_bytes
            ),
        )
        return _ExecutionBuffers(
            description,
            encoded_task_label,
            input_ids,
            updates,
            publication_values,
            action_values,
            contract_values,
            labels,
        )

    def prepare_runtime_trace(
        self, *, event_capacity: int, allocation_event_capacity: int
    ) -> None:
        """Allocate reusable bounded CPU trace buffers without enabling trace."""

        prepare_runtime_trace(
            self.runtime._runtime_handle,
            event_capacity=event_capacity,
            allocation_event_capacity=allocation_event_capacity,
        )

    def begin_runtime_trace(self, *, step_id: int) -> None:
        begin_runtime_trace(self.runtime._runtime_handle, step_id=step_id)

    def end_and_read_runtime_trace(self) -> CapturedRuntimeTrace:
        end_runtime_trace(self.runtime._runtime_handle)
        return read_runtime_trace(self.runtime._runtime_handle)

    def statistics(self) -> AdapterStatistics:
        result = AdapterStatistics()
        self._require(
            self.library.shadowspill_pytorch_allocator_statistics(ctypes.byref(result)),
            "read runtime statistics",
        )
        return result

    def alias_for_object(self, object_id: str) -> str:
        try:
            return self._alias_by_object[object_id]
        except KeyError as exc:
            raise PlanningError(f"unknown program object {object_id!r}") from exc

    def register_spill_tensor(
        self, alias_id: str, tensor: torch.Tensor, *, retain_spill_copy: bool
    ) -> None:
        if tensor.device.type != "cpu":
            raise PlanningError("initial object payload must be CPU resident")
        storage = tensor.untyped_storage()
        expected = self._size(alias_id)
        if storage.nbytes() != expected:
            raise PlanningError(
                f"host payload for {alias_id!r} has {storage.nbytes()} bytes; "
                f"the plan requires {expected}"
            )
        if expected == 0:
            self._registered.add(alias_id)
            self._zero_generations.setdefault(alias_id, 0)
            return
        runtime_object_id = self._allocate_runtime_object_id(alias_id)
        self._require(
            self.library.shadowspill_pytorch_register_object(
                self.spill_pool_id,
                runtime_object_id,
                expected,
                int(retain_spill_copy),
                storage.data_ptr(),
            ),
            "register host object",
        )
        self._registered.add(alias_id)
        self._bind_plan_object(alias_id)

    def adopt_persistent_object(
        self,
        alias_id: str,
        *,
        current_object_id: int,
        pool_id: int,
        size_bytes: int,
        pool_pointer: int,
    ) -> int:
        """Adopt one preloaded spill lease without allocating or copying it."""

        expected = self._size(alias_id)
        if size_bytes != expected:
            raise PlanningError(
                f"persistent payload for {alias_id!r} has {size_bytes} bytes; "
                f"the plan requires {expected}"
            )
        if pool_id != self.spill_pool_id:
            raise PlanningError(
                f"persistent object for {alias_id!r} resides in pool {pool_id}; "
                f"the plan selected spill pool {self.spill_pool_id}"
            )
        self._require(
            self.library.shadowspill_pytorch_validate_object_binding(
                pool_id, current_object_id, pool_pointer, size_bytes
            ),
            f"validate persistent object {alias_id}",
        )
        self._record_runtime_object(alias_id, current_object_id)
        self._registered.add(alias_id)
        self._bind_plan_object(alias_id)
        return current_object_id

    def adopt_shared_object(
        self,
        alias_id: str,
        reference: ObjectRef,
        *,
        consistency: ObjectConsistency,
    ) -> None:
        """Bind a plan alias to an externally owned runtime object."""

        reference._require_open()
        if not reference._belongs_to(self.runtime):
            raise PlanningError("shared input belongs to another Runtime")
        expected = self._size(alias_id)
        if reference.size_bytes != expected:
            raise PlanningError(
                f"shared input for {alias_id!r} has "
                f"{reference.size_bytes} bytes; the plan requires {expected}"
            )
        runtime_object_id = reference.object_id
        self._record_runtime_object(alias_id, runtime_object_id)
        self._registered.add(alias_id)
        self._borrowed.add(alias_id)
        consistency_code = 0 if consistency is ObjectConsistency.CAUSAL else 1
        self._bind_plan_object(
            alias_id,
            consistency=consistency_code,
        )

    def register_placeholder(self, alias_id: str) -> None:
        """Register a logical alias bundle before its first production."""

        if alias_id in self._registered:
            return
        if not self.requires_storage(alias_id):
            self._registered.add(alias_id)
            self._zero_generations.setdefault(alias_id, 0)
            return
        runtime_object_id = self._allocate_runtime_object_id(alias_id)
        self._require(
            self.library.shadowspill_pytorch_register_placeholder_object(
                runtime_object_id, self._size(alias_id), 0
            ),
            "register placeholder object",
        )
        self._registered.add(alias_id)
        self._bind_plan_object(alias_id)

    def write_spill_tensor(self, alias_id: str, tensor: torch.Tensor) -> None:
        if alias_id not in self._registered:
            raise RuntimeExecutionError(f"object {alias_id!r} is not registered")
        if tensor.device.type != "cpu":
            raise RuntimeExecutionError("runtime input payload must be CPU resident")
        storage = tensor.untyped_storage()
        expected = self._size(alias_id)
        if storage.nbytes() != expected:
            raise RuntimeExecutionError(
                f"runtime storage for {alias_id!r} has {storage.nbytes()} bytes; "
                f"the plan requires {expected}"
            )
        if expected == 0:
            return
        self._require(
            self.runtime_library.shadowspill_write_object(
                self.runtime._runtime_handle,
                self._runtime_object_id(alias_id),
                self.spill_pool_id,
                storage.data_ptr(),
                expected,
            ),
            "write host object",
        )

    def read_spill_tensor(self, alias_id: str, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu":
            raise RuntimeExecutionError("writeback destination must be CPU resident")
        storage = tensor.untyped_storage()
        expected = self._size(alias_id)
        if storage.nbytes() != expected:
            raise RuntimeExecutionError(
                f"writeback storage for {alias_id!r} has {storage.nbytes()} bytes; "
                f"the plan requires {expected}"
            )
        if expected == 0:
            return
        self._require(
            self.runtime_library.shadowspill_read_object(
                self.runtime._runtime_handle,
                self._runtime_object_id(alias_id),
                self.spill_pool_id,
                storage.data_ptr(),
                expected,
            ),
            "read host object",
        )

    def unregister(self, alias_ids: Iterable[str]) -> None:
        for alias_id in dict.fromkeys(alias_ids):
            if alias_id not in self._registered:
                continue
            if alias_id in self._borrowed:
                self._borrowed.remove(alias_id)
                self._registered.remove(alias_id)
                self._runtime_object_ids.pop(alias_id, None)
                self._binding_consistency.pop(alias_id, None)
                continue
            if not self.requires_storage(alias_id):
                self._registered.remove(alias_id)
                self._zero_generations.pop(alias_id, None)
                continue
            self._require(
                self.runtime_library.shadowspill_unregister_object(
                    self.runtime._runtime_handle,
                    self._runtime_object_id(alias_id),
                ),
                "unregister object",
            )
            self._registered.remove(alias_id)
            self._runtime_object_ids.pop(alias_id, None)
            self._binding_consistency.pop(alias_id, None)

    def publish_initial_tensor(
        self, alias_id: str, tensor: torch.Tensor
    ) -> ObjectBinding:
        """Publish cold materialization through the plan-local object record."""

        if not self.requires_storage(alias_id):
            self._registered.add(alias_id)
            return self._zero_binding(alias_id)
        binding = ObjectBinding()
        storage = tensor.untyped_storage()
        self._require(
            self.runtime_library.shadowspill_plan_publish_initial_allocation(
                self.plan_handle,
                _plan_local_id(alias_id, "alias_"),
                storage.data_ptr(),
                ctypes.byref(binding),
            ),
            "publish initial plan allocation",
        )
        return binding

    def acquire_for_caller(
        self,
        alias_ids: tuple[str, ...],
        tensors: tuple[torch.Tensor, ...],
        *,
        acquisition_handle: int,
    ) -> tuple[ObjectBinding, ...]:
        """Acquire an admitted public-object set without opening a task."""

        if len(alias_ids) != len(tensors):
            raise RuntimeExecutionError("caller output binding count differs")
        runtime_aliases = tuple(
            alias_id for alias_id in alias_ids if self.requires_storage(alias_id)
        )
        if not runtime_aliases:
            if acquisition_handle != 0:
                raise RuntimeExecutionError(
                    "zero-byte caller outputs must not own an acquisition handle"
                )
            return self._expand_bindings(alias_ids, (), ())
        admitted = self._admitted_acquisitions.get(runtime_aliases)
        if admitted != acquisition_handle or acquisition_handle == 0:
            raise RuntimeExecutionError(
                "caller output acquisition does not match its admitted object set"
            )
        stream = torch.cuda.current_stream()
        bindings = (ObjectBinding * len(runtime_aliases))()
        self._require(
            self.library.shadowspill_pytorch_acquire_objects_handle(
                acquisition_handle,
                stream.cuda_stream,
                bindings if runtime_aliases else None,
                len(runtime_aliases),
            ),
            "acquire caller outputs",
        )
        expanded = self._expand_bindings(alias_ids, runtime_aliases, bindings)
        for alias_id, tensor, binding in zip(alias_ids, tensors, expanded, strict=True):
            self.rebind(tensor, alias_id, binding)
        return expanded

    def admit_caller_acquisition(self, alias_ids: tuple[str, ...]) -> int:
        """Admit one immutable ordered public-object acquisition handle."""

        runtime_aliases = tuple(
            alias_id for alias_id in alias_ids if self.requires_storage(alias_id)
        )
        existing = self._admitted_acquisitions.get(runtime_aliases)
        if existing is not None:
            return existing
        if not runtime_aliases:
            return 0
        for alias_id in dict.fromkeys(runtime_aliases):
            self.register_placeholder(alias_id)
        self._bind_plan_objects(runtime_aliases)
        identifiers = (ctypes.c_uint64 * len(runtime_aliases))(
            *(_plan_local_id(value, "alias_") for value in runtime_aliases)
        )
        handle = ctypes.c_size_t()
        self._require(
            self.runtime_library.shadowspill_plan_admit_object_acquisition(
                self.plan_handle,
                identifiers,
                len(runtime_aliases),
                ctypes.byref(handle),
            ),
            "admit caller output acquisition",
        )
        if handle.value == 0:
            raise RuntimeExecutionError(
                "runtime returned a null caller output acquisition handle"
            )
        self._admitted_acquisitions[runtime_aliases] = int(handle.value)
        return int(handle.value)

    def submit_initial_actions(
        self,
        actions: tuple[MemoryAction, ...],
        *,
        task_number: int,
    ) -> None:
        runtime_actions_values = tuple(
            item for item in actions if self.requires_storage(item.alias_group_id)
        )
        if not runtime_actions_values:
            return
        stream = torch.cuda.current_stream()
        admitted = self._admitted_action_batches.get(task_number)
        if admitted is None:
            raise RuntimeExecutionError(
                f"initial action batch {task_number} was not admitted"
            )
        handle, expected = admitted
        observed = tuple(
            (action.alias_group_id, action.kind) for action in runtime_actions_values
        )
        if observed != expected:
            raise RuntimeExecutionError(
                "initial action batch changed after admission: "
                f"task={task_number}, expected={expected}, observed={observed}"
            )
        self._require(
            self.library.shadowspill_pytorch_submit_action_batch_handle(
                handle, stream.cuda_stream
            ),
            "submit admitted initial actions",
        )

    def clear_tasks(self) -> None:
        """Discard immutable task records and their fixed layout."""

        self._require(
            self.runtime_library.shadowspill_plan_clear_tasks(self.plan_handle),
            "clear plan tasks",
        )
        self._admitted_task_handles.clear()
        self._admitted_action_batches.clear()
        self._admitted_acquisitions.clear()
        self._fixed_layout_installed = False

    def wait_plan_idle(self) -> None:
        """Actively wait only for work owned by this admitted plan."""

        self._require(
            self.runtime_library.shadowspill_plan_wait_idle(self.plan_handle),
            "wait for plan idle",
        )

    def transfer_outputs_to_caller(
        self,
        alias_ids: tuple[str, ...],
        tensors: tuple[torch.Tensor, ...],
        bindings: tuple[ObjectBinding, ...],
        *,
        acquisition_handle: int,
    ) -> None:
        if not (len(alias_ids) == len(tensors) == len(bindings)):
            raise RuntimeExecutionError("caller output lease count differs")
        seen: set[str] = set()
        for object_ordinal, (alias_id, tensor, binding) in enumerate(
            zip(alias_ids, tensors, bindings, strict=True)
        ):
            if alias_id in seen:
                continue
            if not self.requires_storage(alias_id):
                self._zero_generations.pop(alias_id, None)
                seen.add(alias_id)
                continue
            torch.ops.shadowspill._transfer_acquired_storage_to_caller(
                tensor,
                acquisition_handle,
                object_ordinal,
                binding.generation,
                binding.allocation_id,
            )
            seen.add(alias_id)

    def rebind(
        self, tensor: torch.Tensor, alias_id: str, binding: ObjectBinding
    ) -> None:
        if not self.requires_storage(alias_id):
            return
        torch.ops.shadowspill._acquire_storages([tensor], [binding.pointer])

    def rebind_many(
        self,
        items: Sequence[tuple[torch.Tensor, str, ObjectBinding]],
    ) -> None:
        """Install bindings already validated by the runtime task boundary."""

        materialized = tuple(item for item in items if self.requires_storage(item[1]))
        if not materialized:
            return
        torch.ops.shadowspill._acquire_storages(
            [tensor for tensor, _, _ in materialized],
            [binding.pointer for _, _, binding in materialized],
        )

    def before_task_and_acquire(
        self,
        task_handle: int,
        device_ordinal: int,
        tensors: Sequence[torch.Tensor],
    ) -> None:
        """Acquire one predecoded storage-only input vector."""

        torch.ops.shadowspill._before_task_storages(
            tensors,
            task_handle,
            device_ordinal,
        )

    def after_task_and_update(
        self,
        task_handle: int,
        device_ordinal: int,
        adopted: Sequence[PublishedStorage],
        publication_ordinals: Sequence[int],
        dematerialized: Sequence[torch.Tensor],
        *,
        replacements: Sequence[ReplacementStorageViews] = (),
    ) -> None:
        """Overwrite logical objects and publish one admitted task boundary."""

        if len(adopted) != len(publication_ordinals):
            raise RuntimeExecutionError(
                "task publication tensors and ordinals have different lengths"
            )
        if not replacements:
            torch.ops.shadowspill._after_task_storages(
                tuple(item.tensor for item in adopted),
                publication_ordinals,
                (),
                (),
                dematerialized,
                task_handle,
                device_ordinal,
            )
            return
        materialized = tuple(enumerate(adopted))
        replacement_by_alias = {item.alias_id: item for item in replacements}
        if len(replacement_by_alias) != len(replacements):
            raise RuntimeExecutionError("task replacement aliases are not unique")
        replacement_aliases = frozenset(replacement_by_alias)
        adopted_aliases = {item.alias_id for item in adopted}
        unknown_replacements = replacement_aliases - adopted_aliases
        if unknown_replacements:
            raise RuntimeExecutionError(
                "task replacement has no adopted output: "
                f"{sorted(unknown_replacements)}"
            )
        replacement_tensors: list[torch.Tensor] = []
        replacement_target_indices: list[int] = []
        for target_index, (_index, item) in enumerate(materialized):
            replacement = replacement_by_alias.get(item.alias_id)
            if replacement is None:
                continue
            for tensor in replacement.tensors:
                replacement_tensors.append(tensor)
                replacement_target_indices.append(target_index)
        torch.ops.shadowspill._after_task_storages(
            tuple(item.tensor for _, item in materialized),
            publication_ordinals,
            replacement_tensors,
            replacement_target_indices,
            dematerialized,
            task_handle,
            device_ordinal,
        )

    def current_generation(self, alias_id: str) -> int:
        """Snapshot one public object's authoritative runtime generation."""

        if alias_id not in self._registered or not self.requires_storage(alias_id):
            raise RuntimeExecutionError(
                f"cannot snapshot unregistered plan object {alias_id!r}"
            )
        snapshot = ObjectSnapshot()
        self._require(
            self.runtime_library.shadowspill_object_snapshot(
                self.runtime._runtime_handle,
                self._runtime_object_id(alias_id),
                ctypes.byref(snapshot),
            ),
            f"snapshot object {alias_id}",
        )
        return int(snapshot.generation)

    def dematerialize(
        self, tensor: torch.Tensor, alias_id: str, generation: int
    ) -> None:
        del alias_id, generation
        if tensor.untyped_storage().data_ptr() == 0:
            return
        torch.ops.shadowspill._dematerialize_storages([tensor])

    def dematerialize_many(
        self,
        items: Sequence[tuple[torch.Tensor, str, int]],
    ) -> None:
        """Transactionally dematerialize distinct alias bundles."""

        materialized = tuple(
            item for item in items if item[0].untyped_storage().data_ptr() != 0
        )
        if not materialized:
            return
        torch.ops.shadowspill._dematerialize_storages(
            [tensor for tensor, _, _ in materialized]
        )

    def wait_idle(self) -> None:
        """Wait for runtime-global quiescence at lifecycle boundaries."""

        self._require(
            self.library.shadowspill_pytorch_allocator_wait_idle(), "wait idle"
        )

    def input_failure_states(self, alias_ids: Iterable[str]) -> tuple[str, ...]:
        """Describe unavailable inputs without changing runtime state."""

        result: list[str] = []
        for alias_id in dict.fromkeys(alias_ids):
            snapshot = ObjectSnapshot()
            status = int(
                self.runtime_library.shadowspill_object_snapshot(
                    self.runtime._runtime_handle,
                    self._runtime_object_id(alias_id),
                    ctypes.byref(snapshot),
                )
            )
            if status != 0:
                result.append(f"{alias_id}:snapshot_status={status}")
                continue
            allocation_id = int(snapshot.allocation_id)
            pointer = int(snapshot.execution_pointer or 0)
            if int(snapshot.residency) in {1, 2} and pointer != 0:
                continue
            result.append(
                f"{alias_id}:residency={int(snapshot.residency)},"
                f"allocation={allocation_id},generation={int(snapshot.generation)},"
                f"pointer={pointer},spill_current={int(snapshot.spill_current)}"
            )
        return tuple(result)

    def abort_task(self, task_handle: int) -> None:
        """Close the matching admitted task scope after frontend failure."""

        self._require(
            self.library.shadowspill_pytorch_abort_task_handle(task_handle),
            "abort admitted task",
        )

    def raise_if_allocator_failed(self, operation: str) -> None:
        """Raise the first callback failure without touching the device timeline."""

        raise_if_allocator_failed(self.library, operation)

    def registered_aliases(self) -> frozenset[str]:
        return frozenset(self._registered)

    def registered_runtime_objects(self) -> Mapping[str, int]:
        """Return plan aliases and their stable runtime object identities."""

        return {
            alias_id: runtime_object_id
            for alias_id, runtime_object_id in self._runtime_object_ids.items()
            if alias_id in self._registered
        }

    def adopt_registered(self, objects: Mapping[str, int]) -> None:
        """Adopt objects registered by a compatible provisional bridge."""

        for alias_id, runtime_object_id in objects.items():
            self._size(alias_id)
            self._registered.add(alias_id)
            self._record_runtime_object(alias_id, runtime_object_id)
            self._bind_plan_object(alias_id)
            if not self.requires_storage(alias_id):
                self._zero_generations.setdefault(alias_id, 0)

    def requires_storage(self, alias_id: str) -> bool:
        """Return whether an alias bundle owns any physical payload bytes."""

        return self._size(alias_id) != 0

    def _zero_binding(self, alias_id: str) -> ObjectBinding:
        if self.requires_storage(alias_id):
            raise AssertionError("zero binding requested for a materialized alias")
        return ObjectBinding(
            _plan_local_id(alias_id, "alias_"),
            self._zero_generations.setdefault(alias_id, 0),
            0,
            0,
            None,
        )

    def _expand_bindings(
        self,
        aliases: Sequence[str],
        materialized_aliases: Sequence[str],
        materialized_bindings: ctypes.Array[ObjectBinding] | Sequence[ObjectBinding],
    ) -> tuple[ObjectBinding, ...]:
        if len(materialized_aliases) != len(materialized_bindings):
            raise RuntimeExecutionError("runtime binding count differs")
        iterator = iter(materialized_bindings)
        result: list[ObjectBinding] = []
        for alias_id in aliases:
            result.append(
                next(iterator)
                if self.requires_storage(alias_id)
                else self._zero_binding(alias_id)
            )
        try:
            next(iterator)
        except StopIteration:
            return tuple(result)
        raise RuntimeExecutionError("runtime returned excess object bindings")

    def _size(self, alias_id: str) -> int:
        try:
            return self._size_by_alias[alias_id]
        except KeyError as exc:
            raise PlanningError(f"unknown alias group {alias_id!r}") from exc

    def _require(self, raw_status: Any, operation: str) -> None:
        status = int(raw_status)
        if status == 0:
            return
        diagnostics = read_allocator_failure(self.library, operation)
        if diagnostics is not None:
            raise generic_runtime_error(diagnostics)
        raise RuntimeExecutionError(f"{operation} failed with status {status}")


def actions_by_task(
    actions: Iterable[MemoryAction],
) -> Mapping[str, tuple[MemoryAction, ...]]:
    """Preserve schedule ordering while grouping actions by trigger task."""

    result: dict[str, list[MemoryAction]] = {}
    for action in actions:
        result.setdefault(action.trigger_task_id, []).append(action)
    return {task_id: tuple(values) for task_id, values in result.items()}


__all__ = ["RuntimeBridge", "RuntimeExecutionError", "actions_by_task"]
