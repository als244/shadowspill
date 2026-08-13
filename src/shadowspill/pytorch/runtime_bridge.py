"""Narrow high-level bridge over the private PyTorch adapter ABI."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from shadowspill.ir import (
    MemoryAction,
    MemoryActionKind,
    MutationSpec,
    Program,
    TaskSpec,
)

from ._abi import (
    AdapterFailure,
    AdapterStatistics,
    ExecutionDescription,
    ObjectBinding,
    ObjectSnapshot,
    ObjectUpdate,
    RuntimeAction,
    TaskHostTiming,
)
from ._runtime_trace import (
    CapturedRuntimeTrace,
    begin_runtime_trace,
    end_runtime_trace,
    prepare_runtime_trace,
    read_runtime_trace,
)
from .contracts import PlanningError


class RuntimeExecutionError(RuntimeError):
    """A native runtime transition rejected the immutable execution plan."""


_ACTION_KIND = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.OFFLOAD: 1,
    MemoryActionKind.PREFETCH: 2,
}


@dataclass(frozen=True, slots=True)
class TaskHostTimestamps:
    before_task_enter_ns: int
    before_task_exit_ns: int
    after_task_enter_ns: int
    after_task_exit_ns: int


def _dense_id(value: str, prefix: str) -> int:
    if not value.startswith(prefix):
        raise PlanningError(f"dense identity {value!r} does not start with {prefix!r}")
    suffix = value.removeprefix(prefix)
    if not suffix.isdigit():
        raise PlanningError(f"dense identity {value!r} has a nonnumeric suffix")
    return int(suffix)


class RuntimeBridge:
    """Translate canonical string identities into the adapter's dense C ABI."""

    def __init__(self, library: Any, program: Program) -> None:
        self.library = library
        self._alias_by_object: dict[str, str] = {
            item.object_id: item.alias_group_id for item in program.objects
        }
        self._size_by_alias: dict[str, int] = {
            item.alias_group_id: item.size_bytes for item in program.alias_groups
        }
        self._zero_generations: dict[str, int] = {}
        self._registered: set[str] = set()
        self._debug_task_timing_capacity = 0
        self._admitted_tasks: dict[str, tuple[str, ...]] = {}

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

    def admit_execution(
        self,
        task: TaskSpec,
        input_alias_ids: tuple[str, ...],
        actions: tuple[MemoryAction, ...],
        action_trace_labels: tuple[str, ...] | None = None,
    ) -> int:
        """Resolve one immutable task topology in the neutral runtime."""

        if action_trace_labels is None:
            action_trace_labels = ("",) * len(actions)
        if len(action_trace_labels) != len(actions):
            raise ValueError("action trace labels must align with ordered actions")

        runtime_inputs = tuple(
            alias_id for alias_id in input_alias_ids if self.requires_storage(alias_id)
        )
        mutation_values = tuple(
            item
            for item in task.mutations
            if self.requires_storage(self.alias_for_object(item.object_id))
        )
        runtime_action_pairs = tuple(
            (action, label)
            for action, label in zip(
                actions, action_trace_labels, strict=True
            )
            if self.requires_storage(action.alias_group_id)
        )
        referenced_aliases = tuple(
            dict.fromkeys(
                (
                    *runtime_inputs,
                    *(
                        self.alias_for_object(item.object_id)
                        for item in mutation_values
                    ),
                    *(item.alias_group_id for item, _ in runtime_action_pairs),
                )
            )
        )
        for alias_id in referenced_aliases:
            self.register_placeholder(alias_id)
        identifiers = (ctypes.c_uint64 * len(runtime_inputs))(
            *(_dense_id(value, "alias_") for value in runtime_inputs)
        )
        updates = (ObjectUpdate * len(mutation_values))(
            *(
                ObjectUpdate(
                    _dense_id(self.alias_for_object(item.object_id), "alias_"),
                    item.version_delta,
                )
                for item in mutation_values
            )
        )
        runtime_actions_values = tuple(item for item, _ in runtime_action_pairs)
        encoded_action_labels = tuple(
            label.encode("utf-8") if label else None
            for _, label in runtime_action_pairs
        )
        runtime_actions = (RuntimeAction * len(runtime_actions_values))(
            *(
                RuntimeAction(
                    _dense_id(item.alias_group_id, "alias_"),
                    _ACTION_KIND[item.kind],
                    label,
                )
                for item, label in zip(
                    runtime_actions_values, encoded_action_labels, strict=True
                )
            )
        )
        description = ExecutionDescription(
            task_id=_dense_id(task.task_id, "task_"),
            input_object_ids=identifiers if runtime_inputs else None,
            input_count=len(runtime_inputs),
            updates=updates if mutation_values else None,
            update_count=len(mutation_values),
            actions=runtime_actions if runtime_actions_values else None,
            action_count=len(runtime_actions_values),
        )
        self._require(
            self.library.shadowspill_pytorch_admit_execution(ctypes.byref(description)),
            f"admit execution {task.task_id}",
        )
        self._admitted_tasks[task.task_id] = runtime_inputs
        handle = ctypes.c_size_t()
        self._require(
            self.library.shadowspill_pytorch_resolve_execution(
                _dense_id(task.task_id, "task_"), ctypes.byref(handle)
            ),
            f"resolve execution {task.task_id}",
        )
        if handle.value == 0:
            raise RuntimeExecutionError(
                f"execution {task.task_id} resolved to a null handle"
            )
        return int(handle.value)

    def before_execution(
        self,
        execution_handle: int,
        task_id: str,
        stream: torch.cuda.Stream,
        input_count: int,
    ) -> tuple[ObjectBinding, ...]:
        bindings = (ObjectBinding * input_count)()
        self._require(
            self.library.shadowspill_pytorch_before_execution_handle(
                execution_handle,
                _dense_id(task_id, "task_"),
                stream.cuda_stream,
                bindings if input_count else None,
                input_count,
            ),
            f"before admitted task {task_id}",
        )
        return tuple(bindings)

    def after_execution(
        self,
        execution_handle: int,
        task_id: str,
        stream: torch.cuda.Stream,
    ) -> None:
        self._require(
            self.library.shadowspill_pytorch_after_execution_handle(
                execution_handle,
                _dense_id(task_id, "task_"),
                stream.cuda_stream,
            ),
            f"after admitted task {task_id}",
        )

    def enable_debug_task_timing(self, task_ids: Iterable[str]) -> None:
        """Enable optional compute-stream host callbacks for selected tasks."""

        dense_ids = tuple(_dense_id(task_id, "task_") for task_id in task_ids)
        capacity = max(dense_ids, default=-1) + 1
        if capacity <= 0:
            raise RuntimeExecutionError("debug task timing requires a task")
        self._require(
            self.library.shadowspill_pytorch_debug_task_timing_enable(capacity),
            "enable debug task timing",
        )
        self._debug_task_timing_capacity = capacity

    def read_debug_task_timing(
        self,
    ) -> dict[str, TaskHostTimestamps]:
        """Read host-callback timestamps after the compute stream is idle."""

        capacity = self._debug_task_timing_capacity
        if capacity <= 0:
            return {}
        records = (TaskHostTiming * capacity)()
        count = ctypes.c_uint32()
        self._require(
            self.library.shadowspill_pytorch_debug_task_timing_read(
                records, capacity, ctypes.byref(count)
            ),
            "read debug task timing",
        )
        return {
            f"task_{int(item.task_id):06d}": TaskHostTimestamps(
                before_task_enter_ns=int(item.before_task_enter_timestamp_ns),
                before_task_exit_ns=int(item.before_task_exit_timestamp_ns),
                after_task_enter_ns=int(item.after_task_enter_timestamp_ns),
                after_task_exit_ns=int(item.after_task_exit_timestamp_ns),
            )
            for item in records[: count.value]
        }

    def disable_debug_task_timing(self) -> None:
        if self._debug_task_timing_capacity <= 0:
            return
        self._require(
            self.library.shadowspill_pytorch_debug_task_timing_disable(),
            "disable debug task timing",
        )
        self._debug_task_timing_capacity = 0

    def configure_task_labels(self, labels_by_task: Mapping[str, str]) -> None:
        """Install cold-path semantic NVTX labels for dense task identities."""

        dense_labels = {
            _dense_id(task_id, "task_"): label
            for task_id, label in labels_by_task.items()
        }
        capacity = max(dense_labels, default=-1) + 1
        if capacity == 0:
            self._require(
                self.library.shadowspill_pytorch_task_labels_configure(None, 0),
                "clear task trace labels",
            )
            return
        encoded = [dense_labels.get(index, "").encode() for index in range(capacity)]
        values = (ctypes.c_char_p * capacity)(*encoded)
        self._require(
            self.library.shadowspill_pytorch_task_labels_configure(values, capacity),
            "configure task trace labels",
        )

    def prepare_runtime_trace(
        self, *, event_capacity: int, allocation_event_capacity: int
    ) -> None:
        """Allocate reusable bounded CPU trace buffers without enabling trace."""

        prepare_runtime_trace(
            self.library,
            event_capacity=event_capacity,
            allocation_event_capacity=allocation_event_capacity,
        )

    def begin_runtime_trace(self, *, step_id: int) -> None:
        begin_runtime_trace(self.library, step_id=step_id)

    def end_and_read_runtime_trace(self) -> CapturedRuntimeTrace:
        end_runtime_trace(self.library)
        return read_runtime_trace(self.library)

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

    def register_host_tensor(
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
        self._require(
            self.library.shadowspill_pytorch_register_host_object(
                _dense_id(alias_id, "alias_"),
                expected,
                int(retain_spill_copy),
                storage.data_ptr(),
            ),
            "register host object",
        )
        self._registered.add(alias_id)

    def register_placeholder(self, alias_id: str) -> None:
        """Register a logical alias bundle before its first production."""

        if alias_id in self._registered:
            return
        if not self.requires_storage(alias_id):
            self._registered.add(alias_id)
            self._zero_generations.setdefault(alias_id, 0)
            return
        self._require(
            self.library.shadowspill_pytorch_register_placeholder_object(
                _dense_id(alias_id, "alias_"), self._size(alias_id), 0
            ),
            "register placeholder object",
        )
        self._registered.add(alias_id)

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
            self.library.shadowspill_pytorch_write_spill_object(
                _dense_id(alias_id, "alias_"), expected, storage.data_ptr()
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
            self.library.shadowspill_pytorch_read_spill_object(
                _dense_id(alias_id, "alias_"), expected, storage.data_ptr()
            ),
            "read host object",
        )

    def unregister(self, alias_ids: Iterable[str]) -> None:
        for alias_id in dict.fromkeys(alias_ids):
            if alias_id not in self._registered:
                continue
            if not self.requires_storage(alias_id):
                self._registered.remove(alias_id)
                self._zero_generations.pop(alias_id, None)
                continue
            self._require(
                self.library.shadowspill_pytorch_unregister_object(
                    _dense_id(alias_id, "alias_")
                ),
                "unregister object",
            )
            self._registered.remove(alias_id)

    def bind_registered_tensor(
        self, alias_id: str, tensor: torch.Tensor
    ) -> ObjectBinding:
        if not self.requires_storage(alias_id):
            self._registered.add(alias_id)
            return self._zero_binding(alias_id)
        binding = ObjectBinding()
        storage = tensor.untyped_storage()
        self._require(
            self.library.shadowspill_pytorch_bind_registered_allocation(
                _dense_id(alias_id, "alias_"),
                storage.data_ptr(),
                self._size(alias_id),
                ctypes.byref(binding),
            ),
            "bind registered allocation",
        )
        return binding

    def promote_output(self, alias_id: str, tensor: torch.Tensor) -> ObjectBinding:
        if not self.requires_storage(alias_id):
            self._registered.add(alias_id)
            return self._zero_binding(alias_id)
        binding = ObjectBinding()
        storage = tensor.untyped_storage()
        function = (
            self.library.shadowspill_pytorch_bind_registered_allocation
            if alias_id in self._registered
            else self.library.shadowspill_pytorch_promote_allocation
        )
        self._require(
            function(
                _dense_id(alias_id, "alias_"),
                storage.data_ptr(),
                self._size(alias_id),
                ctypes.byref(binding),
            ),
            "bind task output",
        )
        self._registered.add(alias_id)
        return binding

    def replace_output(self, alias_id: str, tensor: torch.Tensor) -> ObjectBinding:
        """Install a fresh functional output as an existing object's lease."""

        if alias_id not in self._registered:
            raise RuntimeExecutionError(
                f"replacement object {alias_id!r} is not registered"
            )
        if not self.requires_storage(alias_id):
            self._zero_generations[alias_id] = (
                self._zero_generations.get(alias_id, 0) + 1
            )
            return self._zero_binding(alias_id)
        binding = ObjectBinding()
        storage = tensor.untyped_storage()
        self._require(
            self.library.shadowspill_pytorch_replace_registered_allocation(
                _dense_id(alias_id, "alias_"),
                storage.data_ptr(),
                self._size(alias_id),
                ctypes.byref(binding),
            ),
            "replace task output",
        )
        return binding

    def adopt_many(
        self,
        items: Sequence[tuple[torch.Tensor, str]],
        *,
        replacement_aliases: frozenset[str] = frozenset(),
    ) -> tuple[int, ...]:
        """Adopt task outputs and replace their owning storage in one call."""

        if not items:
            return ()
        materialized = tuple(
            (index, tensor, alias_id)
            for index, (tensor, alias_id) in enumerate(items)
            if self.requires_storage(alias_id)
        )
        generations = (
            torch.ops.shadowspill._adopt_storages(
                [tensor for _, tensor, _ in materialized],
                [_dense_id(alias_id, "alias_") for _, _, alias_id in materialized],
                [self._size(alias_id) for _, _, alias_id in materialized],
                [
                    2
                    if alias_id in replacement_aliases
                    else int(alias_id in self._registered)
                    for _, _, alias_id in materialized
                ],
            )
            if materialized
            else ()
        )
        if len(generations) != len(materialized):
            raise RuntimeExecutionError(
                "storage adoption returned the wrong generation count"
            )
        self._registered.update(alias_id for _, alias_id in items)
        result = [0] * len(items)
        for (index, _tensor, _alias_id), generation in zip(
            materialized, generations, strict=True
        ):
            result[index] = int(generation)
        for index, (_tensor, alias_id) in enumerate(items):
            if self.requires_storage(alias_id):
                continue
            if alias_id in replacement_aliases:
                self._zero_generations[alias_id] = (
                    self._zero_generations.get(alias_id, 0) + 1
                )
            result[index] = self._zero_generations.setdefault(alias_id, 0)
        return tuple(result)

    def before_task(
        self,
        task_id: str,
        stream: torch.cuda.Stream,
        input_alias_ids: tuple[str, ...],
    ) -> tuple[ObjectBinding, ...]:
        runtime_inputs = tuple(
            alias_id for alias_id in input_alias_ids if self.requires_storage(alias_id)
        )
        bindings = (ObjectBinding * len(runtime_inputs))()
        admitted_inputs = self._admitted_tasks.get(task_id)
        if admitted_inputs is not None:
            if admitted_inputs != runtime_inputs:
                raise RuntimeExecutionError(
                    f"admitted task {task_id} input storage contract changed"
                )
            self._require(
                self.library.shadowspill_pytorch_before_execution(
                    _dense_id(task_id, "task_"),
                    stream.cuda_stream,
                    bindings if runtime_inputs else None,
                    len(runtime_inputs),
                ),
                f"before admitted task {task_id}",
            )
            return self._expand_bindings(input_alias_ids, runtime_inputs, bindings)
        identifiers = (ctypes.c_uint64 * len(runtime_inputs))(
            *(_dense_id(value, "alias_") for value in runtime_inputs)
        )
        self._require(
            self.library.shadowspill_pytorch_before_task(
                _dense_id(task_id, "task_"),
                stream.cuda_stream,
                identifiers if runtime_inputs else None,
                len(runtime_inputs),
                bindings if runtime_inputs else None,
                len(runtime_inputs),
            ),
            f"before task {task_id}",
        )
        return self._expand_bindings(input_alias_ids, runtime_inputs, bindings)

    def acquire_for_caller(
        self,
        alias_ids: tuple[str, ...],
        tensors: tuple[torch.Tensor, ...],
        *,
        task_number: int,
    ) -> tuple[ObjectBinding, ...]:
        """Wait/rebind final caller outputs without host synchronization."""

        if len(alias_ids) != len(tensors):
            raise RuntimeExecutionError("caller output binding count differs")
        runtime_aliases = tuple(
            alias_id for alias_id in alias_ids if self.requires_storage(alias_id)
        )
        stream = torch.cuda.current_stream()
        identifiers = (ctypes.c_uint64 * len(runtime_aliases))(
            *(_dense_id(value, "alias_") for value in runtime_aliases)
        )
        bindings = (ObjectBinding * len(runtime_aliases))()
        task_open = False
        try:
            self._require(
                self.library.shadowspill_pytorch_before_task(
                    task_number,
                    stream.cuda_stream,
                    identifiers if runtime_aliases else None,
                    len(runtime_aliases),
                    bindings if runtime_aliases else None,
                    len(runtime_aliases),
                ),
                "acquire caller outputs",
            )
            task_open = True
            expanded = self._expand_bindings(alias_ids, runtime_aliases, bindings)
            for alias_id, tensor, binding in zip(
                alias_ids, tensors, expanded, strict=True
            ):
                self.rebind(tensor, alias_id, binding)
            self._require(
                self.library.shadowspill_pytorch_after_task(
                    task_number, stream.cuda_stream, None, 0, None, 0
                ),
                "close caller output acquisition",
            )
            task_open = False
            return expanded
        finally:
            if task_open:
                self.abort_task()

    def after_task(
        self,
        task_id: str,
        stream: torch.cuda.Stream,
        mutations: Iterable[MutationSpec],
        actions: Iterable[MemoryAction],
    ) -> None:
        if task_id in self._admitted_tasks:
            self._require(
                self.library.shadowspill_pytorch_after_execution(
                    _dense_id(task_id, "task_"), stream.cuda_stream
                ),
                f"after admitted task {task_id}",
            )
            return
        mutation_values = tuple(
            item
            for item in mutations
            if self.requires_storage(self.alias_for_object(item.object_id))
        )
        action_values = tuple(
            item for item in actions if self.requires_storage(item.alias_group_id)
        )
        updates = (ObjectUpdate * len(mutation_values))(
            *(
                ObjectUpdate(
                    _dense_id(self.alias_for_object(item.object_id), "alias_"),
                    item.version_delta,
                )
                for item in mutation_values
            )
        )
        runtime_actions = (RuntimeAction * len(action_values))(
            *(
                RuntimeAction(
                    _dense_id(item.alias_group_id, "alias_"),
                    _ACTION_KIND[item.kind],
                )
                for item in action_values
            )
        )
        self._require(
            self.library.shadowspill_pytorch_after_task(
                _dense_id(task_id, "task_"),
                stream.cuda_stream,
                updates if mutation_values else None,
                len(mutation_values),
                runtime_actions if action_values else None,
                len(action_values),
            ),
            f"after task {task_id}",
        )

    def submit_initial_actions(
        self, actions: tuple[MemoryAction, ...], *, task_number: int
    ) -> None:
        runtime_actions_values = tuple(
            item for item in actions if self.requires_storage(item.alias_group_id)
        )
        if not runtime_actions_values:
            return
        stream = torch.cuda.current_stream()
        runtime_actions = (RuntimeAction * len(runtime_actions_values))(
            *(
                RuntimeAction(
                    _dense_id(item.alias_group_id, "alias_"),
                    _ACTION_KIND[item.kind],
                )
                for item in runtime_actions_values
            )
        )
        self._require(
            self.library.shadowspill_pytorch_after_task(
                task_number,
                stream.cuda_stream,
                None,
                0,
                runtime_actions,
                len(runtime_actions_values),
            ),
            "submit initial actions",
        )

    def transfer_outputs_to_caller(
        self,
        alias_ids: tuple[str, ...],
        tensors: tuple[torch.Tensor, ...],
        bindings: tuple[ObjectBinding, ...],
    ) -> None:
        if not (len(alias_ids) == len(tensors) == len(bindings)):
            raise RuntimeExecutionError("caller output lease count differs")
        seen: set[str] = set()
        for alias_id, tensor, binding in zip(alias_ids, tensors, bindings, strict=True):
            if alias_id in seen:
                continue
            if not self.requires_storage(alias_id):
                self._registered.discard(alias_id)
                self._zero_generations.pop(alias_id, None)
                seen.add(alias_id)
                continue
            torch.ops.shadowspill._transfer_storage_to_caller(
                tensor,
                _dense_id(alias_id, "alias_"),
                binding.generation,
                binding.allocation_id,
            )
            seen.add(alias_id)
            self._registered.discard(alias_id)

    def rebind(
        self, tensor: torch.Tensor, alias_id: str, binding: ObjectBinding
    ) -> None:
        if not self.requires_storage(alias_id):
            return
        torch.ops.shadowspill._rebind_storage(
            tensor,
            binding.pointer,
            _dense_id(alias_id, "alias_"),
            binding.generation,
        )

    def rebind_many(
        self,
        items: Sequence[tuple[torch.Tensor, str, ObjectBinding]],
    ) -> None:
        """Install bindings already validated by the runtime task boundary."""

        materialized = tuple(
            item for item in items if self.requires_storage(item[1])
        )
        if not materialized:
            return
        torch.ops.shadowspill._acquire_storages(
            [tensor for tensor, _, _ in materialized],
            [binding.pointer for _, _, binding in materialized],
        )

    def replace_many(
        self,
        tensors: Sequence[torch.Tensor],
        alias_id: str,
        *,
        previous_generation: int,
        target_tensor: torch.Tensor,
        target_generation: int,
    ) -> None:
        """Rebind all frontend views after a no-copy lease replacement."""

        if not tensors or not self.requires_storage(alias_id):
            return
        torch.ops.shadowspill._replace_storages(
            list(tensors),
            _dense_id(alias_id, "alias_"),
            previous_generation,
            target_tensor.untyped_storage().data_ptr(),
            target_generation,
        )

    def before_execution_and_acquire(
        self,
        execution_handle: int,
        task_id: int,
        device_ordinal: int,
        tensors: Sequence[torch.Tensor],
        alias_ids: Sequence[str],
    ) -> tuple[int, ...]:
        """Run the admitted neutral boundary and install acquired storages."""

        if len(tensors) != len(alias_ids):
            raise RuntimeExecutionError(
                "task tensors and input aliases have different lengths"
            )
        materialized = tuple(
            (index, tensor, alias_id)
            for index, (tensor, alias_id) in enumerate(
                zip(tensors, alias_ids, strict=True)
            )
            if self.requires_storage(alias_id)
        )
        generations = torch.ops.shadowspill._before_execution_storages(
            [tensor for _, tensor, _ in materialized],
            execution_handle,
            task_id,
            device_ordinal,
        )
        if len(generations) != len(materialized):
            raise RuntimeExecutionError(
                "task acquisition returned the wrong generation count"
            )
        result = [0] * len(tensors)
        for (index, _tensor, _alias_id), generation in zip(
            materialized, generations, strict=True
        ):
            result[index] = int(generation)
        for index, alias_id in enumerate(alias_ids):
            if not self.requires_storage(alias_id):
                result[index] = self._zero_generations.setdefault(alias_id, 0)
        return tuple(result)

    def after_execution_and_update(
        self,
        execution_handle: int,
        task_id: int,
        device_ordinal: int,
        adopted: Sequence[tuple[torch.Tensor, str]],
        dematerialized: Sequence[torch.Tensor],
        *,
        replacement_aliases: frozenset[str] = frozenset(),
    ) -> tuple[int, ...]:
        """Publish storages and one admitted completion/action batch."""

        materialized = tuple(
            (index, tensor, alias_id)
            for index, (tensor, alias_id) in enumerate(adopted)
            if self.requires_storage(alias_id)
        )
        generations = torch.ops.shadowspill._after_execution_storages(
            [tensor for _, tensor, _ in materialized],
            [_dense_id(alias_id, "alias_") for _, _, alias_id in materialized],
            [self._size(alias_id) for _, _, alias_id in materialized],
            [
                2
                if alias_id in replacement_aliases
                else int(alias_id in self._registered)
                for _, _, alias_id in materialized
            ],
            list(dematerialized),
            execution_handle,
            task_id,
            device_ordinal,
        )
        if len(generations) != len(materialized):
            raise RuntimeExecutionError(
                "task publication returned the wrong generation count"
            )
        self._registered.update(alias_id for _, alias_id in adopted)
        result = [0] * len(adopted)
        for (index, _tensor, _alias_id), generation in zip(
            materialized, generations, strict=True
        ):
            result[index] = int(generation)
        for index, (_tensor, alias_id) in enumerate(adopted):
            if self.requires_storage(alias_id):
                continue
            if alias_id in replacement_aliases:
                self._zero_generations[alias_id] = (
                    self._zero_generations.get(alias_id, 0) + 1
                )
            result[index] = self._zero_generations.setdefault(alias_id, 0)
        return tuple(result)

    def dematerialize(
        self, tensor: torch.Tensor, alias_id: str, generation: int
    ) -> None:
        if tensor.untyped_storage().data_ptr() == 0:
            return
        torch.ops.shadowspill._rebind_storage(
            tensor, 0, _dense_id(alias_id, "alias_"), generation
        )

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
        self._require(
            self.library.shadowspill_pytorch_allocator_wait_idle(), "wait idle"
        )

    def input_failure_states(self, alias_ids: Iterable[str]) -> tuple[str, ...]:
        """Describe unavailable inputs without changing runtime state."""

        result: list[str] = []
        for alias_id in dict.fromkeys(alias_ids):
            snapshot = ObjectSnapshot()
            status = int(
                self.library.shadowspill_pytorch_object_snapshot(
                    _dense_id(alias_id, "alias_"), ctypes.byref(snapshot)
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

    def abort_task(self) -> None:
        self.library.shadowspill_pytorch_abort_task_range()

    def abort_task_after_failure(self, operation: str, cause: BaseException) -> None:
        """Close a failed task boundary and expose a latched allocator cause."""

        self.abort_task()
        try:
            self.raise_if_allocator_failed(operation)
        except RuntimeExecutionError as error:
            raise error from cause

    def raise_if_allocator_failed(self, operation: str) -> None:
        """Raise the first callback failure without touching the device timeline."""

        failure = AdapterFailure()
        status = int(
            self.library.shadowspill_pytorch_allocator_failure(ctypes.byref(failure))
        )
        if status == 0:
            return
        raise RuntimeExecutionError(self._failure_message(operation, status, failure))

    def registered_aliases(self) -> frozenset[str]:
        return frozenset(self._registered)

    def adopt_registered(self, alias_ids: Iterable[str]) -> None:
        """Adopt objects registered by a compatible provisional bridge."""

        for alias_id in alias_ids:
            self._size(alias_id)
            self._registered.add(alias_id)
            if not self.requires_storage(alias_id):
                self._zero_generations.setdefault(alias_id, 0)

    def requires_storage(self, alias_id: str) -> bool:
        """Return whether an alias bundle owns any physical payload bytes."""

        return self._size(alias_id) != 0

    def _zero_binding(self, alias_id: str) -> ObjectBinding:
        if self.requires_storage(alias_id):
            raise AssertionError("zero binding requested for a materialized alias")
        return ObjectBinding(
            _dense_id(alias_id, "alias_"),
            self._zero_generations.setdefault(alias_id, 0),
            0,
            0,
            None,
        )

    def _expand_bindings(
        self,
        aliases: Sequence[str],
        materialized_aliases: Sequence[str],
        materialized_bindings: ctypes.Array[ObjectBinding]
        | Sequence[ObjectBinding],
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
        failure = AdapterFailure()
        self.library.shadowspill_pytorch_allocator_failure(ctypes.byref(failure))
        raise RuntimeExecutionError(self._failure_message(operation, status, failure))

    @staticmethod
    def _failure_message(operation: str, status: int, failure: AdapterFailure) -> str:
        requested = max(
            int(failure.requested_bytes), int(failure.runtime.requested_bytes)
        )
        return (
            f"{operation} failed with status {status}; "
            f"device={failure.device_ordinal}, "
            f"object={failure.runtime.object_id}, "
            f"allocation={failure.runtime.allocation_id}, "
            f"requested={requested}, "
            f"free={failure.runtime.free_bytes}, "
            f"largest_free_range={failure.runtime.largest_free_range_bytes}"
        )


def actions_by_task(
    actions: Iterable[MemoryAction],
) -> Mapping[str, tuple[MemoryAction, ...]]:
    """Preserve schedule ordering while grouping actions by trigger task."""

    result: dict[str, list[MemoryAction]] = {}
    for action in actions:
        result.setdefault(action.trigger_task_id, []).append(action)
    return {task_id: tuple(values) for task_id, values in result.items()}


__all__ = ["RuntimeBridge", "RuntimeExecutionError", "actions_by_task"]
