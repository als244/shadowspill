"""Narrow high-level bridge over the private PyTorch adapter ABI."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from shadowspill.ir import MemoryAction, MemoryActionKind, MutationSpec, Program

from ._abi import AdapterFailure, ObjectBinding, ObjectUpdate, RuntimeAction
from .contracts import PlanningError


class RuntimeExecutionError(RuntimeError):
    """A native runtime transition rejected the immutable execution plan."""


_ACTION_KIND = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.OFFLOAD: 1,
    MemoryActionKind.PREFETCH: 2,
}


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
        self._alias_by_object = {
            item.object_id: item.alias_group_id for item in program.objects
        }
        self._size_by_alias = {
            item.alias_group_id: item.size_bytes for item in program.alias_groups
        }
        self._registered: set[str] = set()

    def alias_for_object(self, object_id: str) -> str:
        try:
            return self._alias_by_object[object_id]
        except KeyError as exc:
            raise PlanningError(f"unknown program object {object_id!r}") from exc

    def register_host_tensor(
        self, alias_id: str, tensor: torch.Tensor, *, retain_host_backing: bool
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
        self._require(
            self.library.shadowspill_pytorch_register_host_object(
                _dense_id(alias_id, "alias_"),
                expected,
                int(retain_host_backing),
                storage.data_ptr(),
            ),
            "register host object",
        )
        self._registered.add(alias_id)

    def write_host_tensor(self, alias_id: str, tensor: torch.Tensor) -> None:
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
        self._require(
            self.library.shadowspill_pytorch_write_host_object(
                _dense_id(alias_id, "alias_"), expected, storage.data_ptr()
            ),
            "write host object",
        )

    def read_host_tensor(self, alias_id: str, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu":
            raise RuntimeExecutionError("writeback destination must be CPU resident")
        storage = tensor.untyped_storage()
        expected = self._size(alias_id)
        if storage.nbytes() != expected:
            raise RuntimeExecutionError(
                f"writeback storage for {alias_id!r} has {storage.nbytes()} bytes; "
                f"the plan requires {expected}"
            )
        self._require(
            self.library.shadowspill_pytorch_read_host_object(
                _dense_id(alias_id, "alias_"), expected, storage.data_ptr()
            ),
            "read host object",
        )

    def unregister(self, alias_ids: Iterable[str]) -> None:
        for alias_id in dict.fromkeys(alias_ids):
            if alias_id not in self._registered:
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

    def before_task(
        self,
        task_id: str,
        stream: torch.cuda.Stream,
        input_alias_ids: tuple[str, ...],
    ) -> tuple[ObjectBinding, ...]:
        identifiers = (ctypes.c_uint64 * len(input_alias_ids))(
            *(_dense_id(value, "alias_") for value in input_alias_ids)
        )
        bindings = (ObjectBinding * len(input_alias_ids))()
        self._require(
            self.library.shadowspill_pytorch_before_task(
                _dense_id(task_id, "task_"),
                stream.cuda_stream,
                identifiers if input_alias_ids else None,
                len(input_alias_ids),
                bindings if input_alias_ids else None,
                len(input_alias_ids),
            ),
            f"before task {task_id}",
        )
        return tuple(bindings)

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
        stream = torch.cuda.current_stream()
        identifiers = (ctypes.c_uint64 * len(alias_ids))(
            *(_dense_id(value, "alias_") for value in alias_ids)
        )
        bindings = (ObjectBinding * len(alias_ids))()
        task_open = False
        try:
            self._require(
                self.library.shadowspill_pytorch_before_task(
                    task_number,
                    stream.cuda_stream,
                    identifiers if alias_ids else None,
                    len(alias_ids),
                    bindings if alias_ids else None,
                    len(alias_ids),
                ),
                "acquire caller outputs",
            )
            task_open = True
            for alias_id, tensor, binding in zip(
                alias_ids, tensors, bindings, strict=True
            ):
                self.rebind(tensor, alias_id, binding)
            self._require(
                self.library.shadowspill_pytorch_after_task(
                    task_number, stream.cuda_stream, None, 0, None, 0
                ),
                "close caller output acquisition",
            )
            task_open = False
            return tuple(bindings)
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
        mutation_values = tuple(mutations)
        action_values = tuple(actions)
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
        if not actions:
            return
        stream = torch.cuda.current_stream()
        runtime_actions = (RuntimeAction * len(actions))(
            *(
                RuntimeAction(
                    _dense_id(item.alias_group_id, "alias_"),
                    _ACTION_KIND[item.kind],
                )
                for item in actions
            )
        )
        self._require(
            self.library.shadowspill_pytorch_after_task(
                task_number,
                stream.cuda_stream,
                None,
                0,
                runtime_actions,
                len(actions),
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
        torch.ops.shadowspill._rebind_storage(
            tensor,
            binding.pointer,
            _dense_id(alias_id, "alias_"),
            binding.generation,
        )

    def dematerialize(
        self, tensor: torch.Tensor, alias_id: str, generation: int
    ) -> None:
        if tensor.untyped_storage().data_ptr() == 0:
            return
        torch.ops.shadowspill._rebind_storage(
            tensor, 0, _dense_id(alias_id, "alias_"), generation
        )

    def wait_idle(self) -> None:
        self._require(
            self.library.shadowspill_pytorch_allocator_wait_idle(), "wait idle"
        )

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
