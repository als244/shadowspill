"""Qualification helpers for comparing the observer with the execution allocator."""

import ctypes
import itertools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.runtime.abi import PROFILING_SCOPE_BASE, Allocation
from shadowspill.runtime.failures import wait_allocator_idle
from shadowspill.runtime.telemetry import AllocationTelemetryError
from shadowspill.runtime.timing import wait_for_stream
from shadowspill.task.profiles import TaskOutputInputBinding

_scope_ids = itertools.count(PROFILING_SCOPE_BASE)


class AllocatorBoundary:
    def __init__(self, library: Any, *, runtime_handle: int, plan_id: int):
        self.library = library
        self.runtime_handle = runtime_handle
        self.plan_id = plan_id

    @contextmanager
    def scope(self, stream: torch.cuda.Stream) -> Iterator[int]:
        scope_id = next(_scope_ids)
        status = self.library.shadowspill_pytorch_allocation_scope_begin(
            self.plan_id, scope_id
        )
        if status:
            raise CaptureError(f"allocation scope begin failed: {status}")
        try:
            yield scope_id
        except BaseException:
            self.library.shadowspill_pytorch_allocation_scope_abort()
            raise
        status = self.library.shadowspill_pytorch_allocation_scope_end(
            scope_id, stream.cuda_stream
        )
        if status:
            raise CaptureError(f"allocation scope end failed: {status}")

    def invoke(self, task: Callable[[], object], stream: torch.cuda.Stream) -> None:
        with self.scope(stream):
            output = task()
            del output
        self.drain(stream, problem="allocator warmup")

    def drain(self, stream: torch.cuda.Stream, *, problem: str) -> None:
        wait_for_stream(self.runtime_handle, int(stream.cuda_stream))
        error = wait_allocator_idle(self.library, self.runtime_handle, problem=problem)
        if error is not None:
            raise AllocationTelemetryError(error)

    def allocation_for_pointer(self, address: int) -> Allocation:
        allocation = Allocation()
        status = self.library.shadowspill_pytorch_allocation_for_pointer(
            address, ctypes.byref(allocation)
        )
        if status:
            raise CaptureError(
                "compiled task returned storage outside the ShadowSpill slab"
            )
        return allocation


def output_allocation_views(
    boundary: AllocatorBoundary,
    output: object,
    *,
    inputs: Sequence[object] = (),
) -> tuple[
    dict[int, tuple[tuple[int, int], ...]],
    tuple[TaskOutputInputBinding, ...],
]:
    """Map each output leaf to its slab allocation, or to the input it aliases."""

    views_by_allocation: dict[int, list[tuple[int, int]]] = {}
    input_by_allocation: dict[int, int] = {}
    for input_position, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor) or not value.is_cuda:
            continue
        address = value.untyped_storage().data_ptr()
        if address == 0:
            continue
        allocation = boundary.allocation_for_pointer(address)
        input_by_allocation.setdefault(int(allocation.allocation_id), input_position)
    input_bindings: list[TaskOutputInputBinding] = []
    leaves, _ = tree_flatten(output)
    for leaf_index, leaf in enumerate(leaves):
        if not isinstance(leaf, torch.Tensor) or not leaf.is_cuda:
            continue
        address = leaf.untyped_storage().data_ptr()
        if address == 0:
            continue
        allocation = boundary.allocation_for_pointer(address)
        allocation_pointer = int(allocation.pointer or 0)
        # Tensor.data_ptr() is zero for an empty view even when it retains
        # a nonempty storage. Its storage address/offset still describe the
        # allocation whose lifetime this returned view keeps alive.
        offset_bytes = (
            address
            + int(leaf.storage_offset()) * leaf.element_size()
            - allocation_pointer
        )
        if offset_bytes < 0 or offset_bytes > int(allocation.requested_bytes):
            raise CaptureError("compiled output view lies outside its allocator record")
        donated_input_position = input_by_allocation.get(int(allocation.allocation_id))
        if donated_input_position is not None:
            input_bindings.append(
                TaskOutputInputBinding(leaf_index, donated_input_position, offset_bytes)
            )
        else:
            views_by_allocation.setdefault(allocation.allocation_id, []).append(
                (leaf_index, offset_bytes)
            )
    return (
        {
            allocation_id: tuple(views)
            for allocation_id, views in views_by_allocation.items()
        },
        tuple(input_bindings),
    )
