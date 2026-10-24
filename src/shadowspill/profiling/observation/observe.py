"""Observe one invocation while preserving its actual output/input aliasing."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.task.profiles import TaskOutputInputBinding

from .decode import AllocationObservation, decode_allocations
from .native import observe_allocations


def observe_task(
    function: Callable[..., Any],
    arguments: Sequence[Any],
    *,
    device: torch.device,
    capacity: int = 131072,
) -> AllocationObservation:
    assert device.index is not None
    torch.cuda.synchronize(device)
    with observe_allocations(device.index, capacity=capacity) as events:
        output = function(*arguments)
        views, bindings, off_device = _output_views(arguments, output)
        del output
        torch.cuda.synchronize(device)
    return decode_allocations(events, views, bindings, off_device)


def _output_views(
    inputs: Sequence[Any],
    outputs: Any,
) -> tuple[dict[int, list[tuple[int, int]]], list[TaskOutputInputBinding], list[int]]:
    input_pointers: dict[int, int] = {}
    for position, value in enumerate(inputs):
        if isinstance(value, torch.Tensor) and value.is_cuda:
            input_pointers.setdefault(value.untyped_storage().data_ptr(), position)
    views: dict[int, list[tuple[int, int]]] = {}
    bindings, off_device = [], []
    leaves, _ = tree_flatten(outputs)
    for index, value in enumerate(leaves):
        if not isinstance(value, torch.Tensor):
            continue
        if not value.is_cuda:
            off_device.append(index)
            continue
        pointer = value.untyped_storage().data_ptr()
        if pointer == 0:
            continue
        offset = int(value.storage_offset()) * value.element_size()
        if pointer in input_pointers:
            bindings.append(
                TaskOutputInputBinding(index, input_pointers[pointer], offset)
            )
        else:
            views.setdefault(pointer, []).append((index, offset))
    return views, bindings, off_device
