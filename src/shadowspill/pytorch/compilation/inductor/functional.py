"""Functional intermediates with the task's existing input-mutation ABI."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._prims_common import compute_required_storage_length
from torch._subclasses.functional_tensor import (
    FunctionalTensor,
    FunctionalTensorMode,
    dispatch_functionalize,
)
from torch.fx import GraphModule

type _View = tuple[int, torch.dtype, tuple[int, ...], tuple[int, ...], int]
type _Base = tuple[int, int, tuple[_View, ...]]


def _shared_bases(inputs: Sequence[object]) -> tuple[_Base, ...]:
    tensors = {
        i: value for i, value in enumerate(inputs) if isinstance(value, torch.Tensor)
    }
    groups: dict[int, list[int]] = defaultdict(list)
    for index, value in tensors.items():
        groups[value.untyped_storage()._cdata].append(index)
    bases = []
    for positions in groups.values():
        if len(positions) < 2:
            continue
        owner = min(positions, key=lambda index: tensors[index].element_size())
        extent = max(
            compute_required_storage_length(
                value.shape, value.stride(), int(value.storage_offset())
            )
            * value.element_size()
            for index in positions
            for value in (tensors[index],)
        )
        views = tuple(
            (
                index,
                value.dtype,
                tuple(value.shape),
                tuple(value.stride()),
                int(value.storage_offset()),
            )
            for index in positions
            for value in (tensors[index],)
        )
        bases.append((owner, extent // tensors[owner].element_size(), views))
    return tuple(bases)


def functional_task(
    graph: GraphModule, inputs: Sequence[object]
) -> Callable[..., object]:
    """Normalize mutations before Inductor's functional graph optimizations.

    Shared bases preserve overlapping input views. Base extents cover only the
    declared input views, not any unused capacity of their original allocations.
    The Python dispatcher supports opaque mutable operators. Terminal copies
    publish input updates without another autograd dispatcher or saved tensors.
    """
    bases = _shared_bases(inputs)

    def capture(*arguments: Any) -> Any:
        logical = list(arguments)
        for owner, _, views in bases:
            base = arguments[owner]
            for index, dtype, shape, stride, offset in views:
                typed = base if base.dtype == dtype else base.view(dtype)
                logical[index] = typed.as_strided(shape, stride, offset)
        output = graph(*logical)
        changed = tuple(
            isinstance(value, FunctionalTensor)
            and torch._functionalize_has_data_mutation(value.elem)  # type: ignore[attr-defined]
            for value in arguments
        )
        return output, arguments, changed

    functional = dispatch_functionalize(capture, FunctionalTensorMode())

    def execute(*arguments: Any) -> Any:
        physical = list(arguments)
        for owner, elements, _ in bases:
            physical[owner] = arguments[owner].as_strided((elements,), (1,), 0)
        output, updated, changed = functional(*physical)
        for original, value, mutated in zip(physical, updated, changed, strict=True):
            if mutated:
                original.copy_(value)
        return output

    return execute
