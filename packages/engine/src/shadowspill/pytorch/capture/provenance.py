"""Track logical inputs and saved forward values through AOT differentiation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    GraphArtifact,
    TaskInputProvenance,
)
from shadowspill.pytorch.capture.storage import StorageRootKind
from shadowspill.pytorch.representations import tensor_components
from shadowspill.task.inputs import TaskInputRole


def physical_input_provenance(
    inputs: Sequence[object], provenance: tuple[TaskInputProvenance, ...] | None
) -> tuple[TaskInputProvenance, ...] | None:
    """Expand logical input provenance in AOT's physical-component order."""
    if provenance is None:
        return None
    expanded = []
    for value, item in zip(inputs, provenance, strict=True):
        if not isinstance(value, torch.Tensor):
            continue
        references = (
            dict(tensor_components(item.representative_value))
            if item.representative_value is not None
            else {}
        )
        for path, _component in tensor_components(value):
            expanded.append(replace(item, representative_value=references.get(path)))
    return tuple(expanded)


def _tensor_input_provenance(
    inputs: Sequence[object],
    provenance: tuple[TaskInputProvenance, ...] | None,
) -> tuple[TaskInputProvenance, ...] | None:
    if provenance is None:
        return None
    return tuple(
        item
        for value, item in zip(inputs, provenance, strict=True)
        if isinstance(value, torch.Tensor)
    )


def _backward_input_provenance(
    forward: GraphArtifact,
    *,
    original_output_count: int,
    saved_value_count: int,
    backward_argument_count: int,
) -> tuple[TaskInputProvenance, ...]:
    """Project saved forward values and terminal tangents onto backward inputs."""

    if saved_value_count > backward_argument_count:
        raise CaptureError("AOT backward has fewer arguments than saved values")
    views = {item.leaf_index: item for item in forward.storage_contract.output_views}
    result: list[TaskInputProvenance] = []
    for offset in range(saved_value_count):
        leaf_index = original_output_count + offset
        view = views.get(leaf_index)
        if view is None:
            result.append(
                TaskInputProvenance(
                    TaskInputRole.RESIDUAL,
                    f"forward_output_{leaf_index}",
                )
            )
            continue
        root = forward.storage_contract.roots[view.root_id]
        if root.kind is StorageRootKind.INPUT:
            assert root.source_input is not None
            try:
                source = forward.input_provenance[root.source_input]
            except IndexError as exc:
                raise CaptureError(
                    "saved forward input root is outside the task contract"
                ) from exc
            result.append(_saved_input_view_provenance(source, view))
        else:
            result.append(
                TaskInputProvenance(
                    (
                        TaskInputRole.RESIDUAL
                        if _is_continuous_dtype_name(view.dtype)
                        else TaskInputRole.CONTROL
                    ),
                    f"forward_output_{leaf_index}",
                )
            )
    result.extend(
        TaskInputProvenance(TaskInputRole.TANGENT, f"tangent_{index}")
        for index in range(backward_argument_count - saved_value_count)
    )
    return tuple(result)


def _is_continuous_dtype_name(dtype_name: str) -> bool:
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise CaptureError(f"AOT saved value has unknown dtype {dtype_name!r}")
    return bool(dtype.is_floating_point or dtype.is_complex)


def rebind_backward_input_provenance(
    pair: AotGraphPair,
    forward: GraphArtifact,
) -> tuple[TaskInputProvenance, ...]:
    """Rebuild saved-value provenance for one occurrence-local forward contract."""

    original_output_count = forward.output_count - pair.saved_value_count
    if original_output_count < 0:
        raise CaptureError("AOT graph pair has an invalid saved-value count")
    # A backward that accumulates takes the running gradients as further
    # arguments. They come from the plan rather than from the forward, so
    # projecting the forward cannot rebuild them; carry them across instead.
    priors = tuple(
        item
        for item in pair.backward.input_provenance
        if item.role is TaskInputRole.GRADIENT
    )
    if priors and pair.backward.input_provenance[-len(priors) :] != priors:
        raise CaptureError(
            "accumulated gradient arguments are not the backward's last ones"
        )
    return (
        *_backward_input_provenance(
            forward,
            original_output_count=original_output_count,
            saved_value_count=pair.saved_value_count,
            backward_argument_count=pair.backward.argument_count - len(priors),
        ),
        *priors,
    )


def _saved_input_view_provenance(
    source: TaskInputProvenance,
    view: Any,
) -> TaskInputProvenance:
    """Preserve authentic values through an AOT-saved input view."""

    reference = source.representative_value
    if reference is None:
        return source
    itemsize = reference.element_size()
    if view.dtype != str(reference.dtype) or view.offset_bytes % itemsize:
        raise CaptureError(
            "saved input view is incompatible with its representative storage: "
            f"source={source.source}, dtype={view.dtype}, "
            f"offset_bytes={view.offset_bytes}"
        )
    storage_bytes = reference.untyped_storage().nbytes()
    if view.offset_bytes + view.span_bytes > storage_bytes:
        raise CaptureError(
            "saved input view exceeds its representative storage: "
            f"source={source.source}, required={view.offset_bytes + view.span_bytes}, "
            f"available={storage_bytes}"
        )
    # This helper runs while the enclosing AOT capture owns a FakeTensorMode.
    # The occurrence-local reference is intentionally a real CPU tensor, so
    # construct its metadata-only view below Python dispatch.
    with torch._C._DisableTorchDispatch():
        result = reference.as_strided(
            view.shape,
            view.stride,
            view.offset_bytes // itemsize,
        )
    return TaskInputProvenance(
        source.role,
        source.source,
        source.consumer_targets,
        result,
    )
