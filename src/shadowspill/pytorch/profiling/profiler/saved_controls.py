"""Backward saved controls taken from the paired forward task that produces them."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import (
    AotGraphPair,
    TaskInputProvenance,
    TaskInputRole,
)

if TYPE_CHECKING:
    from . import TaskProfiler


def resolve_graph_pair_controls(
    profiler: TaskProfiler,
    pair: AotGraphPair,
    metadata_digest: str | None,
    produced: dict[tuple[str, str | None, int], tuple[torch.Tensor | None, ...]],
) -> AotGraphPair:
    """Populate a backward's unbound saved controls from its forward task.

    Only non-floating saved leaves are snapshotted: a control decides what
    the backward does, so its authentic value matters, while a continuous
    saved value only has to have the right shape.  ``produced`` memoizes one
    forward run per (forward contract, metadata, saved arity).
    """

    provenance = pair.backward.input_provenance
    missing = tuple(
        position
        for position, item in enumerate(provenance[: pair.saved_value_count])
        if item.role is TaskInputRole.CONTROL and item.representative_value is None
    )
    if not missing:
        return pair
    key = (pair.forward.compatibility_digest, metadata_digest, pair.saved_value_count)
    values = produced.get(key)
    if values is None:
        values = _run_producer(profiler, pair)
        produced[key] = values
    rebound = tuple(
        _bind(item, values[position]) if position in missing else item
        for position, item in enumerate(provenance)
    )
    backward = pair.backward.rebind_examples(
        pair.backward.example_arguments, input_provenance=rebound
    )
    return replace(pair, backward=backward)


def _run_producer(
    profiler: TaskProfiler,
    pair: AotGraphPair,
) -> tuple[torch.Tensor | None, ...]:
    """Run one forward task and snapshot only its non-floating saved leaves."""

    executables = profiler.executables
    executable = executables.get(pair.forward)
    if not executable.example_arguments:
        executable = executables.with_arguments(executable)
    boundary = profiler.boundary
    stream = boundary.stream()
    try:
        with boundary.scope(stream):
            output = executable()
            leaves, _ = tree_flatten(output)
            original_count = pair.forward.output_count - pair.saved_value_count
            saved = leaves[original_count:]
            if len(saved) != pair.saved_value_count:
                raise CaptureError("paired forward changed its saved-value arity")
            roles = pair.backward.input_provenance
            values = tuple(
                _snapshot(value)
                if roles[position].role is TaskInputRole.CONTROL
                else None
                for position, value in enumerate(saved)
            )
            del output
        boundary.drain(stream, problem="saved-control producer")
        return values
    finally:
        executables.release_occurrence_values(executable)


def _bind(
    provenance: TaskInputProvenance,
    value: torch.Tensor | None,
) -> TaskInputProvenance:
    if value is None:
        raise CaptureError(
            "paired forward did not produce an authentic saved control: "
            f"source={provenance.source}"
        )
    return replace(provenance, representative_value=value)


def _snapshot(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CaptureError("paired forward saved control is not a tensor")
    if value.is_floating_point() or value.is_complex():
        raise CaptureError("paired forward saved control has a continuous dtype")
    source = value.detach().to(device="cpu")
    result = torch.empty_strided(
        tuple(source.shape),
        tuple(source.stride()),
        dtype=source.dtype,
        device="cpu",
    )
    result.copy_(source)
    return result
