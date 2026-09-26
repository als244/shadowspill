"""Backward saved values taken from the forward task that produces them."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import AotGraphPair
from shadowspill.pytorch.profiling.geometry import distinct_locations

if TYPE_CHECKING:
    from . import TaskProfiler


def resolve_graph_pair_saved_values(
    profiler: TaskProfiler,
    pair: AotGraphPair,
    metadata_digest: str | None,
    produced: dict[tuple[str, str | None, int], tuple[tuple[torch.Tensor, str], ...]],
) -> AotGraphPair:
    """Populate a backward's saved inputs from the forward that produces them.

    A backward is not a task that can be measured on its own. Its saved
    inputs are what its forward kept: the activations it will need again,
    the statistics a fused operator needs to rebuild its result, the random
    state it drew from. They mean something only together, and inventing
    them one at a time asks a kernel to undo a forward pass that never
    happened -- which is how a value chosen for its shape alone reaches a
    kernel entitled to assume it came from somewhere.

    So a pair is measured as a pair. The forward runs on its own
    representative inputs and the backward runs on what came out of it, with
    only its tangents invented. Nothing here decides which saved values may
    be invented, because none of them may.  ``produced`` memoizes one
    forward run per (forward contract, metadata, saved arity).
    """

    provenance = pair.backward.input_provenance
    missing = tuple(
        position
        for position, item in enumerate(provenance[: pair.saved_value_count])
        if item.representative_value is None
    )
    if not missing:
        return pair
    key = (pair.forward.compatibility_digest, metadata_digest, pair.saved_value_count)
    values = produced.get(key)
    if values is None:
        values = _run_producer(profiler, pair)
        produced[key] = values
    rebound = tuple(
        replace(
            item,
            representative_value=values[position][0],
            produced_device_type=values[position][1],
        )
        if position in missing
        else item
        for position, item in enumerate(provenance)
    )
    backward = pair.backward.rebind_examples(
        pair.backward.example_arguments, input_provenance=rebound
    )
    return replace(pair, backward=backward)


def _run_producer(
    profiler: TaskProfiler,
    pair: AotGraphPair,
) -> tuple[tuple[torch.Tensor, str], ...]:
    """Run one forward task and snapshot every value it saves, and where."""

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
            values = _snapshot(profiler, saved)
            del output
        boundary.drain(stream, problem="saved-value producer")
        return values
    finally:
        executables.release_occurrence_values(executable)


def _snapshot(
    profiler: TaskProfiler, saved: list[object]
) -> tuple[tuple[torch.Tensor, str], ...]:
    """What one forward saved, on the host, in the geometry the forward gave it.

    Moving a tensor to the host is free to choose its own layout, and for a
    broadcast it chooses a dense one -- which is a different tensor from the
    one the backward's contract describes. The host copy is built to the
    produced geometry and filled through the locations that geometry
    actually has.

    Whatever the forward produced is what the backward is owed, infinities
    included: an attention mask is a plane of zeros and negative infinities,
    and a backward given a finite stand-in for one is given a mask that
    masks nothing.

    The copies are given to the profiler to keep before anything is written
    to them, and it has them filled in the spill pool.
    """

    sources = []
    for value in saved:
        if not isinstance(value, torch.Tensor):
            raise CaptureError("paired forward saved value is not a tensor")
        sources.append(value.detach())
    copies = tuple(
        torch.empty_strided(
            tuple(source.shape),
            tuple(source.stride()),
            dtype=source.dtype,
            device="cpu",
        )
        for source in sources
    )

    def fill() -> None:
        for copy, source in zip(copies, sources, strict=True):
            distinct_locations(copy).copy_(distinct_locations(source))

    profiler.keep_saved_values(copies, fill)
    return tuple(
        (copy, source.device.type) for copy, source in zip(copies, sources, strict=True)
    )
