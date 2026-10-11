"""Authentic control-value propagation across partition boundaries.

Integer and boolean task inputs can select kernels, allocation paths, and
kernel caches. They therefore cannot use geometry-only synthetic values.
This module evaluates the producer dependency slices needed to construct such
a value from caller-supplied roots, and nothing else: a stage the value does
not depend on is never executed to manufacture a profiling input.

What a value depends on is not always another control value.  A model that
selects which tokens to attend to computes that selection from the
activations of the layer that produces it, so the slice that computes an
integer needs a float the stage before it produced. Discover those dependencies
backwards, then evaluate them in stage order. Only the current slice's inputs
are restored to their captured devices. Host snapshots are released at last use,
except for the control values returned to profiling. This also supports long
stage chains without depending on Python's recursion limit.

**Each slice input uses its captured device.** A root may be a pool-backed
tensor and an earlier stage's value may be a CPU snapshot. Restore each to
the device recorded for that input: CUDA payloads and explicit CPU control
inputs can coexist in the same producer slice.

It has to be that device and not the planning host, because a captured graph
is specialized to the layouts of the device it was captured on. Export
records `aten.view` wherever a reshape was legal as a view, and whether it
was legal depends on strides that the two implementations of an operator do
not agree about: attention returns its result laid out by sequence on an
accelerator and by head on a host, so the merge-heads idiom after it -- a
transpose and a view -- is free on one and impossible on the other. Replaying
such a graph anywhere but where it was traced is unsound, and only ever
worked while slices were short enough not to reach such an operator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx import Graph, GraphModule, Node
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError

from .artifacts import StageRecord
from .split import SplitExportGraph

StageOutputKey = tuple[int, int]


def derive_authentic_control_values(
    split: SplitExportGraph,
    representative_root_inputs: tuple[object, ...] | None,
    *,
    caller_values: Mapping[StageOutputKey, torch.Tensor] | None = None,
) -> dict[StageOutputKey, torch.Tensor]:
    """Resolve every produced integer/boolean value consumed by another stage.

    Caller values are an explicit escape hatch for a producer slice that
    cannot execute on the planning host.  They obey the same geometry checks
    as values derived from the graph.
    """

    required = _required_control_outputs(split)
    if not required:
        return {}
    resolved = _validate_caller_values(split, required, caller_values or {})
    if required.issubset(resolved):
        return resolved
    if representative_root_inputs is None:
        raise CaptureError(
            "partitioned integer/boolean inputs require authentic "
            "producer-derived values or explicit caller-supplied stage values"
        )
    slices, last_use = _plan_control_slices(split, required - resolved.keys(), resolved)
    release: dict[int, list[StageOutputKey]] = {}
    for key, index in last_use.items():
        if key not in required:
            release.setdefault(index, []).append(key)
    for stage_index in sorted(slices):
        resolved.update(
            _evaluate_control_slice(
                split.stages[stage_index],
                stage_index=stage_index,
                selected=slices[stage_index],
                required=required,
                representative_root_inputs=representative_root_inputs,
                resolved=resolved,
            )
        )
        for key in release.get(stage_index, ()):
            del resolved[key]
    missing = sorted(required - resolved.keys())
    if missing:
        raise AssertionError(f"control-value propagation is incomplete: {missing}")
    return {key: value for key, value in resolved.items() if key in required}


@dataclass(frozen=True)
class _ControlSlice:
    graph: GraphModule
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


def _plan_control_slices(
    split: SplitExportGraph,
    required: set[StageOutputKey],
    resolved: Mapping[StageOutputKey, torch.Tensor],
) -> tuple[dict[int, _ControlSlice], dict[StageOutputKey, int]]:
    """Find the needed slices without restoring any tensor to a device.

    StageRecord order is topological. A backwards pass therefore discovers
    every requested output before visiting its producer, even across branches.
    """
    needed: dict[int, set[int]] = {}
    for stage, output_index in required:
        if not 0 <= stage < len(split.stages):
            raise CaptureError(f"control-value producer names missing stage {stage}")
        needed.setdefault(stage, set()).add(output_index)
    slices: dict[int, _ControlSlice] = {}
    last_use: dict[StageOutputKey, int] = {}
    for index in reversed(range(len(split.stages))):
        if index not in needed:
            continue
        record = split.stages[index]
        outputs = tuple(sorted(needed[index]))
        graph, positions = _slice_outputs(record.graph_module, outputs)
        slices[index] = _ControlSlice(graph, positions, outputs)
        for position in positions:
            source = record.input_sources[position]
            if source is None or source.producer_stage_index is None:
                continue
            producer = source.producer_stage_index
            output = source.producer_output_index
            assert output is not None
            if producer >= index:
                raise CaptureError(
                    f"stage_{index:04d} depends on nonpreceding stage {producer}; "
                    "control-value dependencies must be topologically ordered"
                )
            key = (producer, output)
            last_use[key] = max(last_use.get(key, index), index)
            if key not in resolved:
                needed.setdefault(producer, set()).add(output)
    return slices, last_use


def _required_control_outputs(split: SplitExportGraph) -> set[StageOutputKey]:
    result: set[StageOutputKey] = set()
    for record in split.stages:
        if len(record.inputs) != len(record.input_sources):
            raise CaptureError("stage inputs differ from their provenance")
        for value, source in zip(record.inputs, record.input_sources, strict=True):
            if not isinstance(value, torch.Tensor) or not _requires_authentic(value):
                continue
            if source is None or source.producer_stage_index is None:
                continue
            assert source.producer_output_index is not None
            result.add((source.producer_stage_index, source.producer_output_index))
    return result


def _validate_caller_values(
    split: SplitExportGraph,
    required: set[StageOutputKey],
    supplied: Mapping[StageOutputKey, torch.Tensor],
) -> dict[StageOutputKey, torch.Tensor]:
    result: dict[StageOutputKey, torch.Tensor] = {}
    unexpected = sorted(set(supplied) - required)
    if unexpected:
        raise CaptureError(
            f"caller supplied values for unused stage outputs: {unexpected}"
        )
    for key, value in supplied.items():
        if not isinstance(value, torch.Tensor):
            raise CaptureError(f"caller control value {key} is not a tensor")
        expected = _stage_output_tensor(
            split.stages[key[0]], key[1], key=key, integral=True
        )
        _validate_geometry(value, expected, key=key, origin="caller")
        result[key] = _snapshot(value)
    return result


def _evaluate_control_slice(
    record: StageRecord,
    *,
    stage_index: int,
    selected: _ControlSlice,
    required: set[StageOutputKey],
    representative_root_inputs: tuple[object, ...],
    resolved: dict[StageOutputKey, torch.Tensor],
) -> dict[StageOutputKey, torch.Tensor]:
    sliced, input_positions, output_indices = (
        selected.graph,
        selected.inputs,
        selected.outputs,
    )
    arguments = tuple(
        _resolve_slice_input(
            record,
            position,
            stage_index=stage_index,
            representative_root_inputs=representative_root_inputs,
            resolved=resolved,
        )
        for position in input_positions
    )
    try:
        with unset_fake_temporarily(), torch.no_grad():
            output = sliced(*arguments)
    except CaptureError:
        raise
    except BaseException as error:
        names = ", ".join(str(index) for index in output_indices)
        # Say where every value the slice was given lives, and where the
        # module it runs holds anything of its own. A device mismatch here
        # is the common failure and the cause alone does not say which side
        # of it came from where.
        places = ", ".join(
            f"{position}:{value.device}"
            if isinstance(value, torch.Tensor)
            else f"{position}:static"
            for position, value in zip(input_positions, arguments, strict=True)
        )
        held = ", ".join(
            f"{name}:{value.device}"
            for name, value in sliced.named_buffers()
            if isinstance(value, torch.Tensor)
        )
        raise CaptureError(
            "failed to derive authentic integer/boolean stage output values: "
            f"producer=stage_{stage_index:04d}, outputs=[{names}], "
            f"inputs=[{places}]"
            + (f", module holds [{held}]" if held else "")
            + f", cause={error}"
        ) from error

    leaves, _ = tree_flatten(output)
    if len(leaves) != len(output_indices):
        raise CaptureError("control-value producer slice changed its output arity")
    result: dict[StageOutputKey, torch.Tensor] = {}
    for output_index, value in zip(output_indices, leaves, strict=True):
        key = (stage_index, output_index)
        if not isinstance(value, torch.Tensor):
            raise CaptureError(f"derived control value {key} is not a tensor")
        expected = _stage_output_tensor(
            record, output_index, key=key, integral=key in required
        )
        _validate_geometry(value, expected, key=key, origin="producer")
        result[key] = _snapshot(value)
    return result


def _slice_outputs(
    module: GraphModule,
    output_indices: tuple[int, ...],
) -> tuple[GraphModule, tuple[int, ...]]:
    output_node = next(node for node in module.graph.nodes if node.op == "output")
    leaves, _ = tree_flatten(output_node.args[0])
    targets: list[Node] = []
    for output_index in output_indices:
        try:
            target = leaves[output_index]
        except IndexError as error:
            raise CaptureError(
                f"producer output index {output_index} is outside its stage contract"
            ) from error
        if not isinstance(target, Node):
            raise CaptureError(
                f"producer output {output_index} has no FX value dependency"
            )
        targets.append(target)

    required: set[Node] = set(targets)
    pending = list(targets)
    while pending:
        node = pending.pop()
        for dependency in node.all_input_nodes:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)

    graph = Graph()
    environment: dict[Node, Node] = {}
    placeholder_position = 0
    used_positions: list[int] = []
    for node in module.graph.nodes:
        if node.op == "placeholder":
            if node in required:
                environment[node] = graph.placeholder(
                    str(node.target), type_expr=node.type
                )
                used_positions.append(placeholder_position)
            placeholder_position += 1
            continue
        if node.op == "output" or node not in required:
            continue
        copied = graph.node_copy(node, lambda item: environment[item])
        environment[node] = copied
    graph.output(tuple(environment[target] for target in targets))
    graph.lint()
    return GraphModule(module, graph), tuple(used_positions)


def _resolve_slice_input(
    record: StageRecord,
    position: int,
    *,
    stage_index: int,
    representative_root_inputs: tuple[object, ...],
    resolved: dict[StageOutputKey, torch.Tensor],
) -> object:
    try:
        captured = record.inputs[position]
        source = record.input_sources[position]
    except IndexError as error:
        raise CaptureError(
            "producer slice input is outside the stage contract"
        ) from error
    if not isinstance(captured, torch.Tensor):
        return captured
    if source is None:
        raise CaptureError(
            f"stage_{stage_index:04d} tensor input {position} has no provenance"
        )
    if source.root_input_index is not None:
        try:
            value = representative_root_inputs[source.root_input_index]
        except IndexError as error:
            raise CaptureError(
                f"stage_{stage_index:04d} root input {source.root_input_index} "
                "has no caller-supplied value"
            ) from error
        if not isinstance(value, torch.Tensor):
            raise CaptureError(
                f"stage_{stage_index:04d} root input {source.root_input_index} "
                "is not an authentic tensor"
            )
        return _on_capture_device(value, captured.device)
    assert source.producer_stage_index is not None
    assert source.producer_output_index is not None
    key = (source.producer_stage_index, source.producer_output_index)
    value = resolved[key]
    return _on_capture_device(value, captured.device)


def _stage_output_tensor(
    record: StageRecord,
    output_index: int,
    *,
    key: StageOutputKey,
    integral: bool,
) -> torch.Tensor:
    leaves, _ = tree_flatten(record.output)
    try:
        value = leaves[output_index]
    except IndexError as error:
        raise CaptureError(
            f"stage output {key} is outside its recorded contract"
        ) from error
    if not isinstance(value, torch.Tensor):
        raise CaptureError(f"stage output {key} is not a tensor")
    if integral and not _requires_authentic(value):
        raise CaptureError(f"stage output {key} is not integer or boolean")
    return value


def _requires_authentic(value: torch.Tensor) -> bool:
    return not value.is_floating_point() and not value.is_complex()


def _validate_geometry(
    value: torch.Tensor,
    expected: torch.Tensor,
    *,
    key: StageOutputKey,
    origin: str,
) -> None:
    actual_geometry = (tuple(value.shape), tuple(value.stride()), value.dtype)
    expected_geometry = (
        tuple(expected.shape),
        tuple(expected.stride()),
        expected.dtype,
    )
    if actual_geometry != expected_geometry:
        raise CaptureError(
            f"{origin} control value {key} has geometry {actual_geometry}, "
            f"expected {expected_geometry}"
        )


def _on_capture_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return this value where the producer slice runs, copying if it is not.

    Values are what a control input is for, and a copy keeps them exactly,
    so this changes where the slice reads from and nothing else.
    """

    if value.device == device:
        return value
    with unset_fake_temporarily(), torch.no_grad():
        return value.detach().to(device=device)


def _snapshot(value: torch.Tensor) -> torch.Tensor:
    # Partitioning normally runs under the capture FakeTensorMode, while an
    # authentic representative is deliberately a real CPU tensor. Keep the
    # snapshot outside fake dispatch just like the producer-slice execution.
    with unset_fake_temporarily(), torch.no_grad():
        source = value.detach().to(device="cpu")
        result = torch.empty_strided(
            tuple(source.shape),
            tuple(source.stride()),
            dtype=source.dtype,
            device="cpu",
        )
        result.copy_(source)
        return result


__all__ = ["StageOutputKey", "derive_authentic_control_values"]
