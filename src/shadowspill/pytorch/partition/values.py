"""Authentic control-value propagation across partition boundaries.

Integer and boolean task inputs can select kernels, allocation paths, and
provider caches.  They therefore cannot use geometry-only synthetic values.
This module evaluates only the producer dependency slice needed to construct
such a value from caller-supplied roots or previously resolved control values.
It never executes an unrelated model stage to manufacture profiling inputs.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx import Graph, GraphModule, Node
from torch.utils._pytree import tree_flatten

from ..contracts import CaptureError
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
    for stage_index in sorted({stage for stage, _output in required}):
        output_indices = tuple(
            output
            for stage, output in sorted(required)
            if stage == stage_index and (stage, output) not in resolved
        )
        if not output_indices:
            continue
        observed = _evaluate_control_slice(
            split.stages[stage_index],
            stage_index=stage_index,
            output_indices=output_indices,
            representative_root_inputs=representative_root_inputs,
            resolved=resolved,
        )
        resolved.update(observed)
    missing = sorted(required - resolved.keys())
    if missing:
        raise AssertionError(f"control-value propagation is incomplete: {missing}")
    return resolved


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
        expected = _stage_output_tensor(split.stages[key[0]], key[1], key=key)
        _validate_geometry(value, expected, key=key, origin="caller")
        result[key] = _snapshot(value)
    return result


def _evaluate_control_slice(
    record: StageRecord,
    *,
    stage_index: int,
    output_indices: tuple[int, ...],
    representative_root_inputs: tuple[object, ...],
    resolved: dict[StageOutputKey, torch.Tensor],
) -> dict[StageOutputKey, torch.Tensor]:
    sliced, input_positions = _slice_outputs(record.graph_module, output_indices)
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
        raise CaptureError(
            "failed to derive authentic integer/boolean stage output values: "
            f"producer=stage_{stage_index:04d}, outputs=[{names}], "
            f"cause={error}"
        ) from error

    leaves, _ = tree_flatten(output)
    if len(leaves) != len(output_indices):
        raise CaptureError("control-value producer slice changed its output arity")
    result: dict[StageOutputKey, torch.Tensor] = {}
    for output_index, value in zip(output_indices, leaves, strict=True):
        key = (stage_index, output_index)
        if not isinstance(value, torch.Tensor):
            raise CaptureError(f"derived control value {key} is not a tensor")
        expected = _stage_output_tensor(record, output_index, key=key)
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
        environment[node] = graph.node_copy(node, lambda item: environment[item])
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
        return value
    assert source.producer_stage_index is not None
    assert source.producer_output_index is not None
    key = (source.producer_stage_index, source.producer_output_index)
    value = resolved.get(key)
    if value is None:
        raise CaptureError(
            "authentic control-value derivation depends on an unresolved stage "
            f"output: consumer=stage_{stage_index:04d}, input={position}, "
            f"producer=stage_{key[0]:04d}.output_{key[1]:04d}; provide an "
            "authentic caller value or choose a partition exposing its producer"
        )
    return value


def _stage_output_tensor(
    record: StageRecord,
    output_index: int,
    *,
    key: StageOutputKey,
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
    if not _requires_authentic(value):
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
