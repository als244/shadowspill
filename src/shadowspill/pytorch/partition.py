"""Automatic graph stages derived from repeated PyTorch module structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.fx import GraphModule, Interpreter, Node
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten

from .aot import ExportCapture, TrainingObjectiveCapture, capture_graph_pair
from .capture import AotGraphPair, GraphArtifact
from .contracts import CaptureError


@dataclass(frozen=True, slots=True)
class StageExample:
    """One split graph plus exact values observed at its functional ABI."""

    stage_id: str
    module_target: str
    graph_module: GraphModule
    inputs: tuple[object, ...]
    output: object


@dataclass(frozen=True, slots=True)
class TrainingStage:
    """One automatic stage and its two legal differentiation variants."""

    example: StageExample
    differentiable_output_indices: tuple[int, ...]
    save_pair: AotGraphPair
    recompute_pair: AotGraphPair


@dataclass(frozen=True, slots=True)
class PartitionedTrainingCapture:
    """One objective capture decomposed into executable training stages."""

    training: TrainingObjectiveCapture
    partitioned: PartitionedExport
    stages: tuple[TrainingStage, ...]


@dataclass(frozen=True, slots=True)
class PartitionedExport:
    """Executable split root and topologically ordered stage examples."""

    root: GraphModule
    root_inputs: tuple[object, ...]
    stages: tuple[StageExample, ...]
    repeated_groups: tuple[str, ...]


class _StageRecorder(Interpreter):
    def __init__(self, module: GraphModule) -> None:
        super().__init__(module)
        self.calls: list[tuple[str, tuple[object, ...], object]] = []

    def call_module(
        self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        if kwargs:
            raise CaptureError("automatic stage calls require a positional ABI")
        output = super().call_module(target, args, kwargs)
        self.calls.append((str(target), args, output))
        return output


class _TrainingGraphPairCache:
    """Reuse AOT graph code while preserving occurrence-specific storages."""

    def __init__(self) -> None:
        self._pairs: dict[
            tuple[str, tuple[int, ...]], tuple[AotGraphPair, AotGraphPair]
        ] = {}
        self.hits = 0
        self.misses = 0

    def resolve(
        self, example: StageExample, roots: tuple[int, ...]
    ) -> tuple[AotGraphPair, AotGraphPair]:
        stage_abi = GraphArtifact.capture(
            kind="inference",
            graph_module=example.graph_module,
            example_inputs=example.inputs,
        )
        key = (stage_abi.compatibility_digest, roots)
        existing = self._pairs.get(key)
        if existing is None:
            existing = (
                capture_graph_pair(
                    example.graph_module,
                    example.inputs,
                    original_output=example.output,
                    recomputation=False,
                    root_output_positions=roots,
                ),
                capture_graph_pair(
                    example.graph_module,
                    example.inputs,
                    original_output=example.output,
                    recomputation=True,
                    root_output_positions=roots,
                ),
            )
            self._pairs[key] = existing
            self.misses += 1
            return existing
        self.hits += 1
        save_pair, recompute_pair = existing
        return (
            _rebind_graph_pair(save_pair, example, roots),
            _rebind_graph_pair(recompute_pair, example, roots),
        )


def partition_export(
    capture: ExportCapture, module: nn.Module, *, partition: str = "auto"
) -> PartitionedExport:
    """Split at outer repeated-module boundaries or retain one whole graph."""

    if partition not in {"auto", "whole"}:
        raise CaptureError("partition must be 'auto' or 'whole'")
    repeated = _outer_repeated_groups(module) if partition == "auto" else ()
    graph_module = capture.exported_program.graph_module
    assignments = _partition_assignments(graph_module, repeated)
    try:
        root = split_module(
            graph_module,
            graph_module,
            lambda node: assignments[node],
            keep_original_order=True,
        )
        recorder = _StageRecorder(root)
        recorder.run(*capture.flat_inputs)
    except BaseException as exc:
        if isinstance(exc, CaptureError):
            raise
        raise CaptureError(f"automatic stage partition failed: {exc}") from exc
    stages: list[StageExample] = []
    for index, (target, inputs, output) in enumerate(recorder.calls):
        child = root.get_submodule(target)
        if not isinstance(child, GraphModule):
            raise CaptureError(f"partition {target!r} is not an FX GraphModule")
        stages.append(
            StageExample(
                stage_id=f"stage_{index:04d}",
                module_target=target,
                graph_module=child,
                inputs=inputs,
                output=output,
            )
        )
    if not stages:
        raise CaptureError("partitioning produced no executable stage")
    return PartitionedExport(
        root=root,
        root_inputs=capture.flat_inputs,
        stages=tuple(stages),
        repeated_groups=repeated,
    )


def capture_training_stages(
    partitioned: PartitionedExport,
    *,
    graph_pair_cache: _TrainingGraphPairCache | None = None,
) -> tuple[TrainingStage, ...]:
    """Differentiate every stage independently for save/recompute planning."""

    cache = graph_pair_cache or _TrainingGraphPairCache()
    stages: list[TrainingStage] = []
    for index, example in enumerate(partitioned.stages):
        leaves, _ = tree_flatten(example.output)
        if not leaves or any(not isinstance(value, torch.Tensor) for value in leaves):
            raise CaptureError("training stage outputs must be tensors")
        differentiable = tuple(
            position
            for position, value in enumerate(leaves)
            if value.requires_grad and (value.is_floating_point() or value.is_complex())
        )
        if not differentiable:
            raise CaptureError(f"training {example.stage_id} has no gradient output")
        roots = (0,) if index == len(partitioned.stages) - 1 else differentiable
        if any(position not in differentiable for position in roots):
            raise CaptureError("terminal objective loss is not differentiable")
        save_pair, recompute_pair = cache.resolve(example, roots)
        stages.append(
            TrainingStage(
                example=example,
                differentiable_output_indices=roots,
                save_pair=save_pair,
                recompute_pair=recompute_pair,
            )
        )
    return tuple(stages)


def partition_training_capture(
    capture: TrainingObjectiveCapture,
    *,
    partition: str = "auto",
    graph_pair_cache: _TrainingGraphPairCache | None = None,
) -> PartitionedTrainingCapture:
    """Partition and differentiate one captured objective template."""

    partitioned = partition_export(
        capture.exported, capture.capture_module, partition=partition
    )
    return PartitionedTrainingCapture(
        training=capture,
        partitioned=partitioned,
        stages=capture_training_stages(partitioned, graph_pair_cache=graph_pair_cache),
    )


def _rebind_graph_pair(
    pair: AotGraphPair,
    example: StageExample,
    roots: tuple[int, ...],
) -> AotGraphPair:
    """Bind shared AOT code to one stage occurrence's FakeTensor storages."""

    forward_arguments: list[torch.Tensor] = []
    for position in pair.forward.tensor_argument_positions:
        try:
            value = example.inputs[position]
        except IndexError as exc:
            raise CaptureError(
                "reused stage forward argument positions changed"
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise CaptureError("reused stage tensor argument became static")
        forward_arguments.append(value.detach())
    if len(forward_arguments) != pair.forward.argument_count:
        raise CaptureError("reused stage forward tensor argument count changed")
    forward = GraphArtifact.capture(
        kind="forward",
        graph_module=pair.forward.graph_module,
        example_inputs=tuple(forward_arguments),
    )
    if forward.compatibility_digest != pair.forward.compatibility_digest:
        raise CaptureError("reused stage forward ABI differs from its representative")

    with torch.no_grad():
        forward_values, _ = tree_flatten(
            forward.graph_module(*forward.example_arguments)
        )
    residuals = (
        tuple(forward_values[-pair.saved_value_count :])
        if pair.saved_value_count
        else ()
    )
    tangents: list[torch.Tensor] = []
    for position in roots:
        try:
            value = forward_values[position]
        except IndexError as exc:
            raise CaptureError("reused stage tangent position changed") from exc
        if not isinstance(value, torch.Tensor):
            raise CaptureError("reused stage tangent output became static")
        tangents.append(torch.ones_like(value))
    backward = GraphArtifact.capture(
        kind="backward",
        graph_module=pair.backward.graph_module,
        example_inputs=(*residuals, *tangents),
    )
    if backward.compatibility_digest != pair.backward.compatibility_digest:
        raise CaptureError("reused stage backward ABI differs from its representative")
    return AotGraphPair(
        forward=forward,
        backward=backward,
        recomputation=pair.recomputation,
        saved_value_count=pair.saved_value_count,
    )


def capture_forward_stages(
    partitioned: PartitionedExport,
) -> tuple[GraphArtifact, ...]:
    """Return one structural inference ABI for each automatic stage."""

    return tuple(
        GraphArtifact.capture(
            kind="inference",
            graph_module=stage.graph_module,
            example_inputs=stage.inputs,
        )
        for stage in partitioned.stages
    )


def _outer_repeated_groups(module: nn.Module) -> tuple[str, ...]:
    candidates: list[str] = []
    for path, parent in module.named_modules():
        children = tuple(parent.named_children())
        if len(children) < 2:
            continue
        type_counts: dict[type[nn.Module], int] = {}
        for _name, child in children:
            type_counts[type(child)] = type_counts.get(type(child), 0) + 1
        if max(type_counts.values(), default=0) < 2:
            continue
        if any(
            path == selected or path.startswith(f"{selected}.")
            for selected in candidates
        ):
            continue
        candidates.append(path)
    return tuple(candidates)


def _anchor(node: Node, repeated_groups: tuple[str, ...]) -> str | None:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, dict):
        return None
    paths = tuple(
        value[0]
        for value in stack.values()
        if isinstance(value, tuple) and value and isinstance(value[0], str)
    )
    matches: list[tuple[int, str]] = []
    for group in repeated_groups:
        prefix = f"{group}." if group else ""
        for path in paths:
            if path == group or not path.startswith(prefix):
                continue
            child = path[len(prefix) :].split(".", 1)[0]
            matches.append((len(group), f"{prefix}{child}"))
    return max(matches)[1] if matches else None


def _partition_assignments(
    graph_module: GraphModule, repeated_groups: tuple[str, ...]
) -> dict[Node, int]:
    assignments: dict[Node, int] = {}
    partition_id = 0
    previous_anchor: str | None = None
    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        current_anchor = _anchor(node, repeated_groups)
        if current_anchor is not None and current_anchor != previous_anchor:
            if previous_anchor is not None:
                partition_id += 1
            previous_anchor = current_anchor
        assignments[node] = partition_id
    if not assignments:
        raise CaptureError("export graph has no executable operations")
    return assignments
