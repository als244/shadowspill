"""Automatic graph stages derived from repeated PyTorch module structure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch.fx import GraphModule, Interpreter, Node
from torch.fx.passes.split_module import split_module

from .aot import ExportCapture, capture_graph_pair
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
    save_pair: AotGraphPair
    recompute_pair: AotGraphPair


@dataclass(frozen=True, slots=True)
class PartitionedExport:
    """Executable split root and topologically ordered stage examples."""

    root: GraphModule
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
        stages=tuple(stages),
        repeated_groups=repeated,
    )


def capture_training_stages(
    partitioned: PartitionedExport,
) -> tuple[TrainingStage, ...]:
    """Differentiate every stage independently for save/recompute planning."""

    return tuple(
        TrainingStage(
            example=example,
            save_pair=capture_graph_pair(
                example.graph_module,
                example.inputs,
                original_output=example.output,
                recomputation=False,
            ),
            recompute_pair=capture_graph_pair(
                example.graph_module,
                example.inputs,
                original_output=example.output,
                recomputation=True,
            ),
        )
        for example in partitioned.stages
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
