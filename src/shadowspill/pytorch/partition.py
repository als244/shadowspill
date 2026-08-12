"""Automatic graph stages derived from repeated PyTorch module structure."""

from __future__ import annotations

import operator
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule, Interpreter, Node
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten

from .aot import (
    ExportCapture,
    TrainingObjectiveCapture,
    _tensor_only_mutations,
    capture_graph_pair,
)
from .capture import AotGraphPair, GraphArtifact
from .contracts import CaptureError
from .output_contract import ExplicitMutation


@dataclass(frozen=True, slots=True)
class StageValueSource:
    """Root-graph provenance for one positional stage input."""

    root_input_index: int | None = None
    producer_stage_index: int | None = None
    producer_output_index: int | None = None

    def __post_init__(self) -> None:
        root = self.root_input_index is not None
        produced = self.producer_stage_index is not None
        if root == produced:
            raise ValueError("stage source must be either a root input or stage output")
        if root:
            if self.root_input_index is None or self.root_input_index < 0:
                raise ValueError("stage root-input source is invalid")
            if self.producer_output_index is not None:
                raise ValueError("root-input source cannot name a producer output")
        else:
            if (
                self.producer_stage_index is None
                or self.producer_stage_index < 0
                or self.producer_output_index is None
                or self.producer_output_index < 0
            ):
                raise ValueError("stage-output source is invalid")


@dataclass(frozen=True, slots=True)
class StageExample:
    """One split graph plus exact values observed at its functional ABI."""

    stage_id: str
    module_target: str
    graph_module: GraphModule
    inputs: tuple[object, ...]
    input_sources: tuple[StageValueSource | None, ...]
    mutations: tuple[ExplicitMutation, ...]
    user_output_indices: tuple[int, ...]
    output: object


_StageRecord = tuple[
    str,
    GraphModule,
    tuple[object, ...],
    tuple[StageValueSource | None, ...],
    object,
]


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
    user_output_indices: tuple[int, ...]


def training_parameter_stage_owners(
    captures: tuple[PartitionedTrainingCapture, ...],
    parameter_names: Collection[str],
) -> dict[str, tuple[int, ...]]:
    """Return the training stages whose backward passes contribute each parameter.

    Export makes parameters explicit root inputs.  Stage partitioning preserves
    that provenance in :class:`StageValueSource`, so optimizer grouping can use
    the same semantic stage boundaries without inspecting module-name patterns
    or runtime allocation behavior.
    """

    known = frozenset(parameter_names)
    owners: dict[str, set[int]] = {}
    expected_stage_count: int | None = None
    for capture in captures:
        if expected_stage_count is None:
            expected_stage_count = len(capture.stages)
        elif len(capture.stages) != expected_stage_count:
            raise CaptureError(
                "microbatch positions produced different training-stage counts"
            )
        input_specs = tuple(
            capture.training.exported.exported_program.graph_signature.input_specs
        )
        for stage_index, stage in enumerate(capture.stages):
            for source in stage.example.input_sources:
                if source is None or source.root_input_index is None:
                    continue
                try:
                    spec = input_specs[source.root_input_index]
                except IndexError as exc:
                    raise CaptureError(
                        "stage parameter provenance refers outside the Export ABI"
                    ) from exc
                if spec.kind is not InputKind.PARAMETER:
                    continue
                target = spec.target
                if not isinstance(target, str) or not target.startswith("model."):
                    raise CaptureError(
                        "objective Export parameter target is not rooted at model: "
                        f"{target!r}"
                    )
                name = target.removeprefix("model.")
                if name not in known:
                    raise CaptureError(
                        f"stage parameter {name!r} is absent from the optimizer model"
                    )
                owners.setdefault(name, set()).add(stage_index)
    return {name: tuple(sorted(indices)) for name, indices in owners.items()}


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
            tuple[str, tuple[int, ...], bool], tuple[AotGraphPair, AotGraphPair]
        ] = {}
        self.hits = 0
        self.misses = 0

    def resolve(
        self,
        example: StageExample,
        roots: tuple[int, ...],
        *,
        specialize_unit_tangents: bool,
    ) -> tuple[AotGraphPair, AotGraphPair]:
        stage_abi = GraphArtifact.capture(
            kind="inference",
            graph_module=example.graph_module,
            example_inputs=example.inputs,
            explicit_mutations=example.mutations,
        )
        key = (stage_abi.compatibility_digest, roots, specialize_unit_tangents)
        existing = self._pairs.get(key)
        if existing is None:
            existing = (
                capture_graph_pair(
                    example.graph_module,
                    example.inputs,
                    original_output=example.output,
                    recomputation=False,
                    root_output_positions=roots,
                    specialize_unit_tangents=specialize_unit_tangents,
                    explicit_mutations=example.mutations,
                ),
                capture_graph_pair(
                    example.graph_module,
                    example.inputs,
                    original_output=example.output,
                    recomputation=True,
                    root_output_positions=roots,
                    specialize_unit_tangents=specialize_unit_tangents,
                    explicit_mutations=example.mutations,
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
    stage_records: list[_StageRecord] = []
    call_nodes = tuple(node for node in root.graph.nodes if node.op == "call_module")
    if len(call_nodes) != len(recorder.calls):
        raise CaptureError("split root task topology differs from recorded calls")
    placeholders = tuple(node for node in root.graph.nodes if node.op == "placeholder")
    placeholder_index = {node: index for index, node in enumerate(placeholders)}
    stage_index_by_node = {node: index for index, node in enumerate(call_nodes)}
    for (target, inputs, output), call_node in zip(
        recorder.calls, call_nodes, strict=True
    ):
        child = root.get_submodule(target)
        if not isinstance(child, GraphModule):
            raise CaptureError(f"partition {target!r} is not an FX GraphModule")
        argument_leaves, _ = tree_flatten(call_node.args)
        input_leaves, _ = tree_flatten(inputs)
        if len(argument_leaves) != len(input_leaves):
            raise CaptureError("stage input structure differs from split-root topology")
        sources = tuple(
            _stage_value_source(
                leaf,
                placeholder_index=placeholder_index,
                stage_index_by_node=stage_index_by_node,
            )
            if isinstance(value, torch.Tensor)
            else None
            for leaf, value in zip(argument_leaves, input_leaves, strict=True)
        )
        stage_records.append((target, child, inputs, sources, output))
    if not stage_records:
        raise CaptureError("partitioning produced no executable stage")
    mutations_by_stage = _partition_mutations(
        capture,
        root,
        call_nodes=call_nodes,
        placeholder_index=placeholder_index,
        stage_index_by_node=stage_index_by_node,
        stage_records=tuple(stage_records),
    )
    user_outputs_by_stage = _partition_user_outputs(
        capture,
        root,
        placeholder_index=placeholder_index,
        stage_index_by_node=stage_index_by_node,
    )
    stages = tuple(
        StageExample(
            stage_id=f"stage_{index:04d}",
            module_target=target,
            graph_module=child,
            inputs=inputs,
            input_sources=sources,
            mutations=mutations_by_stage.get(index, ()),
            user_output_indices=user_outputs_by_stage.get(index, ()),
            output=output,
        )
        for index, (target, child, inputs, sources, output) in enumerate(stage_records)
    )
    return PartitionedExport(
        root=root,
        root_inputs=capture.flat_inputs,
        stages=stages,
        repeated_groups=repeated,
        user_output_indices=capture.user_output_indices,
    )


def _partition_user_outputs(
    capture: ExportCapture,
    root: GraphModule,
    *,
    placeholder_index: dict[Node, int],
    stage_index_by_node: dict[Node, int],
) -> dict[int, tuple[int, ...]]:
    """Project root user outputs onto their stage-local output positions."""

    output_node = next(node for node in root.graph.nodes if node.op == "output")
    output_leaves, _ = tree_flatten(output_node.args[0])
    result: dict[int, list[int]] = {}
    for output_index in capture.user_output_indices:
        try:
            root_output = output_leaves[output_index]
        except IndexError as exc:
            raise CaptureError("Export user output is absent from split root") from exc
        source = _stage_value_source(
            root_output,
            placeholder_index=placeholder_index,
            stage_index_by_node=stage_index_by_node,
        )
        if source.producer_stage_index is None or source.producer_output_index is None:
            raise CaptureError("Export user output is not stage-produced")
        result.setdefault(source.producer_stage_index, []).append(
            source.producer_output_index
        )
    return {index: tuple(values) for index, values in result.items()}


def _stage_value_source(
    node: object,
    *,
    placeholder_index: dict[Node, int],
    stage_index_by_node: dict[Node, int],
) -> StageValueSource:
    if not isinstance(node, Node):
        raise CaptureError("tensor stage input has no split-root FX provenance")
    root_index = placeholder_index.get(node)
    if root_index is not None:
        return StageValueSource(root_input_index=root_index)
    stage_index = stage_index_by_node.get(node)
    if stage_index is not None:
        return StageValueSource(
            producer_stage_index=stage_index,
            producer_output_index=0,
        )
    if node.op == "call_function" and node.target is operator.getitem:
        producer, output_index = node.args[:2]
        if isinstance(producer, Node) and isinstance(output_index, int):
            stage_index = stage_index_by_node.get(producer)
            if stage_index is not None:
                return StageValueSource(
                    producer_stage_index=stage_index,
                    producer_output_index=output_index,
                )
    raise CaptureError(
        "tensor stage input has unsupported split-root provenance: "
        f"node={node.name}, op={node.op}, target={node.target}"
    )


def _partition_mutations(
    capture: ExportCapture,
    root: GraphModule,
    *,
    call_nodes: tuple[Node, ...],
    placeholder_index: dict[Node, int],
    stage_index_by_node: dict[Node, int],
    stage_records: tuple[_StageRecord, ...],
) -> dict[int, tuple[ExplicitMutation, ...]]:
    """Project root Export mutations onto the stage that creates the value."""

    del call_nodes
    output_node = next(node for node in root.graph.nodes if node.op == "output")
    output_leaves, _ = tree_flatten(output_node.args[0])
    result: dict[int, list[ExplicitMutation]] = {}
    for mutation in capture.mutations:
        try:
            root_output = output_leaves[mutation.output_index]
        except IndexError as exc:
            raise CaptureError(
                "Export mutation output is absent from split root"
            ) from exc
        source = _stage_value_source(
            root_output,
            placeholder_index=placeholder_index,
            stage_index_by_node=stage_index_by_node,
        )
        if source.producer_stage_index is None or source.producer_output_index is None:
            raise CaptureError("Export mutation replacement is not stage-produced")
        stage_index = source.producer_stage_index
        sources = stage_records[stage_index][3]
        candidates = tuple(
            position
            for position, input_source in enumerate(sources)
            if input_source is not None
            and input_source.root_input_index == mutation.input_index
        )
        if len(candidates) != 1:
            raise CaptureError(
                "Export mutation target does not resolve to one producer-stage input: "
                f"stage={stage_index}, target={mutation.target!r}, "
                f"root_input={mutation.input_index}, candidates={candidates}"
            )
        result.setdefault(stage_index, []).append(
            ExplicitMutation(
                candidates[0],
                source.producer_output_index,
                mutation.target,
            )
        )
    return {index: tuple(values) for index, values in result.items()}


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
        roots = (
            (partitioned.user_output_indices[0],)
            if index == len(partitioned.stages) - 1
            else differentiable
        )
        if any(position not in differentiable for position in roots):
            raise CaptureError("terminal objective loss is not differentiable")
        save_pair, recompute_pair = cache.resolve(
            example,
            roots,
            specialize_unit_tangents=index == len(partitioned.stages) - 1,
        )
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
    """Bind shared AOT code to one occurrence's symbolic input geometry."""

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
        explicit_mutations=_tensor_only_mutations(
            example.mutations, tuple(example.inputs)
        ),
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
    explicit_root_count = len(roots) - pair.specialized_unit_tangent_count
    if explicit_root_count < 0:
        raise CaptureError("specialized tangent count exceeds stage roots")
    tangents: list[torch.Tensor] = []
    for position in roots[:explicit_root_count]:
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
        specialized_unit_tangent_count=pair.specialized_unit_tangent_count,
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
            explicit_mutations=stage.mutations,
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
