"""The traced update partitioned into independent tasks at the backward frontier
where each parameter's gradient is final."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.fx import GraphModule

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
)

from .artifacts import (
    OptimizerTask,
    OptimizerTensorBinding,
)
from .bindings import (
    completion_stage,
)


@dataclass(frozen=True, slots=True)
class _OptimizerComponent:
    nodes: tuple[torch.fx.Node, ...]
    input_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _OptimizerComponentGroup:
    completion_stage: int | None
    components: tuple[_OptimizerComponent, ...]


class _DisjointSets:
    """Minimal union-find used to find independent optimizer updates."""

    def __init__(self, size: int) -> None:
        self._parents = list(range(size))

    def find(self, index: int) -> int:
        while self._parents[index] != index:
            self._parents[index] = self._parents[self._parents[index]]
            index = self._parents[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parents[right_root] = left_root


def partition_optimizer_graph(
    artifact: GraphArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
    *,
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[OptimizerTask, ...]:
    """Partition dependency-closed updates at their backward-ready frontier."""

    placeholders, operations = _optimizer_graph_nodes(artifact, bindings)
    if len(operations) < 2:
        return (_whole_optimizer_task(artifact, bindings, parameter_stage_owners),)
    components = _optimizer_components(placeholders, operations, bindings)
    if len(components) == 1:
        return (_whole_optimizer_task(artifact, bindings, parameter_stage_owners),)
    groups = _group_optimizer_components(components, bindings, parameter_stage_owners)
    return tuple(
        _build_optimizer_component_task(
            artifact, bindings, placeholders, operations, group
        )
        for group in groups
    )


def _optimizer_graph_nodes(
    artifact: GraphArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
) -> tuple[tuple[torch.fx.Node, ...], tuple[torch.fx.Node, ...]]:
    placeholders = tuple(
        node for node in artifact.graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(bindings):
        raise CaptureError("optimizer placeholder inventory changed after lifting")
    operations = tuple(
        node
        for node in artifact.graph_module.graph.nodes
        if node.op not in {"placeholder", "output"}
    )
    return placeholders, operations


def _whole_optimizer_task(
    artifact: GraphArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None,
) -> OptimizerTask:
    return OptimizerTask(
        artifact,
        tuple(binding.name for binding in bindings),
        tuple(binding.name for binding in bindings if binding.mutable),
        completion_stage(bindings, parameter_stage_owners),
    )


def _optimizer_components(
    placeholders: tuple[torch.fx.Node, ...],
    operations: tuple[torch.fx.Node, ...],
    bindings: tuple[OptimizerTensorBinding, ...],
) -> tuple[_OptimizerComponent, ...]:
    positions = {node: index for index, node in enumerate(operations)}
    sets = _DisjointSets(len(operations))
    dependencies = _optimizer_dependencies(operations, positions, sets)
    _join_mutable_consumers(placeholders, bindings, positions, dependencies, sets)
    component_nodes = _ordered_component_nodes(operations, positions, sets)
    return tuple(
        _OptimizerComponent(
            nodes=nodes,
            input_positions=_component_input_positions(
                nodes, placeholders, dependencies
            ),
        )
        for nodes in component_nodes
    )


def _optimizer_dependencies(
    operations: tuple[torch.fx.Node, ...],
    positions: Mapping[torch.fx.Node, int],
    sets: _DisjointSets,
) -> dict[torch.fx.Node, set[torch.fx.Node]]:
    dependencies_by_operation: dict[torch.fx.Node, set[torch.fx.Node]] = {}
    for operation in operations:
        dependencies: set[torch.fx.Node] = set()
        stack = list(operation.all_input_nodes)
        while stack:
            dependency = stack.pop()
            if dependency in dependencies:
                continue
            dependencies.add(dependency)
            if dependency in positions:
                sets.union(positions[operation], positions[dependency])
            if dependency.op != "placeholder":
                stack.extend(dependency.all_input_nodes)
        dependencies_by_operation[operation] = dependencies
    return dependencies_by_operation


def _join_mutable_consumers(
    placeholders: tuple[torch.fx.Node, ...],
    bindings: tuple[OptimizerTensorBinding, ...],
    positions: Mapping[torch.fx.Node, int],
    dependencies: Mapping[torch.fx.Node, set[torch.fx.Node]],
    sets: _DisjointSets,
) -> None:
    for placeholder, binding in zip(placeholders, bindings, strict=True):
        if not binding.mutable:
            continue
        consumers = sorted(
            positions[operation]
            for operation, required in dependencies.items()
            if placeholder in required
        )
        for consumer in consumers[1:]:
            sets.union(consumers[0], consumer)


def _ordered_component_nodes(
    operations: tuple[torch.fx.Node, ...],
    positions: Mapping[torch.fx.Node, int],
    sets: _DisjointSets,
) -> tuple[tuple[torch.fx.Node, ...], ...]:
    components: dict[int, list[torch.fx.Node]] = {}
    for operation in operations:
        components.setdefault(sets.find(positions[operation]), []).append(operation)
    return tuple(
        tuple(nodes)
        for _root, nodes in sorted(
            components.items(),
            key=lambda item: min(positions[node] for node in item[1]),
        )
    )


def _component_input_positions(
    nodes: tuple[torch.fx.Node, ...],
    placeholders: tuple[torch.fx.Node, ...],
    dependencies: Mapping[torch.fx.Node, set[torch.fx.Node]],
) -> tuple[int, ...]:
    required = {
        dependency
        for operation in nodes
        for dependency in dependencies[operation]
        if dependency.op == "placeholder"
    }
    return tuple(
        index
        for index, placeholder in enumerate(placeholders)
        if placeholder in required
    )


def _group_optimizer_components(
    components: tuple[_OptimizerComponent, ...],
    bindings: tuple[OptimizerTensorBinding, ...],
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None,
) -> tuple[_OptimizerComponentGroup, ...]:
    if parameter_stage_owners is None:
        return tuple(
            _OptimizerComponentGroup(None, (component,)) for component in components
        )
    grouped: dict[int | None, list[_OptimizerComponent]] = defaultdict(list)
    for component in components:
        bound = tuple(bindings[index] for index in component.input_positions)
        grouped[completion_stage(bound, parameter_stage_owners)].append(component)
    return tuple(
        _OptimizerComponentGroup(completion_stage, tuple(members))
        for completion_stage, members in sorted(
            grouped.items(),
            key=lambda item: (
                item[0] is None,
                -(item[0] if item[0] is not None else -1),
            ),
        )
    )


def _build_optimizer_component_task(
    artifact: GraphArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
    placeholders: tuple[torch.fx.Node, ...],
    operations: tuple[torch.fx.Node, ...],
    group: _OptimizerComponentGroup,
) -> OptimizerTask:
    positions = tuple(
        sorted(
            {
                position
                for component in group.components
                for position in component.input_positions
            }
        )
    )
    component_nodes = {
        node for component in group.components for node in component.nodes
    }
    graph = _copy_optimizer_component_graph(
        placeholders, operations, positions, component_nodes, bindings
    )
    component_artifact = GraphArtifact.capture(
        kind="optimizer",
        graph_module=GraphModule(artifact.graph_module, graph),
        example_inputs=tuple(
            artifact.example_arguments[position] for position in positions
        ),
        input_provenance=tuple(
            artifact.input_provenance[position] for position in positions
        ),
    )
    mutable_positions = tuple(
        position for position in positions if bindings[position].mutable
    )
    return OptimizerTask(
        component_artifact,
        tuple(bindings[position].name for position in positions),
        tuple(bindings[position].name for position in mutable_positions),
        group.completion_stage,
    )


def _copy_optimizer_component_graph(
    placeholders: tuple[torch.fx.Node, ...],
    operations: tuple[torch.fx.Node, ...],
    positions: tuple[int, ...],
    component_nodes: set[torch.fx.Node],
    bindings: tuple[OptimizerTensorBinding, ...],
) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    environment: dict[torch.fx.Node, torch.fx.Node] = {}
    for local_index, position in enumerate(positions):
        environment[placeholders[position]] = graph.placeholder(
            f"optimizer_tensor_{local_index:04d}"
        )
    for operation in operations:
        if operation not in component_nodes:
            continue
        copied = graph.create_node(
            operation.op,
            operation.target,
            torch.fx.map_arg(operation.args, environment.__getitem__),
            torch.fx.map_arg(operation.kwargs, environment.__getitem__),
            type_expr=operation.type,
        )
        copied.meta = copy.copy(operation.meta)
        environment[operation] = copied
    mutable_positions = tuple(
        position for position in positions if bindings[position].mutable
    )
    graph.output(
        tuple(environment[placeholders[position]] for position in mutable_positions)
    )
    return graph
