"""Forward/backward pair records and loss/metric output structure."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from shadowspill.errors import ObjectiveError

from ..retention import RetentionSummary
from .geometry import TaskInputProvenance
from .graph import GraphArtifact


@dataclass(frozen=True, slots=True)
class AotGraphPair:
    """Explicit forward/backward pair for one graph-pair choice."""

    forward: GraphArtifact
    backward: GraphArtifact
    #: What the partition regenerated in the backward, under which policy.
    retention: RetentionSummary
    saved_value_count: int
    specialized_unit_tangent_count: int = 0
    #: Backward outputs follow logical tensors, even when forward inputs expand.
    gradient_provenance: tuple[TaskInputProvenance, ...] = ()


@dataclass(frozen=True, slots=True)
class ObjectiveSchema:
    """Loss/metrics output reconstruction without retaining framework closures."""

    metric_tree_spec: TreeSpec
    metric_leaf_count: int
    tensor_metric_positions: tuple[int, ...]
    static_metric_leaves: tuple[tuple[int, Any], ...]

    def rebuild_metrics(self, tensors: tuple[torch.Tensor, ...]) -> Any:
        if len(tensors) != len(self.tensor_metric_positions):
            raise ObjectiveError("objective metric tensor count changed")
        leaves: list[Any] = [None] * self.metric_leaf_count
        for position, value in self.static_metric_leaves:
            leaves[position] = copy.deepcopy(value)
        for position, value in zip(self.tensor_metric_positions, tensors, strict=True):
            leaves[position] = value
        return tree_unflatten(leaves, self.metric_tree_spec)


def capture_objective_schema(metrics: Any) -> ObjectiveSchema:
    """Freeze tensor/static metric positions after a storage-free probe."""

    leaves, tree_spec = tree_flatten(metrics)
    tensor_positions = tuple(
        index for index, value in enumerate(leaves) if isinstance(value, torch.Tensor)
    )
    static: list[tuple[int, Any]] = []
    try:
        for index, value in enumerate(leaves):
            if not isinstance(value, torch.Tensor):
                static.append((index, copy.deepcopy(value)))
    except BaseException as exc:
        raise ObjectiveError("static objective metrics must be copyable") from exc
    return ObjectiveSchema(
        metric_tree_spec=tree_spec,
        metric_leaf_count=len(leaves),
        tensor_metric_positions=tensor_positions,
        static_metric_leaves=tuple(static),
    )
