"""Resolve lifetime choices and transform one complete explicit task."""

from __future__ import annotations

from dataclasses import asdict
from fnmatch import fnmatchcase
from typing import Any

import torch
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TensorGeometry
from shadowspill.pytorch.capture.source import source_operation
from shadowspill.pytorch.partition.policy import (
    PartitionSpec,
    resolve_partition_assignments,
)

from .lifetime_analysis import _effects, analyze_lifetimes
from .lifetime_policy import ExecutionLifetimes, LifetimeOperation, LifetimeSelection
from .lifetime_schedule import sink_readonly
from .lifetime_views import prepare_internal_views


def _geometry(values: object) -> tuple[TensorGeometry | None, ...]:
    leaves, _ = tree_flatten(values)
    return tuple(
        TensorGeometry.from_tensor(value) if isinstance(value, torch.Tensor) else None
        for value in leaves
    )


def _accept(
    choices: dict[str, bool], selection: LifetimeSelection, info: LifetimeOperation
) -> None:
    for name in ("move", "barrier", "compact"):
        value = getattr(selection, name)
        if value is None:
            continue
        if name in choices and choices[name] != value:
            raise CaptureError(
                f"conflicting lifetime {name} choices for "
                f"operation {info.index} ({info.target})"
            )
        choices[name] = value


def apply_lifetimes(
    graph: GraphModule,
    artifact: GraphArtifact,
    policy: ExecutionLifetimes,
    *,
    partition: PartitionSpec,
    stage_id: str,
    checkpoint_stage_id: str | None,
    variant: str | None,
    gradient_leaves: tuple[int, ...] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return separate structural identity and occurrence inspection records."""
    selections, occurrences = [], []
    movable, barriers, compactable = set(), set(), set()
    fences, contracts = _effects(
        tuple(graph.graph.nodes), {item.target: item for item in policy.readonly}
    )
    for index, node in enumerate(graph.graph.nodes):
        if node.op != "call_function":
            continue
        source = source_operation(node)
        paths = tuple(
            dict.fromkeys(
                path
                for path, _ in node.meta.get("nn_module_stack", {}).values()
                if path
            )
        )
        operands = tree_map(
            lambda value: value.meta.get("val") if isinstance(value, Node) else value,
            (node.args, node.kwargs),
        )
        info = LifetimeOperation(
            index,
            str(node.target),
            source.target if source else None,
            paths,
            fences.get(node),
            artifact.kind,
            variant,
            stage_id,
            checkpoint_stage_id or stage_id,
            _geometry(operands),
            _geometry(node.meta.get("val")),
        )
        choices: dict[str, bool] = {}
        for rule in policy.rules:
            if (
                rule.target == info.target
                and (
                    rule.source_target is None
                    or rule.source_target == info.source_target
                )
                and (
                    rule.module_pattern is None
                    or any(fnmatchcase(path, rule.module_pattern) for path in paths)
                )
            ):
                _accept(
                    choices,
                    LifetimeSelection(rule.move, rule.barrier, rule.compact),
                    info,
                )
        if policy.select is not None:
            selected = policy.select(info)
            if selected is not None:
                if not isinstance(selected, LifetimeSelection):
                    raise CaptureError(
                        "lifetime select must return LifetimeSelection or None"
                    )
                _accept(choices, selected, info)
        if choices:
            selections.append({"index": index, "target": info.target, **choices})
            occurrences.append(
                {
                    "index": index,
                    "target": info.target,
                    "source_target": info.source_target,
                    "module_paths": list(paths),
                    **choices,
                }
            )
        if choices.get("move"):
            movable.add(node)
        if choices.get("barrier"):
            barriers.add(node)
        if choices.get("compact"):
            compactable.add(node)
    schedule: dict[str, Any] | None = None
    if movable or barriers:
        analysis = analyze_lifetimes(
            graph, artifact.example_arguments, readonly=policy.readonly
        )
        contracts = dict(analysis.readonly_contracts)
        schedule = asdict(
            sink_readonly(
                graph,
                analysis,
                movable=frozenset(movable),
                barriers=frozenset(barriers),
                gradient_leaves=gradient_leaves,
            )
        )
    identity = {
        "schema": 1,
        "selections": selections,
        "readonly": contracts,
        "schedule": schedule,
    }
    report = {
        "stage_id": stage_id,
        "checkpoint_stage_id": checkpoint_stage_id or stage_id,
        "kind": artifact.kind,
        "variant": variant,
        "selections": occurrences,
        "schedule": schedule,
        "memory_scope": (
            "internal semantic backing spans; "
            "excludes inputs, opaque workspace and allocator overhead"
        ),
    }
    if compactable:
        from .lifetime_compaction import compact_internal_views

        analysis = analyze_lifetimes(
            graph, artifact.example_arguments, readonly=policy.readonly
        )
        preparation = prepare_internal_views(
            graph,
            analysis,
            selected=frozenset(compactable),
            barriers=frozenset(barriers),
        )
        analysis = analyze_lifetimes(
            graph, artifact.example_arguments, readonly=policy.readonly
        )
        assignments, _ = resolve_partition_assignments(graph, graph, partition)
        changed, compaction = compact_internal_views(
            graph,
            analysis,
            artifact.example_arguments,
            selected=frozenset(compactable),
            regions=assignments,
            barriers=frozenset(barriers),
            readonly=policy.readonly,
            layout=policy.layout,
        )
        graph.graph = changed.graph
        graph.recompile()
        identity.update(
            schema=2, views=asdict(preparation), compaction=asdict(compaction)
        )
        report.update(views=asdict(preparation), compaction=asdict(compaction))
    return identity, report
