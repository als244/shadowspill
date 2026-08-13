"""Built-in and user-defined policies for legal contiguous FX stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

import torch.nn as nn
from torch.fx import GraphModule, Node

from ..contracts import CaptureError


@runtime_checkable
class PartitionPolicy(Protocol):
    """Choose a contiguous stage number for every executable FX node.

    Implementations receive the captured graph and its source module and return
    a mapping from FX node name to an integer stage label. Labels need not be
    dense, but every executable node must appear exactly once and each label
    must occupy one contiguous topological interval.
    """

    def assign_stages(
        self,
        graph_module: GraphModule,
        module: nn.Module,
    ) -> Mapping[str, int]: ...


PartitionSpec = Literal["auto", "whole"] | PartitionPolicy


def resolve_partition_assignments(
    graph_module: GraphModule,
    module: nn.Module,
    policy: PartitionSpec,
) -> tuple[dict[Node, int], tuple[str, ...]]:
    """Resolve and validate one built-in or custom partition policy."""

    if isinstance(policy, str):
        return _resolve_builtin_policy(graph_module, module, policy)
    if not isinstance(policy, PartitionPolicy):
        raise CaptureError("custom partition object must implement assign_stages()")
    executable = _executable_nodes(graph_module)
    supplied = _invoke_custom_policy(policy, graph_module, module)
    _validate_custom_coverage(executable, supplied)
    return _normalize_custom_assignments(executable, supplied), ()


def _resolve_builtin_policy(
    graph_module: GraphModule,
    module: nn.Module,
    policy: str,
) -> tuple[dict[Node, int], tuple[str, ...]]:
    if policy not in {"auto", "whole"}:
        raise CaptureError("partition must be 'auto', 'whole', or a PartitionPolicy")
    repeated = _outer_repeated_groups(module) if policy == "auto" else ()
    return _automatic_partition_assignments(graph_module, repeated), repeated


def _executable_nodes(graph_module: GraphModule) -> tuple[Node, ...]:
    return tuple(
        node
        for node in graph_module.graph.nodes
        if node.op not in {"placeholder", "output", "get_attr"}
    )


def _invoke_custom_policy(
    policy: PartitionPolicy,
    graph_module: GraphModule,
    module: nn.Module,
) -> Mapping[str, int]:
    graph_before = str(graph_module.graph)
    try:
        raw = policy.assign_stages(graph_module, module)
    except BaseException as exc:
        if isinstance(exc, CaptureError):
            raise
        raise CaptureError(f"custom partition policy failed: {exc}") from exc
    if str(graph_module.graph) != graph_before:
        raise CaptureError("custom partition policy must not mutate the FX graph")
    if not isinstance(raw, Mapping):
        raise CaptureError("custom partition policy must return a node-name mapping")
    if any(not isinstance(name, str) for name in raw):
        raise CaptureError("custom partition policy keys must be FX node names")
    return raw


def _validate_custom_coverage(
    executable: tuple[Node, ...],
    supplied: Mapping[str, int],
) -> None:
    expected = {node.name for node in executable}
    supplied_names = set(supplied)
    missing = tuple(sorted(expected - supplied_names))
    extra = tuple(sorted(supplied_names - expected))
    if missing or extra:
        raise CaptureError(
            "custom partition coverage differs from executable FX nodes: "
            f"missing={missing}, extra={extra}"
        )


def _normalize_custom_assignments(
    executable: tuple[Node, ...],
    supplied: Mapping[str, int],
) -> dict[Node, int]:
    normalized: dict[int, int] = {}
    closed: set[int] = set()
    previous: int | None = None
    result: dict[Node, int] = {}
    for node in executable:
        label = supplied[node.name]
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise CaptureError(
                "custom partition label for node "
                f"{node.name!r} must be a nonnegative integer"
            )
        if label != previous:
            if label in closed:
                raise CaptureError(
                    f"custom partition label {label} is not topologically contiguous"
                )
            if previous is not None:
                closed.add(previous)
            normalized.setdefault(label, len(normalized))
            previous = label
        result[node] = normalized[label]
    if not result:
        raise CaptureError("export graph has no executable operations")
    return result


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


def _automatic_partition_assignments(
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


__all__ = ["PartitionPolicy", "PartitionSpec", "resolve_partition_assignments"]
