"""Carry declared coordinates through operations that execute as a whole.

Knowing where an output's coordinates lie does not select a chunked execution
for its producer. Unknown operations end propagation; their shape alone cannot
recover a flattened logical factor.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch.fx import Node
from torch.utils._pytree import tree_flatten, tree_map

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact

from .geometry import ChunkAxis
from .policy import OperatorRule
from .rules import infer


def merge_rules(
    known: dict[str, tuple[OperatorRule, ...]],
    domain: str,
    incoming: tuple[OperatorRule, ...],
) -> None:
    rules = {rule.target: rule for rule in known.get(domain, ())}
    for rule in incoming:
        previous = rules.get(rule.target)
        if previous is not None and previous != rule:
            raise CaptureError(
                f"conflicting operator contracts for chunk domain {domain!r}: "
                f"{rule.target}"
            )
        rules[rule.target] = rule
    known[domain] = tuple(rules[target] for target in sorted(rules))


def whole_output_domains(
    artifact: GraphArtifact,
    inputs: list[dict[str, ChunkAxis]],
    contracts: Mapping[str, tuple[OperatorRule, ...]],
) -> list[dict[str, ChunkAxis]]:
    result: list[dict[str, ChunkAxis]] = [{} for _ in range(artifact.output_count)]
    names = set().union(*(set(item) for item in inputs))
    for name in sorted(names):
        extents = set()
        for value, domains in zip(artifact.example_arguments, inputs, strict=True):
            axis = domains.get(name)
            if axis is None:
                continue
            assert isinstance(value, torch.Tensor)
            extent = int(value.shape[axis.dim])
            factor = axis.before * axis.after
            if extent % factor:
                raise CaptureError(
                    f"invalid inherited extent for chunk domain {name!r}"
                )
            extents.add(extent // factor)
        if len(extents) != 1:
            raise CaptureError(
                f"inconsistent inherited extents for chunk domain {name!r}"
            )
        rows = extents.pop()
        positions = iter(item.get(name) for item in inputs)
        layouts: dict[Node, Any] = {}
        rules = {rule.target: rule for rule in contracts.get(name, ())}
        for node in artifact.graph_module.graph.nodes:
            if node.op == "placeholder":
                layouts[node] = next(positions)
            elif node.op == "call_function":
                try:
                    layouts[node] = infer(node, layouts, rows, rules)
                except CaptureError:
                    if str(node.target) in rules:
                        raise
                    # Whole execution needs no independence proof. In particular,
                    # a sequence mixer without a rule cannot propagate an axis.
                    layouts[node] = tree_map(lambda _: None, node.meta.get("val"))
            elif node.op == "output":
                outputs, _ = tree_flatten(
                    tree_map(
                        lambda item, layouts=layouts: (
                            layouts[item] if isinstance(item, Node) else None
                        ),
                        node.args[0],
                    )
                )
                for leaf, layout in enumerate(outputs):
                    if isinstance(layout, ChunkAxis):
                        result[leaf][name] = layout
    return result


def inherit_nested_rules(
    artifact: GraphArtifact, known: dict[str, tuple[OperatorRule, ...]]
) -> None:
    if artifact.regions is not None:
        for domain, rules in artifact.regions.domain_rules.items():
            merge_rules(known, domain, rules)
    if artifact.chunking is not None:
        for _, child in (
            *artifact.chunking.children,
            *artifact.chunking.accumulating_children,
        ):
            inherit_nested_rules(child, known)
