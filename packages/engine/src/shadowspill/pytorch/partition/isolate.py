"""Compose ordinary stage boundaries with explicit operator barriers."""

from __future__ import annotations

import operator
from dataclasses import dataclass

from torch import nn
from torch.fx import GraphModule

from .policy import PartitionSpec, resolve_partition_assignments


@dataclass(frozen=True, slots=True)
class IsolateOperations:
    """Give each selected operator and its tuple unpacking a separate stage.

    ``targets`` contains exact operator overload names. ``base`` supplies the
    surrounding stage boundaries. Overlapping unpacking intervals are merged;
    unrelated nodes inside an interval stay with it to preserve graph order.
    """

    targets: tuple[str, ...]
    base: PartitionSpec = "auto"

    def assign_stages(
        self, graph_module: GraphModule, module: nn.Module
    ) -> dict[str, int]:
        assignments, _ = resolve_partition_assignments(graph_module, module, self.base)
        nodes = tuple(assignments)
        positions = {node: i for i, node in enumerate(nodes)}
        intervals: list[tuple[int, int]] = []
        for node in nodes:
            if str(node.target) not in self.targets:
                continue
            closure = {node}
            queue = [node]
            while queue:
                for user in queue.pop().users:
                    if user.target is operator.getitem and user not in closure:
                        closure.add(user)
                        queue.append(user)
            start, stop = positions[node], max(positions[x] for x in closure)
            if intervals and start <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(stop, intervals[-1][1]))
            else:
                intervals.append((start, stop))
        result = {}
        previous = None
        stage = -1
        for position, node in enumerate(nodes):
            region = next(
                (i for i, (a, b) in enumerate(intervals) if a <= position <= b), None
            )
            key = (
                ("isolate", region)
                if region is not None
                else ("base", assignments[node])
            )
            if key != previous:
                stage += 1
                previous = key
            result[node.name] = stage
        return result
