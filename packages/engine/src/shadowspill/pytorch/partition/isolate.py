"""Compose ordinary stage boundaries with explicit operator barriers."""

from __future__ import annotations

import operator
from dataclasses import dataclass

from torch import nn
from torch.fx import GraphModule

from .policy import PartitionSpec, resolve_partition_assignments


@dataclass(frozen=True, slots=True)
class IsolateOperations:
    """A `PartitionPolicy` that gives each call of selected operators its own stage.

    Each call of a listed operator forms one stage with the tuple extraction
    of its results, and with any other operations between the call and its
    last extraction, which keeps graph order. Calls whose stages would
    overlap share one. ``base`` divides the rest of the graph; a base stage a
    call interrupts continues as a new stage after it.

    Parameters
    ----------
    targets : tuple of str
        Exact operator overload names, such as ``"aten._softmax.default"``.
    base : {"auto", "whole"} or PartitionPolicy, default "auto"
        The partition of the rest of the graph.
    """

    targets: tuple[str, ...]
    base: PartitionSpec = "auto"

    def assign_stages(
        self, graph_module: GraphModule, module: nn.Module
    ) -> dict[str, int]:
        """Assign a stage label to every executable node of ``graph_module``.

        Parameters
        ----------
        graph_module : torch.fx.GraphModule
            The captured graph.
        module : torch.nn.Module
            The model the graph was captured from, passed on to ``base``.
            Under `ChunkRegions`, the task's graph module itself.

        Returns
        -------
        dict of str to int
            Every executable node's name mapped to its stage, numbered from
            zero in graph order.

        Raises
        ------
        CaptureError
            If ``base`` is not a partition, or its mapping is malformed.
        """
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
