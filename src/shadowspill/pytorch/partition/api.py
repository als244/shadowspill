"""Public orchestration for partitioning one exported PyTorch graph."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.fx import Node

from shadowspill.pytorch.capture.aot import ExportCapture

from .artifacts import PartitionedExport
from .differentiability import differentiable_output_positions
from .policy import PartitionSpec, resolve_partition_assignments
from .provenance import build_stage_examples, root_input_provenance
from .split import SplitExportGraph, split_export_graph
from .values import StageOutputKey


def partition_export(
    capture: ExportCapture,
    module: nn.Module,
    *,
    partition: PartitionSpec = "auto",
    representative_root_inputs: tuple[object, ...] | None = None,
    representative_stage_outputs: Mapping[StageOutputKey, torch.Tensor] | None = None,
    fold_control_only_stages: bool = False,
) -> PartitionedExport:
    """Split one Export graph according to a built-in or custom policy.

    ``representative_stage_outputs`` may supply authentic integer/boolean
    values when their producer dependency slice cannot run on the planning
    host. Geometry-only synthetic control values are never accepted.

    ``fold_control_only_stages`` is what a training caller asks for, because
    a stage is differentiated through its outputs and one that produces only
    control values has no backward to take.
    """

    assignments, repeated = resolve_partition_assignments(
        capture.exported_program.graph_module,
        module,
        partition,
    )
    provenance = root_input_provenance(
        capture,
        representative_root_inputs=representative_root_inputs,
    )
    split = split_export_graph(capture, assignments)
    if fold_control_only_stages:
        split = _fold_control_only_stages(capture, assignments, split)
    return PartitionedExport(
        root=split.root,
        root_inputs=capture.flat_inputs,
        root_input_provenance=provenance,
        stages=build_stage_examples(
            capture,
            split,
            provenance,
            representative_root_inputs=representative_root_inputs,
            caller_stage_values=representative_stage_outputs,
        ),
        repeated_groups=repeated,
        user_output_indices=capture.user_output_indices,
    )


def _fold_control_only_stages(
    capture: ExportCapture,
    assignments: dict[Node, int],
    split: SplitExportGraph,
) -> SplitExportGraph:
    """Merge a stage with nothing to differentiate into the one after it.

    A stage whose outputs are all integer or boolean holds no activation, has
    no gradient to produce and no cotangent to receive, so a boundary around
    it costs a task and buys no memory. It is setup for the stage that
    consumes it, and that is where it belongs.

    GPT-2 has one. It computes its position indices at model scope before the
    first transformer block, and the automatic policy makes everything before
    that block a prologue, so the prologue is one `arange`. Folding it is what
    lets the model train; refusing it described a graph the partition drew,
    not a model that cannot be trained.

    The fold repeats, because merging can leave the next stage control-only
    as well. A trailing control-only stage is left alone: there is nothing
    after it to fold into, and stage capture refuses it by name.
    """

    while True:
        target = next(
            (
                index
                for index, record in enumerate(split.stages[:-1])
                if not differentiable_output_positions(record.output)
            ),
            None,
        )
        if target is None:
            return split
        assignments = _merge_stage_forward(assignments, target)
        split = split_export_graph(capture, assignments)


def _merge_stage_forward(
    assignments: dict[Node, int], stage_index: int
) -> dict[Node, int]:
    """Give one stage's nodes the label of the stage after it, and renumber.

    Assignments are built by walking executable nodes in topological order, so
    the labels first appear in stage order and the mapping's own order is the
    stage order. Two adjacent intervals merge into one interval, so the result
    is still contiguous.
    """

    labels = list(dict.fromkeys(assignments.values()))
    source = labels[stage_index]
    destination = labels[stage_index + 1]
    merged = {
        node: destination if label == source else label
        for node, label in assignments.items()
    }
    renumbered = {
        label: index for index, label in enumerate(dict.fromkeys(merged.values()))
    }
    return {node: renumbered[label] for node, label in merged.items()}


__all__ = ["partition_export"]
