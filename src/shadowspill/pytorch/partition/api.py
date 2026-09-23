"""Public orchestration for partitioning one exported PyTorch graph."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.fx import Node

from shadowspill.pytorch.capture.aot import ExportCapture

from .artifacts import PartitionedExport, StageRecord
from .computation import computes_nothing
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
    control values has no backward to take. A stage with no kernel to
    generate is folded whatever the caller, because neither path can compile
    one.
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
    split = _fold_unproductive_stages(
        capture,
        assignments,
        split,
        control_only=fold_control_only_stages,
    )
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


def _fold_unproductive_stages(
    capture: ExportCapture,
    assignments: dict[Node, int],
    split: SplitExportGraph,
    *,
    control_only: bool,
) -> SplitExportGraph:
    """Merge away a boundary drawn where there is nothing for a stage to do.

    Two stages are unproductive, and a boundary around either costs a task
    and buys no memory.

    A stage with **no kernel to generate**, every operation of which only
    renames what it was given. A compiler asked to compile one exposes no
    root graph and the build fails. CLIP's text encoder opens with one: a
    `view` and an `alias`, because the partition cuts before an embedding
    table that the policy reads as a repeated group. Neither path can
    compile such a stage, so this is folded for every caller.

    A stage with **nothing to differentiate**, whose outputs are all integer
    or boolean. It holds no activation, has no gradient to produce and no
    cotangent to receive. GPT-2 has one: it computes its position indices at
    model scope before the first transformer block, so the automatic policy
    makes the prologue a single `arange`. Only a training caller asks for
    this, because only training differentiates a stage through its outputs.

    The fold repeats, because merging can leave the stage it merged into
    unproductive as well. A stage merges into the one after it; a trailing
    one merges with the stage before it instead, which is the same merge
    named from the other side.
    """

    while len(split.stages) > 1:
        target = next(
            (
                index
                for index, record in enumerate(split.stages)
                if _is_unproductive(record, control_only=control_only)
            ),
            None,
        )
        if target is None:
            break
        # The last stage has nothing after it to fold into, so the merge is
        # named from its predecessor and produces the same single stage.
        merged = min(target, len(split.stages) - 2)
        assignments = _merge_stage_forward(assignments, merged)
        split = split_export_graph(capture, assignments)
    return split


def _is_unproductive(record: StageRecord, *, control_only: bool) -> bool:
    if computes_nothing(record.graph_module):
        return True
    return control_only and not differentiable_output_positions(record.output)


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
