"""Public orchestration for partitioning one exported PyTorch graph."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from shadowspill.pytorch.capture.aot import ExportCapture

from .artifacts import PartitionedExport
from .policy import PartitionSpec, resolve_partition_assignments
from .provenance import build_stage_examples, root_input_provenance
from .split import split_export_graph
from .values import StageOutputKey


def partition_export(
    capture: ExportCapture,
    module: nn.Module,
    *,
    partition: PartitionSpec = "auto",
    representative_root_inputs: tuple[object, ...] | None = None,
    representative_stage_outputs: Mapping[StageOutputKey, torch.Tensor] | None = None,
) -> PartitionedExport:
    """Split one Export graph according to a built-in or custom policy.

    ``representative_stage_outputs`` may supply authentic integer/boolean
    values when their producer dependency slice cannot run on the planning
    host. Geometry-only synthetic control values are never accepted.
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


__all__ = ["partition_export"]
