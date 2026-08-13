"""Compose partitioned stages with structural AOT graph-pair portfolios."""

from __future__ import annotations

from collections.abc import Collection

from torch.export.graph_signature import InputKind, InputSpec

from ..aot import TrainingObjectiveCapture
from ..contracts import CaptureError
from ..partition import PartitionSpec, partition_export
from .artifacts import DifferentiatedStage, PartitionedTrainingCapture
from .capture import capture_training_stages
from .repository import GraphPairRepository


def training_parameter_stage_owners(
    captures: tuple[PartitionedTrainingCapture, ...],
    parameter_names: Collection[str],
) -> dict[str, tuple[int, ...]]:
    """Return the training stages whose backward passes contribute each parameter.

    Export makes parameters explicit root inputs.  Stage partitioning preserves
    that provenance in :class:`StageValueSource`, so optimizer grouping can use
    the same semantic stage boundaries without inspecting module-name patterns
    or runtime allocation behavior.
    """

    known = frozenset(parameter_names)
    owners: dict[str, set[int]] = {}
    expected_stage_count: int | None = None
    for capture in captures:
        if expected_stage_count is None:
            expected_stage_count = len(capture.stages)
        elif len(capture.stages) != expected_stage_count:
            raise CaptureError(
                "microbatch positions produced different training-stage counts"
            )
        input_specs = tuple(
            capture.training.exported.exported_program.graph_signature.input_specs
        )
        for stage_index, stage in enumerate(capture.stages):
            for name in _stage_parameter_names(stage, input_specs, known):
                owners.setdefault(name, set()).add(stage_index)
    return {name: tuple(sorted(indices)) for name, indices in owners.items()}


def _stage_parameter_names(
    stage: DifferentiatedStage,
    input_specs: tuple[InputSpec, ...],
    known: frozenset[str],
) -> tuple[str, ...]:
    result: list[str] = []
    for source in stage.example.stage.input_sources:
        if source is None or source.root_input_index is None:
            continue
        try:
            spec = input_specs[source.root_input_index]
        except IndexError as exc:
            raise CaptureError(
                "stage parameter provenance refers outside the Export ABI"
            ) from exc
        if spec.kind is not InputKind.PARAMETER:
            continue
        result.append(_optimizer_parameter_name(spec, known))
    return tuple(result)


def _optimizer_parameter_name(spec: InputSpec, known: frozenset[str]) -> str:
    target = spec.target
    if not isinstance(target, str) or not target.startswith("model."):
        raise CaptureError(
            "objective Export parameter target is not rooted at model: "
            f"{target!r}"
        )
    name = target.removeprefix("model.")
    if name not in known:
        raise CaptureError(
            f"stage parameter {name!r} is absent from the optimizer model"
        )
    return name


def partition_training_capture(
    capture: TrainingObjectiveCapture,
    *,
    partition: PartitionSpec = "auto",
    graph_pair_repository: GraphPairRepository | None = None,
    representative_root_inputs: tuple[object, ...] | None = None,
) -> PartitionedTrainingCapture:
    """Partition and differentiate one captured objective template."""

    partitioned = partition_export(
        capture.exported,
        capture.capture_module,
        partition=partition,
        representative_root_inputs=representative_root_inputs,
    )
    return PartitionedTrainingCapture(
        training=capture,
        partitioned=partitioned,
        stages=capture_training_stages(
            partitioned, graph_pair_repository=graph_pair_repository
        ),
    )


__all__ = [
    "partition_training_capture",
    "training_parameter_stage_owners",
]
