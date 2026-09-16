"""Capture: the objective exported once per input structure, partitioned per position,
and the storage layout lowered from it -- entirely offline, on fake tensors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pipeline.common import (
    PlanningTimer,
    validate_budgets,
)
from shadowspill.profiling.metadata import (
    ProfilingMetadata,
    repeated_profiling_metadata,
)
from shadowspill.pytorch.capture.aot import (
    TrainingObjectiveCapture,
    capture_training_objective,
    rebind_training_objective,
)
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.materialization.training import (
    representative_training_arguments,
)
from shadowspill.pytorch.planning.common import (
    estimate_spill_reservation,
    validate_cpu_model,
)
from shadowspill.runtime.plan import PlanMemory

from ...contracts import (
    ObjectiveResult,
)
from ...graph_pairs import (
    PartitionedTrainingCapture,
    partition_training_capture,
)
from ...guards import InputSignature, capture_training_signatures
from ...lowering.training import (
    lower_training_storage_layout,
)
from ...materialization import representative_cpu_inputs
from ...partition import (
    PartitionSpec,
)
from ..artifacts import (
    TrainingCaptureArtifacts,
)
from ..stores import PlanningStores


def capture_training_graphs(
    model: nn.Module,
    *,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    build_optimizer: Callable[[Any], torch.optim.Optimizer],
    example_inputs: Sequence[Sequence[Any]],
    memory: PlanMemory,
    partition: PartitionSpec,
    profiling_metadata: Sequence[object] | None,
    stores: PlanningStores,
    timer: PlanningTimer,
) -> TrainingCaptureArtifacts:
    """Capture objective and stage-local graph pairs entirely offline."""

    with timer.measure("validation"):
        signatures, cpu_inputs, workloads = _prepare_training_inputs(
            model,
            objective,
            build_optimizer,
            example_inputs,
            memory,
            profiling_metadata,
        )
    with timer.measure("runtime_binding"):
        installed = memory.installed
        device_ordinal = memory.execution_device
    with timer.measure("capture_lowering"):
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_model = fake_device_model(model, fake_mode, device_index=device_ordinal)
        captures = _capture_training_objectives(
            fake_model,
            objective,
            cpu_inputs,
            signatures=signatures,
            fake_mode=fake_mode,
            device_ordinal=device_ordinal,
            stores=stores,
            timer=timer,
        )
        partitioned = _partition_training_graphs(
            model,
            captures,
            cpu_inputs,
            fake_mode=fake_mode,
            partition=partition,
            stores=stores,
            timer=timer,
        )
        with timer.measure("storage_layout_lowering"):
            layout = lower_training_storage_layout(fake_model, captures)
    return TrainingCaptureArtifacts(
        signatures,
        cpu_inputs,
        workloads,
        installed,
        device_ordinal,
        fake_model,
        captures,
        partitioned,
        layout,
    )


def _prepare_training_inputs(
    model: nn.Module,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    build_optimizer: Callable[[Any], torch.optim.Optimizer],
    example_inputs: Sequence[Sequence[Any]],
    memory: PlanMemory,
    profiling_metadata: Sequence[object] | None,
) -> tuple[
    tuple[InputSignature, ...],
    tuple[tuple[object, ...], ...],
    tuple[ProfilingMetadata, ...],
]:
    validate_cpu_model(model)
    validate_budgets(memory.execution_budget, memory.spill_budget)
    if not callable(objective):
        raise TypeError("objective must be callable")
    if not callable(build_optimizer):
        raise TypeError(
            "optimizer must be callable: it is given the model's parameters "
            "and returns a torch.optim.Optimizer"
        )
    signatures = capture_training_signatures(example_inputs)
    cpu_inputs = tuple(
        tuple(representative_cpu_inputs(microbatch)) for microbatch in example_inputs
    )
    estimate_spill_reservation(model, cpu_inputs, memory.spill_budget)
    workloads = repeated_profiling_metadata(
        profiling_metadata,
        repetitions=len(example_inputs),
    )
    return signatures, cpu_inputs, workloads


def _capture_training_objectives(
    fake_model: nn.Module,
    objective: Callable[..., torch.Tensor | ObjectiveResult],
    cpu_inputs: tuple[tuple[object, ...], ...],
    *,
    signatures: tuple[InputSignature, ...],
    fake_mode: FakeTensorMode,
    device_ordinal: int,
    stores: PlanningStores,
    timer: PlanningTimer,
) -> tuple[TrainingObjectiveCapture, ...]:
    """Export the objective once per distinct input structure.

    Positions that share an input signature share the exported program and
    keep their own flattened example inputs, so every per-position
    registration downstream is unchanged. Exporting per position would cost
    one full export and one resident program per accumulation step.
    """

    templates: dict[str, TrainingObjectiveCapture] = {}
    captures: list[TrainingObjectiveCapture] = []
    with fake_mode, timer.measure("objective_export"):
        for microbatch, signature in zip(cpu_inputs, signatures, strict=True):
            inputs = fake_device_inputs(
                microbatch,
                fake_mode,
                device_index=device_ordinal,
            )
            template = templates.get(signature.digest)
            if template is None:
                template = capture_training_objective(fake_model, objective, inputs)
                templates[signature.digest] = template
                captures.append(template)
            else:
                captures.append(rebind_training_objective(template, inputs))
    with timer.measure("export_archival"):
        for position, capture in enumerate(captures):
            if any(capture is template for template in templates.values()):
                stores.archive_export(
                    capture.exported,
                    mode="training_objective",
                    position=position,
                )
    return tuple(captures)


def _partition_training_graphs(
    model: nn.Module,
    captures: tuple[TrainingObjectiveCapture, ...],
    cpu_inputs: tuple[tuple[object, ...], ...],
    *,
    fake_mode: FakeTensorMode,
    partition: PartitionSpec,
    stores: PlanningStores,
    timer: PlanningTimer,
) -> tuple[PartitionedTrainingCapture, ...]:
    """Partition every position against its own example inputs.

    Positions may share an exported program, but a partition binds stage
    inputs to the example tensors it was built from, and the lowering
    registers each position's input objects from those tensors. Sharing a
    partition would make later positions read the first position's inputs.
    """

    representative_roots = tuple(
        representative_training_arguments(capture, model, microbatch)
        for capture, microbatch in zip(captures, cpu_inputs, strict=True)
    )
    with fake_mode, timer.measure("stage_partition_aot"):
        return tuple(
            partition_training_capture(
                capture,
                partition=partition,
                graph_pair_store=stores.graph_pairs,
                representative_root_inputs=root_inputs,
                # Which microbatch creates a stage's gradient and which add
                # into it is decided by the step's ordering, after capture,
                # so every microbatch of an accumulating step carries both
                # forms; the store derives each contract's once.
                accumulating=len(captures) > 1,
            )
            for capture, root_inputs in zip(captures, representative_roots, strict=True)
        )
