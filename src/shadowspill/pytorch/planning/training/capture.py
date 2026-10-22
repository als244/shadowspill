"""Capture: the objective exported once per input structure, partitioned per position,
and the storage layout lowered from it -- entirely offline, on fake tensors."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
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
from shadowspill.pytorch.capture.artifacts import TensorGeometry
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.capture.retention import (
    MEMORY_BOUND_FLOPS_PER_BYTE,
    RetentionPolicy,
)
from shadowspill.pytorch.materialization.training import (
    representative_training_arguments,
)
from shadowspill.pytorch.planning.common import (
    estimate_spill_reservation,
    validate_cpu_model,
)
from shadowspill.pytorch.prepared import PreparedStep
from shadowspill.pytorch.state.initialization import pool_values
from shadowspill.runtime.plan import PlanMemory

from ...contracts import (
    ObjectiveResult,
)
from ...graph_pairs import (
    GRAPH_PAIR_VARIANTS,
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
from ...partition.artifacts import PartitionedExport
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
    grad_dtype: torch.dtype | None = None,
    fuse_narrow_gradient_accumulation: bool = True,
    memory_bound_flops_per_byte: float = MEMORY_BOUND_FLOPS_PER_BYTE,
    graph_pair_variants: tuple[str, ...] = GRAPH_PAIR_VARIANTS,
    prepared_step: PreparedStep | None = None,
) -> TrainingCaptureArtifacts:
    """Capture objective and stage-local graph pairs entirely offline, their
    parameter gradients at ``grad_dtype`` when one is given, their
    accumulating forms rounding once where they may with
    ``fuse_narrow_gradient_accumulation``, their ``save`` variants regenerating
    every operator at or under ``memory_bound_flops_per_byte``, and only
    the ``graph_pair_variants`` named built at all."""

    retention = RetentionPolicy(memory_bound_flops_per_byte)
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
        if prepared_step is None:
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
        else:
            with timer.measure("prepared_binding"):
                prepared_step.validate(
                    model,
                    [item.digest for item in signatures],
                    capture_options(
                        partition,
                        grad_dtype,
                        fuse_narrow_gradient_accumulation,
                        memory_bound_flops_per_byte,
                        graph_pair_variants,
                    ),
                )
                captures = prepared_step.bind(
                    fake_model,
                    cpu_inputs,
                    fake_mode=fake_mode,
                    device_ordinal=device_ordinal,
                )
        partitioned = _partition_training_graphs(
            model,
            captures,
            cpu_inputs,
            fake_mode=fake_mode,
            partition=partition,
            stores=stores,
            timer=timer,
            retention=retention,
            grad_dtype=grad_dtype,
            fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
            graph_pair_variants=graph_pair_variants,
            values=lambda: pool_values(memory.runtime),
        )
        with timer.measure("storage_layout_lowering"):
            layout = lower_training_storage_layout(
                fake_model, captures, device_ordinal=device_ordinal
            )
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
        retention,
    )


def capture_options(
    partition: PartitionSpec,
    grad_dtype: torch.dtype | None,
    fuse_narrow_gradient_accumulation: bool,
    memory_bound_flops_per_byte: float,
    graph_pair_variants: tuple[str, ...],
) -> dict[str, object]:
    """Budget-independent options that define the prepared objective graphs."""
    return {
        "partition": partition if isinstance(partition, str) else repr(partition),
        "grad_dtype": None if grad_dtype is None else str(grad_dtype),
        "fuse_narrow_gradient_accumulation": fuse_narrow_gradient_accumulation,
        "memory_bound_flops_per_byte": memory_bound_flops_per_byte,
        "graph_pair_variants": graph_pair_variants,
    }


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
    estimate_spill_reservation(
        model, cpu_inputs, memory.spill_budget, runtime=memory.runtime
    )
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
    retention: RetentionPolicy,
    grad_dtype: torch.dtype | None,
    fuse_narrow_gradient_accumulation: bool,
    graph_pair_variants: tuple[str, ...] = GRAPH_PAIR_VARIANTS,
    values: Callable[[], AbstractContextManager[object]] = nullcontext,
) -> tuple[PartitionedTrainingCapture, ...]:
    """Partition every position against its own example inputs.

    Positions may share an exported program, but a partition binds stage
    inputs to the example tensors it was built from, and the lowering
    registers each position's input objects from those tensors. Sharing a
    partition would make later positions read the first position's inputs.

    What positions of one exported program do share is their authentic
    control values. Deriving them runs every producer slice for real, from
    the first stage to the last router, so they are derived once per
    program rather than once per position; each position checks them
    against its own geometry, and profiling measures one occurrence per
    contract whichever position it came from.

    They also share the split. The stage policy, the split root and the
    folds are functions of the exported program, and splitting is the cost
    that grows with the graph, so a program is split once: each later
    position whose inputs have the first one's geometry takes the first
    one's stages with their values mapped onto its own inputs, storage for
    storage, and runs the first one's split root only where a value cannot
    be mapped. A position whose inputs differ is split for itself. Nothing
    here decides which microbatch creates a gradient and which adds into
    one; every position still carries both backward forms.

    ``values`` reads state a pool holds for the operations that need it,
    which are those deriving control values: they run producer slices on
    authentic values. Only a program's first position derives them, so only
    it runs under ``values``; every other position works on fake tensors and
    is not slowed by a hook on each of its operations.
    """

    representative_roots = tuple(
        representative_training_arguments(capture, model, microbatch)
        for capture, microbatch in zip(captures, cpu_inputs, strict=True)
    )
    with timer.measure("stage_partition_aot"):
        derived: dict[int, Mapping[tuple[int, int], torch.Tensor]] = {}
        splits: dict[int, PartitionedExport] = {}
        spent: dict[int, list[int]] = {}
        partitioned: list[PartitionedTrainingCapture] = []
        for capture, root_inputs in zip(captures, representative_roots, strict=True):
            # Positions rebound from one template share its exported program.
            template = id(capture.exported.exported_program)
            shared = splits.get(template)
            if shared is not None and not _same_geometry(
                capture.exported.flat_inputs, shared.root_inputs
            ):
                shared = None
            # The reader sits under the fake mode, as it always has, and only
            # where control values are derived.
            reader = values() if template not in derived else nullcontext()
            started = time.perf_counter_ns()
            with reader, fake_mode:
                result = partition_training_capture(
                    capture,
                    partition=partition,
                    graph_pair_store=stores.graph_pairs,
                    representative_root_inputs=root_inputs,
                    representative_stage_outputs=derived.get(template),
                    retention=retention,
                    # Which microbatch creates a stage's gradient and which
                    # add into it is decided by the step's ordering, after
                    # capture, so every microbatch of an accumulating step
                    # carries both forms; the store derives each contract's
                    # once.
                    accumulating=len(captures) > 1,
                    gradient_dtype=grad_dtype,
                    fuse_narrow_gradient_accumulation=fuse_narrow_gradient_accumulation,
                    graph_pair_variants=graph_pair_variants,
                    template=shared,
                )
            spent.setdefault(template, []).append(time.perf_counter_ns() - started)
            derived.setdefault(template, result.partitioned.control_values)
            splits.setdefault(template, result.partitioned)
            partitioned.append(result)
        # A program's first position pays for everything done once: deriving
        # control values, building graph pairs, splitting. What scales with
        # the step is each later position, so the two are reported apart.
        for index, times in enumerate(spent.values(), 1):
            later = times[1:]
            timer.progress(
                f"stage partition of program {index}/{len(spent)}:"
                f" {len(times)} positions, first {times[0] / 1e9:.1f} s"
                + (
                    f", later {sum(later) / len(later) / 1e9:.2f} s each"
                    if later
                    else ""
                )
            )
        return tuple(partitioned)


def _same_geometry(inputs: tuple[object, ...], template: tuple[object, ...]) -> bool:
    """Whether a position's root inputs are laid out as the template's were:
    the same tensors' geometry, and the same values where they are static."""

    if len(inputs) != len(template):
        return False
    for value, reference in zip(inputs, template, strict=True):
        if isinstance(value, torch.Tensor) and isinstance(reference, torch.Tensor):
            if TensorGeometry.from_tensor(value) != TensorGeometry.from_tensor(
                reference
            ):
                return False
        elif (
            isinstance(value, torch.Tensor)
            or isinstance(reference, torch.Tensor)
            or value != reference
        ):
            return False
    return True
