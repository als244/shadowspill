"""The forward stages exported once and partitioned on fake tensors."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export.graph_signature import InputKind
from torch.utils._pytree import TreeSpec, tree_flatten

from shadowspill.errors import (
    CaptureError,
    PlanningError,
)
from shadowspill.pytorch.capture.aot import ExportCapture, capture_forward
from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    capture_forward_stage_artifacts,
)
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.profiling.metadata import (
    ProfilingMetadata,
    canonicalize_profiling_metadata,
)
from shadowspill.runtime.plan import PlanMemory

from ...guards import InputSignature, capture_input_signature
from ...materialization import (
    flat_runtime_arguments,
    representative_cpu_inputs,
)
from ...partition import (
    PartitionedExport,
    PartitionSpec,
    partition_export,
)
from ...sharing import (
    ResolvedSharedInput,
    ResolvedSharedOutput,
    SharedOutput,
    resolve_shared_inputs,
    resolve_shared_outputs,
)
from ..artifacts import (
    ForwardCaptureArtifacts,
)
from ..common import (
    PlanningTimer,
    estimate_spill_reservation,
    validate_budgets,
    validate_cpu_model,
)
from ..stores import PlanningStores


def capture_forward_graph(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    memory: PlanMemory,
    partition: PartitionSpec,
    profiling_metadata: object,
    shared_outputs: Sequence[SharedOutput] = (),
    stores: PlanningStores,
    timer: PlanningTimer,
) -> ForwardCaptureArtifacts:
    """Validate and capture one forward graph without numerical CUDA execution."""

    with timer.measure("validation"):
        signature, cpu_inputs, workload, resolved_shared_inputs = (
            _prepare_forward_inputs(
                model,
                example_inputs,
                memory,
                profiling_metadata,
            )
        )
    with timer.measure("runtime_binding"):
        installed = memory.installed
        device_ordinal = memory.execution_device
    with timer.measure("capture_lowering"):
        (
            fake_model,
            capture,
            partitioned,
            tasks,
            output_tree_spec,
            resolved_shared_outputs,
        ) = _capture_partitioned_forward(
            model,
            cpu_inputs,
            device_ordinal=device_ordinal,
            partition=partition,
            stores=stores,
            timer=timer,
            shared_outputs=shared_outputs,
            pool_names=tuple(memory.runtime.pools),
        )
    return ForwardCaptureArtifacts(
        signature,
        cpu_inputs,
        workload,
        installed,
        device_ordinal,
        fake_model,
        capture,
        partitioned,
        tasks,
        output_tree_spec,
        _resolve_shared_input_roots(capture, resolved_shared_inputs),
        resolved_shared_outputs,
    )


def _prepare_forward_inputs(
    model: nn.Module,
    example_inputs: Sequence[Any],
    memory: PlanMemory,
    profiling_metadata: object,
) -> tuple[
    InputSignature,
    tuple[object, ...],
    ProfilingMetadata,
    tuple[ResolvedSharedInput, ...],
]:
    validate_cpu_model(model)
    validate_budgets(memory.execution_budget, memory.spill_budget)
    if not isinstance(example_inputs, list | tuple):
        raise PlanningError("example_inputs must be a list or tuple")
    resolved_inputs, shared_inputs = resolve_shared_inputs(
        example_inputs,
        pool_names=tuple(memory.runtime.pools),
        runtime=memory.runtime,
    )
    representative_inputs = representative_cpu_inputs(resolved_inputs)
    signature = capture_input_signature(representative_inputs)
    cpu_inputs = tuple(representative_inputs)
    estimate_spill_reservation(model, cpu_inputs, memory.spill_budget)
    return (
        signature,
        cpu_inputs,
        canonicalize_profiling_metadata(profiling_metadata),
        shared_inputs,
    )


def _resolve_shared_input_roots(
    capture: ExportCapture,
    shared_inputs: tuple[ResolvedSharedInput, ...],
) -> tuple[ResolvedSharedInput, ...]:
    """Map public input leaves to Export's explicit root-input positions."""

    input_specs = capture.exported_program.graph_signature.input_specs
    user_positions = tuple(
        index
        for index, spec in enumerate(input_specs)
        if spec.kind is InputKind.USER_INPUT
    )
    result: list[ResolvedSharedInput] = []
    for item in shared_inputs:
        if item.public_leaf_index >= len(user_positions):
            raise CaptureError(
                "shared input leaf has no corresponding Export user input: "
                f"leaf={item.public_leaf_index}, user_inputs={len(user_positions)}"
            )
        result.append(
            replace(item, root_input_index=user_positions[item.public_leaf_index])
        )
    return tuple(result)


def _capture_partitioned_forward(
    model: nn.Module,
    cpu_inputs: tuple[object, ...],
    *,
    device_ordinal: int,
    partition: PartitionSpec,
    stores: PlanningStores,
    timer: PlanningTimer,
    shared_outputs: Sequence[SharedOutput],
    pool_names: tuple[str, ...],
) -> tuple[
    nn.Module,
    ExportCapture,
    PartitionedExport,
    tuple[GraphArtifact, ...],
    TreeSpec,
    tuple[ResolvedSharedOutput, ...],
]:
    try:
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        fake_model = fake_device_model(model, fake_mode, device_index=device_ordinal)
        fake_inputs = fake_device_inputs(
            cpu_inputs,
            fake_mode,
            device_index=device_ordinal,
        )
        with fake_mode, torch.no_grad():
            public_output = fake_model(*fake_inputs)
            output_leaves, output_tree_spec = tree_flatten(public_output)
            resolved_shared_outputs = resolve_shared_outputs(
                public_output,
                shared_outputs,
                pool_names=pool_names,
            )
            del output_leaves
            capture = capture_forward(fake_model, fake_inputs)
        with timer.measure("export_archival"):
            stores.archive_export(capture, mode="forward", position=0)
        representative_roots = tuple(
            value.detach() if isinstance(value, torch.Tensor) else value
            for value in flat_runtime_arguments(capture, model, cpu_inputs)
        )
        with fake_mode, torch.no_grad():
            partitioned = partition_export(
                capture,
                fake_model,
                partition=partition,
                representative_root_inputs=representative_roots,
            )
            tasks = capture_forward_stage_artifacts(partitioned)
    except CaptureError:
        raise
    except BaseException as error:
        raise CaptureError(f"forward graph capture failed: {error}") from error
    return (
        fake_model,
        capture,
        partitioned,
        tasks,
        output_tree_spec,
        resolved_shared_outputs,
    )
