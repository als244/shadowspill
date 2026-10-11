"""The forward stages exported once and partitioned on fake tensors."""

import copy
from collections.abc import Callable, Sequence
from dataclasses import replace
from types import MethodType
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
from shadowspill.pipeline.common import (
    PlanningTimer,
    validate_budgets,
)
from shadowspill.profiling.metadata import (
    ProfilingMetadata,
    canonicalize_profiling_metadata,
)
from shadowspill.pytorch.capture.aot import ExportCapture, capture_forward
from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    capture_forward_stage_artifacts,
)
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.planning.common import (
    estimate_spill_reservation,
    validate_cpu_model,
)
from shadowspill.pytorch.representations import detached_representation
from shadowspill.pytorch.state.initialization import pool_values
from shadowspill.runtime import Runtime
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
from ..stores import PlanningStores


def capture_forward_graph(
    model: nn.Module,
    *,
    example_inputs: Sequence[Any],
    forward_fn: Callable[..., Any] | None = None,
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
            forward_fn=forward_fn,
            partition=partition,
            stores=stores,
            timer=timer,
            shared_outputs=shared_outputs,
            pool_names=tuple(memory.runtime.pools),
            runtime=memory.runtime,
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
    forward_fn: Callable[..., Any] | None,
    partition: PartitionSpec,
    stores: PlanningStores,
    timer: PlanningTimer,
    shared_outputs: Sequence[SharedOutput],
    pool_names: tuple[str, ...],
    runtime: Runtime,
) -> tuple[
    nn.Module,
    ExportCapture,
    PartitionedExport,
    tuple[GraphArtifact, ...],
    TreeSpec,
    tuple[ResolvedSharedOutput, ...],
]:
    try:
        with timer.measure("forward_export"):
            fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_model = fake_device_model(
                model, fake_mode, device_index=device_ordinal
            )
            forward_view = None
            if forward_fn is not None:
                forward_view = _select_forward(fake_model, forward_fn)
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
            if forward_view is not None:
                _restore_registered_module_paths(capture, fake_model, forward_view)
        with timer.measure("export_archival"):
            stores.archive_export(capture, mode="forward", position=0)
        with timer.measure("stage_partition_aot"):
            representative_roots = tuple(
                detached_representation(value)
                if isinstance(value, torch.Tensor)
                else value
                for value in flat_runtime_arguments(capture, model, cpu_inputs)
            )
            with pool_values(runtime), fake_mode, torch.no_grad():
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


def _select_forward(model: nn.Module, forward_fn: Callable[..., Any]) -> nn.Module:
    """Select only the capture copy's call, retaining its registered state names.

    The shallow view calls the model's original forward, so a callback can use
    model(...) without recursively calling itself. It shares the fake copy's
    registered tensors/modules; it never refers to the caller's real state.
    No live execution model is patched or independently copied here.
    """
    view = copy.copy(model)

    def call(_model: nn.Module, *inputs: Any) -> Any:
        return forward_fn(view, *inputs)

    model.forward = MethodType(call, model)
    return view


def _restore_registered_module_paths(
    capture: ExportCapture, model: nn.Module, view: nn.Module
) -> None:
    """Keep callback provenance in the source model's registered namespace.

    Strict Export can name a shared submodule through the callback closure
    instead of its registered path. Its stack key still identifies that same
    module object. Resolve those keys while the capture objects are alive so
    automatic and caller-defined partition policies see the usual layer paths.
    Tensor state names, graph operations and unregistered modules are unchanged.
    """
    paths = {str(id(module)): path for path, module in model.named_modules()}
    paths[str(id(view))] = ""
    for node in capture.exported_program.graph.nodes:
        stack = node.meta.get("nn_module_stack")
        if isinstance(stack, dict):
            node.meta["nn_module_stack"] = {
                key: (paths.get(key, path), module_type)
                for key, (path, module_type) in stack.items()
            }
