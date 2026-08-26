"""Project Export provenance, mutations, and outputs onto partition stages."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import ExportCapture
from shadowspill.pytorch.capture.artifacts import (
    TaskInputProvenance,
    TaskInputRole,
    TensorGeometry,
)

from ..capture.storage import ExplicitMutation
from .artifacts import Stage, StageExample, StageValueSource
from .sources import stage_value_source
from .split import SplitExportGraph
from .values import StageOutputKey, derive_authentic_control_values


def root_input_provenance(
    capture: ExportCapture,
    *,
    representative_root_inputs: tuple[object, ...] | None,
) -> tuple[TaskInputProvenance, ...]:
    """Translate Export input kinds into task-local value-policy roles."""

    inputs = capture.flat_inputs
    if representative_root_inputs is not None and len(
        representative_root_inputs
    ) != len(inputs):
        raise CaptureError("representative root inputs differ from the Export contract")
    specs = tuple(capture.exported_program.graph_signature.input_specs)
    if len(specs) != len(inputs):
        raise CaptureError("Export input signature differs from flattened inputs")
    role_by_kind = {
        InputKind.PARAMETER: TaskInputRole.PARAMETER,
        InputKind.BUFFER: TaskInputRole.BUFFER,
        InputKind.CONSTANT_TENSOR: TaskInputRole.CONSTANT,
        InputKind.USER_INPUT: TaskInputRole.USER_INPUT,
    }
    result: list[TaskInputProvenance] = []
    for index, (spec, value) in enumerate(zip(specs, inputs, strict=True)):
        role = role_by_kind.get(spec.kind, TaskInputRole.USER_INPUT)
        if (
            role is TaskInputRole.USER_INPUT
            and isinstance(value, torch.Tensor)
            and _requires_authentic_value(value)
        ):
            role = TaskInputRole.CONTROL
        source = spec.target if isinstance(spec.target, str) else f"input_{index}"
        reference = _representative_root_tensor(
            value,
            None
            if representative_root_inputs is None
            else representative_root_inputs[index],
        )
        result.append(TaskInputProvenance(role, source, representative_value=reference))
    return tuple(result)


def build_stage_examples(
    capture: ExportCapture,
    split: SplitExportGraph,
    root_provenance: tuple[TaskInputProvenance, ...],
    *,
    representative_root_inputs: tuple[object, ...] | None = None,
    caller_stage_values: Mapping[StageOutputKey, torch.Tensor] | None = None,
) -> tuple[StageExample, ...]:
    """Build occurrence-local stage contracts from one split root graph."""

    mutations = _partition_mutations(capture, split)
    user_outputs = _partition_user_outputs(capture, split)
    control_values = derive_authentic_control_values(
        split,
        representative_root_inputs,
        caller_values=caller_stage_values,
    )
    examples: list[StageExample] = []
    for index, record in enumerate(split.stages):
        provenance: list[TaskInputProvenance] = []
        for input_position, source in enumerate(record.input_sources):
            if source is None:
                provenance.append(TaskInputProvenance(TaskInputRole.USER_INPUT))
                continue
            item = _stage_input_provenance(
                source,
                value=record.inputs[input_position],
                roots=root_provenance,
                control_values=control_values,
            )
            provenance.append(item)
        examples.append(
            StageExample(
                stage=Stage(
                    stage_id=f"stage_{index:04d}",
                    module_target=record.module_target,
                    graph_module=record.graph_module,
                    input_sources=record.input_sources,
                    input_provenance=tuple(provenance),
                    mutations=mutations.get(index, ()),
                    user_output_indices=tuple(
                        local for _public, local in user_outputs.get(index, ())
                    ),
                    public_output_bindings=user_outputs.get(index, ()),
                ),
                inputs=record.inputs,
                output=record.output,
            )
        )
    return tuple(examples)


def _representative_root_tensor(
    captured: object,
    supplied: object | None,
) -> torch.Tensor | None:
    if supplied is None or not isinstance(captured, torch.Tensor):
        return None
    if not isinstance(supplied, torch.Tensor):
        raise CaptureError("representative root tensor became a static value")
    expected = TensorGeometry.from_tensor(captured)
    actual = TensorGeometry.from_tensor(supplied)
    if (
        expected.shape,
        expected.stride,
        expected.storage_offset,
        expected.dtype,
    ) != (
        actual.shape,
        actual.stride,
        actual.storage_offset,
        actual.dtype,
    ):
        raise CaptureError("representative root tensor geometry differs from Export")
    return supplied


def _partition_user_outputs(
    capture: ExportCapture,
    split: SplitExportGraph,
) -> dict[int, tuple[tuple[int, int], ...]]:
    output_leaves = _root_output_leaves(split.root)
    result: dict[int, list[tuple[int, int]]] = {}
    for public_index, output_index in enumerate(capture.user_output_indices):
        try:
            root_output = output_leaves[output_index]
        except IndexError as exc:
            raise CaptureError("Export user output is absent from split root") from exc
        source = stage_value_source(
            root_output,
            placeholder_index=split.placeholder_index,
            stage_index_by_node=split.stage_index_by_node,
        )
        if source.producer_stage_index is None or source.producer_output_index is None:
            raise CaptureError("Export user output is not stage-produced")
        result.setdefault(source.producer_stage_index, []).append(
            (public_index, source.producer_output_index)
        )
    return {index: tuple(values) for index, values in result.items()}


def _partition_mutations(
    capture: ExportCapture,
    split: SplitExportGraph,
) -> dict[int, tuple[ExplicitMutation, ...]]:
    output_leaves = _root_output_leaves(split.root)
    result: dict[int, list[ExplicitMutation]] = {}
    for mutation in capture.mutations:
        try:
            root_output = output_leaves[mutation.output_index]
        except IndexError as exc:
            raise CaptureError(
                "Export mutation output is absent from split root"
            ) from exc
        source = stage_value_source(
            root_output,
            placeholder_index=split.placeholder_index,
            stage_index_by_node=split.stage_index_by_node,
        )
        if source.producer_stage_index is None or source.producer_output_index is None:
            raise CaptureError("Export mutation replacement is not stage-produced")
        stage_index = source.producer_stage_index
        candidates = tuple(
            position
            for position, input_source in enumerate(
                split.stages[stage_index].input_sources
            )
            if input_source is not None
            and input_source.root_input_index == mutation.input_index
        )
        if len(candidates) != 1:
            raise CaptureError(
                "Export mutation target does not resolve to one producer-stage input: "
                f"stage={stage_index}, target={mutation.target!r}, "
                f"root_input={mutation.input_index}, candidates={candidates}"
            )
        result.setdefault(stage_index, []).append(
            ExplicitMutation(
                candidates[0], source.producer_output_index, mutation.target
            )
        )
    return {index: tuple(values) for index, values in result.items()}


def _root_output_leaves(root: GraphModule) -> list[object]:
    output_node = next(node for node in root.graph.nodes if node.op == "output")
    leaves, _ = tree_flatten(output_node.args[0])
    return leaves


def _stage_input_provenance(
    source: StageValueSource,
    *,
    value: object,
    roots: tuple[TaskInputProvenance, ...],
    control_values: dict[StageOutputKey, torch.Tensor],
) -> TaskInputProvenance:
    if source.root_input_index is not None:
        try:
            return roots[source.root_input_index]
        except IndexError as exc:
            raise CaptureError(
                "stage input provenance is outside the root contract"
            ) from exc
    assert source.producer_stage_index is not None
    assert source.producer_output_index is not None
    key = (source.producer_stage_index, source.producer_output_index)
    control = isinstance(value, torch.Tensor) and _requires_authentic_value(value)
    return TaskInputProvenance(
        TaskInputRole.CONTROL if control else TaskInputRole.ACTIVATION,
        (
            f"stage_{source.producer_stage_index:04d}."
            f"output_{source.producer_output_index:04d}"
        ),
        representative_value=control_values.get(key),
    )


def _requires_authentic_value(value: torch.Tensor) -> bool:
    return not value.is_floating_point() and not value.is_complex()


__all__ = ["build_stage_examples", "root_input_provenance"]
