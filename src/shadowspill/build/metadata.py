"""Portable call descriptions and source names for built task artifacts."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind
from torch.fx import GraphModule
from torch.utils._pytree import treespec_dumps

from shadowspill.pytorch.capture.aot import ExportCapture
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.guards import InputSignature
from shadowspill.pytorch.lowering.catalog import RegistrationBinding
from shadowspill.pytorch.representations import tensor_components
from shadowspill.task.inputs import TaskInputRole


def model_tensor_geometry(
    model: nn.Module, registrations: tuple[RegistrationBinding, ...]
) -> dict[str, dict[str, Any]]:
    """Include unused registered tensors so another forward can share them."""
    tensors = {
        (name, path): component
        for name, value in (
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        )
        for path, component in tensor_components(value)
    }
    result = {}
    for item in registrations:
        value = tensors[item.name, item.component_path]
        result[item.object_id] = {
            "dtype": str(value.dtype),
            "shape": tuple(value.shape),
            "stride": tuple(value.stride()),
            "storage_offset": value.storage_offset(),
        }
    return result


def module_paths(graph: GraphModule) -> tuple[str, ...]:
    paths: dict[str, None] = {}
    for node in graph.graph.nodes:
        stack = node.meta.get("nn_module_stack", {})
        for path, _type in stack.values():
            if path:
                paths.setdefault(path, None)
    return tuple(paths)


def signature_record(signature: InputSignature) -> dict[str, Any]:
    return {
        "tree": treespec_dumps(signature.tree_spec),
        "leaves": [leaf.identity() for leaf in signature.leaves],
    }


def root_sources(exported: ExportCapture) -> list[dict[str, Any]]:
    result = []
    user_position = 0
    for index, spec in enumerate(exported.exported_program.graph_signature.input_specs):
        record = {
            "root_index": index,
            "kind": spec.kind.name,
            "name": spec.target or getattr(spec.arg, "name", ""),
        }
        if spec.kind is InputKind.USER_INPUT:
            record["user_leaf"] = user_position
            user_position += 1
        if spec.kind is InputKind.CONSTANT_TENSOR:
            value = exported.exported_program.constants.get(spec.target)
            if (
                isinstance(value, Tensor)
                and not isinstance(value, FakeTensor)
                and value.device.type == "cpu"
                and value.numel() <= 4096
            ):
                record["value"] = value.tolist()
        result.append(record)
    return result


def task_inputs(artifact: GraphArtifact) -> dict[str, Any]:
    return {
        "count": artifact.argument_count,
        "tensor_positions": artifact.tensor_argument_positions,
        "tensor_aliases": artifact.tensor_argument_alias_groups,
        "tensor_geometry": [item.identity() for item in artifact.tensor_inputs],
        "provenance": [_input_source(item) for item in artifact.input_provenance],
        "static_values": [
            [index, value]
            for index, value in enumerate(artifact.example_arguments)
            if not isinstance(value, Tensor)
        ],
    }


def _input_source(item: Any) -> dict[str, Any]:
    record = {
        "role": item.role.value,
        "source": item.source,
        "consumer_targets": item.consumer_targets,
    }
    value = item.representative_value
    if (
        item.role
        in {
            TaskInputRole.CONSTANT,
            TaskInputRole.CONTROL,
            TaskInputRole.OPTIMIZER_HYPERPARAMETER,
        }
        and isinstance(value, Tensor)
        and not isinstance(value, FakeTensor)
        and value.device.type == "cpu"
        and value.numel() <= 4096
    ):
        record["value"] = value.tolist()
    return record
