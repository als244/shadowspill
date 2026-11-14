"""Bind one final scale at each optimizer gradient consumer before compilation."""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace

import torch
from torch._guards import detect_fake_mode

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
from shadowspill.pytorch.capture.torch_deprecations import copy_graph_module
from shadowspill.task.inputs import TaskInputRole

from .artifacts import OptimizerCapture, OptimizerTensorBinding, OptimizerTensorRole
from .tasks import partition_optimizer_graph


def with_gradient_scale(
    captured: OptimizerCapture,
    *,
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[OptimizerCapture, str]:
    """Return a separate capture with a read-only scalar control, defaulting to 1.

    Scaling follows the existing cast to optimizer precision. The stored input
    gradient is unchanged. Repartitioning preserves independent update frontiers:
    sharing a read-only scale does not join unrelated parameter updates.
    """
    graph_module = copy_graph_module(captured.update.graph_module)
    graph = graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    first_operation = next(node for node in graph.nodes if node.op != "placeholder")
    # Precision casts can sit between placeholders in the existing FX graph.
    # Keep its argument order when appending the extra control binding.
    for node in placeholders:
        first_operation.prepend(node)
    with graph.inserting_before(first_operation):
        scale = graph.placeholder("gradient_scale")
    parameters = {
        binding.name: binding.tensor.dtype
        for binding in captured.bindings
        if binding.role is OptimizerTensorRole.PARAMETER
    }
    for node, binding in zip(placeholders, captured.bindings, strict=True):
        if binding.role is not OptimizerTensorRole.GRADIENT or not node.users:
            continue
        source = node
        assert binding.parameter_name is not None
        dtype = parameters[binding.parameter_name]
        if binding.tensor.dtype != dtype:
            users = tuple(node.users)
            if (
                len(users) != 1
                or users[0].target != torch.ops.prims.convert_element_type.default
                or users[0].args != (node, dtype)
            ):
                raise CaptureError("optimizer gradient has no canonical precision cast")
            source = users[0]
        with (
            graph.inserting_before(first_operation)
            if source.op == "placeholder"
            else graph.inserting_after(source)
        ):
            scaled = graph.call_function(torch.ops.aten.mul.Tensor, (source, scale))
        source.replace_all_uses_with(
            scaled, delete_user_cb=lambda user, scaled=scaled: user is not scaled
        )
    graph.lint()
    graph_module.recompile()
    name = "grad_scale"
    names = {binding.name for binding in captured.bindings}
    while name in names:
        name = "_" + name
    representative = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    mode = detect_fake_mode(captured.update.example_arguments)
    with mode if mode is not None else nullcontext():
        control = torch.empty((), dtype=torch.float32, device="cpu")
    bindings = (
        *captured.bindings,
        OptimizerTensorBinding(
            name, OptimizerTensorRole.HYPERPARAMETER, control, False, False
        ),
    )
    artifact = GraphArtifact.capture(
        kind="optimizer",
        graph_module=graph_module,
        example_inputs=(*captured.update.example_arguments, control),
        input_provenance=(
            *captured.update.input_provenance,
            TaskInputProvenance(
                TaskInputRole.OPTIMIZER_HYPERPARAMETER,
                name,
                representative_value=representative,
            ),
        ),
    )
    return replace(
        captured,
        update=artifact,
        bindings=bindings,
        update_tasks=partition_optimizer_graph(
            artifact, bindings, parameter_stage_owners=parameter_stage_owners
        ),
    ), name
