"""The optimizer update traced once as a lifted tensor-only graph, served from the
store when it was traced before, and the bounded opaque task it falls back to."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact

from .artifacts import (
    OpaqueOptimizerArtifact,
    OptimizerCapture,
    OptimizerTask,
    OptimizerTensorBinding,
    OptimizerTensorRole,
)
from .bindings import (
    optimizer_input_provenance,
    restore_binding_values,
    step_bindings,
    tensor_bindings,
)
from .discovery import (
    OptimizerDiscovery,
    fake_update_sandbox,
    is_data_dependent_failure,
)
from .phases import PhaseTimer
from .sandbox import (
    has_optimizer_step_hooks,
)
from .store import OptimizerCaptureStore, update_capture_identity
from .tasks import (
    partition_optimizer_graph,
)


def capture_optimizer_update(
    discovery: OptimizerDiscovery,
    optimizer: torch.optim.Optimizer,
    *,
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None,
    store: OptimizerCaptureStore | None = None,
    timer: PhaseTimer,
    compute_copies: Mapping[str, torch.Tensor] | None = None,
    gradient_dtype: torch.dtype | None = None,
) -> OptimizerCapture:
    """Capture the optimizer's update, or publish a bounded opaque task.

    The update takes each gradient as the step produces it -- at
    ``gradient_dtype``, or at its weights' dtype -- and casts it where the
    optimizer's parameter is at another. ``compute_copies`` names the
    parameters the optimizer holds master copies of, with the weights the
    model computes with, and the update writes each copy from its master once
    it has stepped.
    """

    copies = dict(compute_copies or {})
    for name, copy in copies.items():
        if not isinstance(copy, FakeTensor) and not copy.is_meta:
            discovery.representative_values[f"compute.{name}"] = copy.detach()
    if has_optimizer_step_hooks(optimizer):
        return _hooked_optimizer_capture(discovery)
    key: str | None = None
    if store is not None:
        key = update_capture_identity(
            discovery.sandbox,
            step_bindings(
                tensor_bindings(discovery.sandbox, discovery.name_by_sandbox_id),
                copies,
                gradient_dtype,
            ),
            parameter_stage_owners=parameter_stage_owners,
        )
        stored = store.read(key)
        if stored is not None:
            # The trace is what the store holds; the split of it into stage
            # tasks is graph analysis, derived here as it is on a miss.
            with timer.measure("optimizer_trace_read"):
                fake_update_sandbox(discovery)
                bindings = step_bindings(
                    tensor_bindings(discovery.sandbox, discovery.name_by_sandbox_id),
                    copies,
                    gradient_dtype,
                )
                artifact = stored.restore(
                    bindings,
                    optimizer_input_provenance(
                        bindings, discovery.representative_values
                    ),
                )
                update_tasks = partition_optimizer_graph(
                    artifact,
                    bindings,
                    parameter_stage_owners=parameter_stage_owners,
                )
            return _update_capture(discovery, artifact, update_tasks, bindings)
    # The trace decides whether the update is representable: it exports once
    # and falls back to an opaque task when the export fails, so nothing is
    # exported twice to find out first.
    if discovery.initialized_state_dict is None:
        fake_update_sandbox(discovery)
    with timer.measure("optimizer_trace"):
        captured = _capture_optimizer_artifact(discovery, copies, gradient_dtype)
    if isinstance(captured, OptimizerCapture):
        return captured
    artifact, bindings = captured
    update_tasks = partition_optimizer_graph(
        artifact,
        bindings,
        parameter_stage_owners=parameter_stage_owners,
    )
    if store is not None and key is not None:
        store.write(key, artifact, optimizer_type=discovery.optimizer_type)
    return _update_capture(discovery, artifact, update_tasks, bindings)


def _update_capture(
    discovery: OptimizerDiscovery,
    artifact: GraphArtifact,
    update_tasks: tuple[OptimizerTask, ...],
    bindings: tuple[OptimizerTensorBinding, ...],
) -> OptimizerCapture:
    return OptimizerCapture(
        optimizer_type=discovery.optimizer_type,
        update=artifact,
        update_tasks=update_tasks,
        bindings=bindings,
        mutation_names=tuple(binding.name for binding in bindings if binding.mutable),
        initialized_state_dict=discovery.initialized_state_dict,
    )


def _hooked_optimizer_capture(
    discovery: OptimizerDiscovery,
) -> OptimizerCapture:
    bindings = tensor_bindings(
        discovery.sandbox,
        discovery.name_by_sandbox_id,
    )
    artifact = OpaqueOptimizerArtifact.capture(discovery.sandbox, bindings)
    return _opaque_optimizer_capture(
        discovery,
        artifact,
        bindings,
        reason="optimizer step hooks require ordinary eager execution",
    )


def _capture_optimizer_artifact(
    discovery: OptimizerDiscovery,
    copies: Mapping[str, torch.Tensor],
    gradient_dtype: torch.dtype | None,
) -> tuple[GraphArtifact, tuple[OptimizerTensorBinding, ...]] | OptimizerCapture:
    sandbox = discovery.sandbox
    names = discovery.name_by_sandbox_id
    own = tensor_bindings(sandbox, names)
    bindings = step_bindings(own, copies, gradient_dtype)
    # What the update closes over: the optimizer's own tensors, each gradient
    # among them at its parameter's dtype, and the copies it writes. The
    # graph's inputs are found by those; the bindings then say what each
    # input is, a gradient arriving at the dtype the step produces it in.
    closed = own + bindings[len(own) :]
    masters = {
        binding.name: binding.tensor
        for binding in own
        if binding.role is OptimizerTensorRole.PARAMETER
    }
    written = tuple(
        (binding.tensor, masters[binding.name.removeprefix("compute.")])
        for binding in bindings[len(own) :]
    )
    snapshots = {
        id(binding.tensor): binding.tensor.detach().clone() for binding in closed
    }
    grad_enabled = torch.is_grad_enabled()
    try:
        graph_module = _export_optimizer_graph(sandbox, written)
        restore_binding_values(closed, snapshots)
        graph_module = _lift_optimizer_tensors(graph_module, closed)
        graph_module = _cast_gradients(graph_module, closed, bindings)
        artifact = GraphArtifact.capture(
            kind="optimizer",
            graph_module=graph_module,
            example_inputs=tuple(binding.tensor for binding in bindings),
            input_provenance=optimizer_input_provenance(
                bindings,
                discovery.representative_values,
            ),
        )
    except BaseException as exc:
        restore_binding_values(closed, snapshots)
        # An opaque task runs the real update, so it is captured from the
        # sandbox as it was before it moved onto fake tensors.
        real = discovery.real_sandbox
        real_names = discovery.real_name_by_sandbox_id
        if real is None or real_names is None:
            real, real_names = sandbox, names
        discovery.sandbox = real
        discovery.name_by_sandbox_id = real_names
        real_bindings = tensor_bindings(real, real_names)
        opaque_artifact = OpaqueOptimizerArtifact.capture(real, real_bindings)
        return _opaque_optimizer_capture(
            discovery,
            opaque_artifact,
            real_bindings,
            reason=_opaque_optimizer_reason(exc),
        )
    finally:
        torch.set_grad_enabled(grad_enabled)
    return artifact, bindings


def _opaque_optimizer_reason(failure: BaseException) -> str:
    description = str(failure)
    if is_data_dependent_failure(failure):
        return f"the optimizer update is data-dependent: {description}"
    return f"the optimizer update is opaque: {description}"


def _opaque_optimizer_capture(
    discovery: OptimizerDiscovery,
    artifact: OpaqueOptimizerArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
    *,
    reason: str,
) -> OptimizerCapture:
    mutations = tuple(binding.name for binding in bindings if binding.mutable)
    return OptimizerCapture(
        optimizer_type=discovery.optimizer_type,
        update=artifact,
        update_tasks=(
            OptimizerTask(
                artifact,
                tuple(binding.name for binding in bindings),
                mutations,
            ),
        ),
        bindings=bindings,
        mutation_names=mutations,
        opaque_reason=reason,
        initialized_state_dict=discovery.initialized_state_dict,
    )


def _export_optimizer_graph(
    optimizer: torch.optim.Optimizer,
    written: tuple[tuple[torch.Tensor, torch.Tensor], ...] = (),
) -> GraphModule:
    """The step as a graph; each ``(copy, master)`` in ``written`` is then
    written from its master, in the same graph."""

    raw_step = inspect.unwrap(type(optimizer).step).__get__(optimizer, type(optimizer))

    @torch.no_grad()
    def update() -> Any:
        result = raw_step()
        for copy, master in written:
            copy.copy_(master)
        return result

    with torch._dynamo.config.patch(
        recompile_limit=max(torch._dynamo.config.recompile_limit, 64)
    ):
        exported = torch._dynamo.export(update, aten_graph=True)()
    return exported.graph_module


def _cast_gradients(
    graph_module: GraphModule,
    closed: tuple[OptimizerTensorBinding, ...],
    bindings: tuple[OptimizerTensorBinding, ...],
) -> GraphModule:
    """Take each gradient at the dtype the step produces it in.

    The graph was traced with each gradient at its parameter's dtype, as an
    optimizer requires; where the input it gets is at another -- a master's
    gradient, as the model computes it, or a gradient kept at a wider dtype
    -- the graph casts it before anything reads it. Nothing else changes: the
    cast is the first use of the input and every former use reads the cast.
    """

    graph = graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    changed = False
    for node, before, after in zip(placeholders, closed, bindings, strict=True):
        if before.tensor is after.tensor:
            continue
        with graph.inserting_after(node):
            cast = graph.call_function(
                torch.ops.prims.convert_element_type.default,
                (node, before.tensor.dtype),
            )
        node.replace_all_uses_with(
            cast, delete_user_cb=lambda user, cast=cast: user is not cast
        )
        changed = True
    if changed:
        graph.lint()
        graph_module.recompile()
    return graph_module


def _lift_optimizer_tensors(
    graph_module: GraphModule,
    bindings: tuple[OptimizerTensorBinding, ...],
) -> GraphModule:
    by_identity = {id(binding.tensor): binding for binding in bindings}
    graph = graph_module.graph
    first = next(iter(graph.nodes))
    placeholders: dict[int, torch.fx.Node] = {}
    with graph.inserting_before(first):
        for index, binding in enumerate(bindings):
            placeholders[id(binding.tensor)] = graph.placeholder(
                f"optimizer_tensor_{index:04d}"
            )
    lifted: list[str] = []
    unknown: list[str] = []
    for node in tuple(graph.nodes):
        if node.op != "get_attr":
            continue
        value = getattr(graph_module, node.target)
        resolved = (
            by_identity.get(id(value)) if isinstance(value, torch.Tensor) else None
        )
        if resolved is None:
            unknown.append(f"{node.target}:{type(value).__name__}")
            continue
        node.replace_all_uses_with(placeholders[id(resolved.tensor)])
        lifted.append(str(node.target))
        graph.erase_node(node)
    if unknown:
        raise CaptureError(
            f"optimizer graph closed over untracked values: {tuple(unknown)}"
        )
    mutable = tuple(binding for binding in bindings if binding.mutable)
    output = next(node for node in graph.nodes if node.op == "output")
    output.args = (tuple(placeholders[id(binding.tensor)] for binding in mutable),)
    graph.set_codegen(torch.fx.graph.CodeGen())
    graph.lint()
    graph_module.recompile()
    for target in lifted:
        if hasattr(graph_module, target):
            delattr(graph_module, target)
    return graph_module
