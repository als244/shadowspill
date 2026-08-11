"""Optimizer-agnostic capture with an explicit bounded opaque fallback."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch.fx import GraphModule

from .capture import GraphArtifact
from .contracts import CaptureError


class OptimizerTensorRole(StrEnum):
    PARAMETER = "parameter"
    GRADIENT = "gradient"
    STATE = "state"
    HYPERPARAMETER = "hyperparameter"


@dataclass(frozen=True, slots=True)
class OptimizerTensorBinding:
    name: str
    role: OptimizerTensorRole
    tensor: torch.Tensor
    mutable: bool
    spillable: bool


@dataclass(frozen=True, slots=True)
class OptimizerCapture:
    """First/recurrent optimizer task semantics and explicit tensor inventory."""

    optimizer_type: str
    first_step_is_opaque: bool
    created_state_names: tuple[str, ...]
    recurrent: GraphArtifact | None
    bindings: tuple[OptimizerTensorBinding, ...]
    mutation_names: tuple[str, ...]
    opaque_reason: str | None = None

    @property
    def recurrent_is_opaque(self) -> bool:
        return self.recurrent is None


def capture_optimizer(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> OptimizerCapture:
    """Capture a recurrent tensor update without mutating the caller's state.

    Lazy Python/tensor state is discovered on a deep-copied optimizer. Its first
    semantic update remains an ordinary bounded optimizer task. Once state is
    stable, a lifted tensor-only graph is used when Dynamo can represent it;
    otherwise all steps remain bounded opaque tasks with measured workspace.
    """

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must derive from torch.optim.Optimizer")
    canonical = _canonical_parameters(named_parameters)
    actual_parameters = _optimizer_parameters(optimizer)
    expected = {
        id(parameter) for parameter in canonical.values() if parameter.requires_grad
    }
    observed = {
        id(parameter) for parameter in actual_parameters if parameter.requires_grad
    }
    if expected != observed:
        missing = tuple(
            name
            for name, parameter in canonical.items()
            if parameter.requires_grad and id(parameter) not in observed
        )
        raise CaptureError(
            "optimizer parameter coverage differs from the model: "
            f"missing={missing}, extra={len(observed - expected)}"
        )
    optimizer_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    try:
        sandbox = copy.deepcopy(optimizer)
    except BaseException as exc:
        return OptimizerCapture(
            optimizer_type=optimizer_type,
            first_step_is_opaque=True,
            created_state_names=(),
            recurrent=None,
            bindings=(),
            mutation_names=(),
            opaque_reason=f"optimizer cannot be copied for capture: {exc}",
        )
    sandbox_parameters = _optimizer_parameters(sandbox)
    if len(sandbox_parameters) != len(actual_parameters):
        raise CaptureError("copied optimizer changed its parameter inventory")
    name_by_actual_id = {id(parameter): name for name, parameter in canonical.items()}
    name_by_sandbox_id = {
        id(sandbox_parameter): name_by_actual_id[id(actual_parameter)]
        for actual_parameter, sandbox_parameter in zip(
            actual_parameters, sandbox_parameters, strict=True
        )
        if id(actual_parameter) in name_by_actual_id
    }
    for actual_parameter, sandbox_parameter in zip(
        actual_parameters, sandbox_parameters, strict=True
    ):
        if not sandbox_parameter.requires_grad or sandbox_parameter.grad is not None:
            continue
        if actual_parameter.grad is None:
            sandbox_parameter.grad = torch.ones_like(sandbox_parameter)
        else:
            sandbox_parameter.grad = actual_parameter.grad.detach().clone()

    before_structure = _state_structure(sandbox, name_by_sandbox_id)
    before_state = copy.deepcopy(sandbox.state_dict())
    before_parameters = tuple(
        (
            parameter,
            parameter.detach().clone(),
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in sandbox_parameters
    )
    try:
        with torch.no_grad():
            sandbox.step()
    except BaseException as exc:
        return OptimizerCapture(
            optimizer_type=optimizer_type,
            first_step_is_opaque=True,
            created_state_names=(),
            recurrent=None,
            bindings=(),
            mutation_names=(),
            opaque_reason=f"optimizer discovery step failed: {exc}",
        )
    after_structure = _state_structure(sandbox, name_by_sandbox_id)
    first_step_is_opaque = after_structure != before_structure
    before_state_names = _state_tensor_names(optimizer, name_by_actual_id)
    after_state_names = _state_tensor_names(sandbox, name_by_sandbox_id)
    created_state_names = tuple(
        sorted(set(after_state_names) - set(before_state_names))
    )
    if not first_step_is_opaque:
        sandbox.load_state_dict(before_state)
        with torch.no_grad():
            for parameter, value, gradient in before_parameters:
                parameter.copy_(value)
                if gradient is not None and parameter.grad is not None:
                    parameter.grad.copy_(gradient)

    bindings = _tensor_bindings(sandbox, name_by_sandbox_id)
    snapshots = {
        id(binding.tensor): binding.tensor.detach().clone() for binding in bindings
    }
    try:
        graph_module = _export_optimizer_graph(sandbox)
        _restore_binding_values(bindings, snapshots)
        graph_module = _lift_optimizer_tensors(graph_module, bindings)
        artifact = GraphArtifact.capture(
            kind="optimizer",
            graph_module=graph_module,
            example_inputs=tuple(binding.tensor for binding in bindings),
        )
    except BaseException as exc:
        _restore_binding_values(bindings, snapshots)
        return OptimizerCapture(
            optimizer_type=optimizer_type,
            first_step_is_opaque=first_step_is_opaque,
            created_state_names=created_state_names,
            recurrent=None,
            bindings=bindings,
            mutation_names=tuple(
                binding.name for binding in bindings if binding.mutable
            ),
            opaque_reason=f"recurrent optimizer graph is opaque: {exc}",
        )
    mutations = tuple(binding.name for binding in bindings if binding.mutable)
    return OptimizerCapture(
        optimizer_type=optimizer_type,
        first_step_is_opaque=first_step_is_opaque,
        created_state_names=created_state_names,
        recurrent=artifact,
        bindings=bindings,
        mutation_names=mutations,
    )


def current_optimizer_bindings(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> tuple[OptimizerTensorBinding, ...]:
    """Describe current optimizer tensors with capture-stable names."""

    canonical = _canonical_parameters(named_parameters)
    name_by_id = {id(parameter): name for name, parameter in canonical.items()}
    return _tensor_bindings(optimizer, name_by_id, require_gradients=False)


def _canonical_parameters(
    named_parameters: Mapping[str, torch.nn.Parameter],
) -> dict[str, torch.nn.Parameter]:
    result: dict[str, torch.nn.Parameter] = {}
    seen: set[int] = set()
    for name, parameter in named_parameters.items():
        if not name:
            raise CaptureError("model parameter names must be non-empty")
        if not isinstance(parameter, torch.nn.Parameter):
            raise CaptureError(f"model value {name!r} is not a Parameter")
        if id(parameter) not in seen:
            result[name] = parameter
            seen.add(id(parameter))
    return result


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.nn.Parameter, ...]:
    result: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not isinstance(parameter, torch.nn.Parameter):
                raise CaptureError("optimizer contains a non-Parameter entry")
            if id(parameter) in seen:
                raise CaptureError("optimizer contains one parameter more than once")
            seen.add(id(parameter))
            result.append(parameter)
    return tuple(result)


def _tensor_leaves(
    value: object, prefix: str = ""
) -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "value", value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _tensor_leaves(value[key], child)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from _tensor_leaves(item, child)


def _state_tensor_names(
    optimizer: torch.optim.Optimizer, name_by_id: Mapping[int, str]
) -> tuple[str, ...]:
    names: list[str] = []
    for parameter in _optimizer_parameters(optimizer):
        parameter_name = name_by_id.get(id(parameter))
        if parameter_name is None:
            continue
        for path, _tensor in _tensor_leaves(optimizer.state.get(parameter, {})):
            names.append(f"optimizer.{parameter_name}.{path}")
    return tuple(names)


def _state_structure(
    optimizer: torch.optim.Optimizer, name_by_id: Mapping[int, str]
) -> tuple[tuple[str, str, tuple[int, ...], str], ...]:
    result: list[tuple[str, str, tuple[int, ...], str]] = []
    for parameter in _optimizer_parameters(optimizer):
        parameter_name = name_by_id.get(id(parameter), "unknown")
        for path, tensor in _tensor_leaves(optimizer.state.get(parameter, {})):
            result.append(
                (parameter_name, path, tuple(tensor.shape), str(tensor.dtype))
            )
    return tuple(result)


def _tensor_bindings(
    optimizer: torch.optim.Optimizer,
    name_by_id: Mapping[int, str],
    *,
    require_gradients: bool = True,
) -> tuple[OptimizerTensorBinding, ...]:
    bindings: list[OptimizerTensorBinding] = []
    seen: set[int] = set()

    def add(
        name: str, role: OptimizerTensorRole, tensor: torch.Tensor, mutable: bool
    ) -> None:
        if id(tensor) in seen:
            return
        seen.add(id(tensor))
        # Scalar optimizer control values are deliberately not spill objects.
        # PyTorch optimizers may keep them on the CPU (ordinary AdamW) or on the
        # accelerator (capturable/custom optimizers).  In either case their
        # bounded footprint belongs to task/provider admission, while tensor
        # state such as moments remains fully planned and budgeted.
        spillable = (
            role
            in {
                OptimizerTensorRole.PARAMETER,
                OptimizerTensorRole.GRADIENT,
            }
            or tensor.ndim != 0
        )
        bindings.append(OptimizerTensorBinding(name, role, tensor, mutable, spillable))

    for parameter in _optimizer_parameters(optimizer):
        name = name_by_id.get(id(parameter))
        if name is None or not parameter.requires_grad:
            continue
        add(name, OptimizerTensorRole.PARAMETER, parameter, True)
        if parameter.grad is None:
            if require_gradients:
                raise CaptureError(f"optimizer capture has no gradient for {name!r}")
        else:
            add(
                f"gradient.{name}",
                OptimizerTensorRole.GRADIENT,
                parameter.grad,
                False,
            )
        for path, tensor in _tensor_leaves(optimizer.state.get(parameter, {})):
            add(f"optimizer.{name}.{path}", OptimizerTensorRole.STATE, tensor, True)
    for group_index, group in enumerate(optimizer.param_groups):
        for path, tensor in _tensor_leaves(
            {key: value for key, value in group.items() if key != "params"}
        ):
            add(
                f"optimizer_group.{group_index}.{path}",
                OptimizerTensorRole.HYPERPARAMETER,
                tensor,
                True,
            )
    return tuple(bindings)


def _export_optimizer_graph(optimizer: torch.optim.Optimizer) -> GraphModule:
    raw_step = inspect.unwrap(type(optimizer).step).__get__(optimizer, type(optimizer))

    @torch.no_grad()
    def update() -> Any:
        return raw_step()

    with torch._dynamo.config.patch(
        recompile_limit=max(torch._dynamo.config.recompile_limit, 64)
    ):
        exported = torch._dynamo.export(update, aten_graph=True)()
    return exported.graph_module


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


def _restore_binding_values(
    bindings: tuple[OptimizerTensorBinding, ...], snapshots: Mapping[int, torch.Tensor]
) -> None:
    with torch.no_grad():
        for binding in bindings:
            binding.tensor.copy_(snapshots[id(binding.tensor)])
