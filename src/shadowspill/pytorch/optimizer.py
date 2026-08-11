"""Optimizer-agnostic capture with an explicit bounded opaque fallback."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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
class OpaqueOptimizerArtifact:
    """One eager optimizer task with a deterministic structural identity."""

    optimizer_type: str
    compatibility_digest: str
    optimizer: torch.optim.Optimizer = field(repr=False, compare=False)

    @classmethod
    def capture(
        cls,
        optimizer: torch.optim.Optimizer,
        bindings: tuple[OptimizerTensorBinding, ...],
    ) -> OpaqueOptimizerArtifact:
        optimizer_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
        step = inspect.unwrap(type(optimizer).step)
        code = getattr(step, "__code__", None)
        code_identity = (
            None
            if code is None
            else {
                "bytecode": code.co_code.hex(),
                "constants": tuple(repr(value) for value in code.co_consts),
                "names": code.co_names,
            }
        )
        identity = {
            "kind": "opaque_optimizer",
            "optimizer_type": optimizer_type,
            "step": code_identity,
            "bindings": [
                {
                    "name": binding.name,
                    "role": binding.role.value,
                    "mutable": binding.mutable,
                    "spillable": binding.spillable,
                    "shape": tuple(binding.tensor.shape),
                    "stride": tuple(binding.tensor.stride()),
                    "dtype": str(binding.tensor.dtype),
                    "device": binding.tensor.device.type,
                }
                for binding in bindings
            ],
            "groups": [
                _optimizer_value_identity(
                    {key: value for key, value in group.items() if key != "params"}
                )
                for group in optimizer.param_groups
            ],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return cls(
            optimizer_type,
            hashlib.sha256(encoded.encode()).hexdigest(),
            optimizer,
        )


OptimizerTaskArtifact = GraphArtifact | OpaqueOptimizerArtifact


@dataclass(frozen=True, slots=True)
class OptimizerCapture:
    """First/recurrent optimizer task semantics and explicit tensor inventory."""

    optimizer_type: str
    first_step_is_opaque: bool
    created_state_names: tuple[str, ...]
    recurrent: OptimizerTaskArtifact | None
    recurrent_tasks: tuple[OptimizerTask, ...]
    bindings: tuple[OptimizerTensorBinding, ...]
    mutation_names: tuple[str, ...]
    opaque_reason: str | None = None
    initialized_state_dict: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def recurrent_is_opaque(self) -> bool:
        return self.recurrent is None or isinstance(
            self.recurrent, OpaqueOptimizerArtifact
        )


@dataclass(frozen=True, slots=True)
class OptimizerTask:
    """One dependency-closed recurrent optimizer component."""

    artifact: OptimizerTaskArtifact
    binding_names: tuple[str, ...]
    mutation_names: tuple[str, ...]


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
    initialized_state_dict: dict[str, Any] | None
    try:
        sandbox = _copy_optimizer(optimizer)
    except BaseException as exc:
        return OptimizerCapture(
            optimizer_type=optimizer_type,
            first_step_is_opaque=True,
            created_state_names=(),
            recurrent=None,
            recurrent_tasks=(),
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
        # CUDA-only registered operators commonly reject the CPU discovery
        # sandbox *after* creating lazy optimizer state.  That state inventory
        # is still authoritative.  Re-run capture with FakeTensor CUDA values
        # so the operator's registered fake/meta contract describes the
        # recurrent task without allocating a second real model.
        after_structure = _state_structure(sandbox, name_by_sandbox_id)
        if after_structure == before_structure:
            return OptimizerCapture(
                optimizer_type=optimizer_type,
                first_step_is_opaque=True,
                created_state_names=(),
                recurrent=None,
                recurrent_tasks=(),
                bindings=(),
                mutation_names=(),
                opaque_reason=f"optimizer discovery step failed: {exc}",
            )
        try:
            _complete_failed_state_discovery(
                sandbox,
                name_by_sandbox_id,
                before_parameters,
            )
            initialized_state_dict = sandbox.state_dict()
            sandbox, name_by_sandbox_id = _fake_cuda_optimizer(
                sandbox, name_by_sandbox_id
            )
        except BaseException as fake_exc:
            return OptimizerCapture(
                optimizer_type=optimizer_type,
                first_step_is_opaque=True,
                created_state_names=(),
                recurrent=None,
                recurrent_tasks=(),
                bindings=(),
                mutation_names=(),
                opaque_reason=(
                    f"optimizer discovery step failed: {exc}; "
                    f"fake CUDA inventory failed: {fake_exc}"
                ),
            )
    else:
        initialized_state_dict = None
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

    if _has_optimizer_step_hooks(optimizer):
        hook_bindings = _tensor_bindings(sandbox, name_by_sandbox_id)
        hook_artifact = OpaqueOptimizerArtifact.capture(sandbox, hook_bindings)
        return OptimizerCapture(
            optimizer_type=optimizer_type,
            first_step_is_opaque=first_step_is_opaque,
            created_state_names=created_state_names,
            recurrent=hook_artifact,
            recurrent_tasks=(
                OptimizerTask(
                    hook_artifact,
                    tuple(binding.name for binding in hook_bindings),
                    tuple(binding.name for binding in hook_bindings if binding.mutable),
                ),
            ),
            bindings=hook_bindings,
            mutation_names=tuple(
                binding.name for binding in hook_bindings if binding.mutable
            ),
            opaque_reason="optimizer step hooks require ordinary eager execution",
            initialized_state_dict=initialized_state_dict,
        )

    if initialized_state_dict is None:
        probe_bindings = _tensor_bindings(sandbox, name_by_sandbox_id)
        probe_snapshots = {
            id(binding.tensor): binding.tensor.detach().clone()
            for binding in probe_bindings
        }
        probe_grad_enabled = torch.is_grad_enabled()
        try:
            _export_optimizer_graph(sandbox)
        except BaseException as exc:
            _restore_binding_values(probe_bindings, probe_snapshots)
            opaque_artifact = OpaqueOptimizerArtifact.capture(sandbox, probe_bindings)
            return OptimizerCapture(
                optimizer_type=optimizer_type,
                first_step_is_opaque=first_step_is_opaque,
                created_state_names=created_state_names,
                recurrent=opaque_artifact,
                recurrent_tasks=(
                    OptimizerTask(
                        opaque_artifact,
                        tuple(binding.name for binding in probe_bindings),
                        tuple(
                            binding.name
                            for binding in probe_bindings
                            if binding.mutable
                        ),
                    ),
                ),
                bindings=probe_bindings,
                mutation_names=tuple(
                    binding.name for binding in probe_bindings if binding.mutable
                ),
                opaque_reason=f"recurrent optimizer graph is opaque: {exc}",
            )
        finally:
            torch.set_grad_enabled(probe_grad_enabled)
        _restore_binding_values(probe_bindings, probe_snapshots)
        sandbox, name_by_sandbox_id = _fake_cuda_optimizer(sandbox, name_by_sandbox_id)

    bindings = _tensor_bindings(sandbox, name_by_sandbox_id)
    snapshots = {
        id(binding.tensor): binding.tensor.detach().clone() for binding in bindings
    }
    grad_enabled = torch.is_grad_enabled()
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
        opaque_artifact = OpaqueOptimizerArtifact.capture(sandbox, bindings)
        opaque_tasks = (
            OptimizerTask(
                opaque_artifact,
                tuple(binding.name for binding in bindings),
                tuple(binding.name for binding in bindings if binding.mutable),
            ),
        )
        return OptimizerCapture(
            optimizer_type=optimizer_type,
            first_step_is_opaque=first_step_is_opaque,
            created_state_names=created_state_names,
            recurrent=opaque_artifact,
            recurrent_tasks=opaque_tasks,
            bindings=bindings,
            mutation_names=tuple(
                binding.name for binding in bindings if binding.mutable
            ),
            opaque_reason=f"recurrent optimizer graph is opaque: {exc}",
            initialized_state_dict=initialized_state_dict,
        )
    finally:
        torch.set_grad_enabled(grad_enabled)
    mutations = tuple(binding.name for binding in bindings if binding.mutable)
    recurrent_tasks = _partition_optimizer_graph(artifact, bindings)
    return OptimizerCapture(
        optimizer_type=optimizer_type,
        first_step_is_opaque=first_step_is_opaque,
        created_state_names=created_state_names,
        recurrent=artifact,
        recurrent_tasks=recurrent_tasks,
        bindings=bindings,
        mutation_names=mutations,
        initialized_state_dict=initialized_state_dict,
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


def _has_optimizer_step_hooks(optimizer: torch.optim.Optimizer) -> bool:
    return bool(
        getattr(optimizer, "_optimizer_step_pre_hooks", None)
        or getattr(optimizer, "_optimizer_step_post_hooks", None)
    )


def _copy_optimizer(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Copy complete subclass state without Optimizer.__getstate__ truncation."""

    copied = copy.deepcopy(optimizer)
    if copied.__dict__.keys() == optimizer.__dict__.keys():
        return copied
    copied = object.__new__(type(optimizer))
    copied.__dict__ = copy.deepcopy(optimizer.__dict__)
    if not isinstance(copied, torch.optim.Optimizer):
        raise TypeError("copied optimizer changed its base type")
    return copied


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


def _fake_cuda_optimizer(
    optimizer: torch.optim.Optimizer,
    name_by_id: Mapping[int, str],
) -> tuple[torch.optim.Optimizer, dict[int, str]]:
    """Replace serializable optimizer tensors with FakeTensor CUDA values.

    PyTorch's optimizer protocol defines parameters through ``param_groups``
    and per-parameter tensors through ``state``.  Restricting conversion to
    that protocol keeps this fallback independent of optimizer classes while
    allowing registered CUDA-only operations to participate through their fake
    implementations.  Undeclared tensor closures are rejected later when the
    FX graph is lifted.
    """

    parameters = _optimizer_parameters(optimizer)
    replacements: dict[int, torch.Tensor] = {}
    fake_names: dict[int, str] = {}
    mode = torch._subclasses.fake_tensor.FakeTensorMode(allow_non_fake_inputs=True)

    def fake_tensor(value: torch.Tensor, *, parameter: bool = False) -> torch.Tensor:
        existing = replacements.get(id(value))
        if existing is not None:
            return existing
        if value.layout is not torch.strided:
            raise CaptureError("optimizer fake capture requires strided tensors")
        with mode:
            raw = torch.empty_strided(
                tuple(value.shape),
                tuple(value.stride()),
                dtype=value.dtype,
                device="cuda",
            )
            result: torch.Tensor
            if parameter:
                result = torch.nn.Parameter(raw, requires_grad=value.requires_grad)
            else:
                result = raw.requires_grad_(value.requires_grad)
        replacements[id(value)] = result
        return result

    fake_parameters: dict[int, torch.nn.Parameter] = {}
    for value in parameters:
        converted = fake_tensor(value, parameter=True)
        if not isinstance(converted, torch.nn.Parameter):
            raise AssertionError("parameter conversion changed tensor type")
        if value.grad is not None:
            converted.grad = fake_tensor(value.grad)
        fake_parameters[id(value)] = converted
        name = name_by_id.get(id(value))
        if name is not None:
            fake_names[id(converted)] = name

    for group in optimizer.param_groups:
        group["params"] = [fake_parameters[id(value)] for value in group["params"]]

    original_state = optimizer.state
    converted_state: defaultdict[torch.Tensor, dict[str, Any]] = defaultdict(dict)
    for parameter, value in original_state.items():
        fake_parameter = fake_parameters.get(id(parameter))
        if fake_parameter is None:
            raise CaptureError("optimizer state is keyed by an unknown parameter")
        converted = _map_optimizer_tensors(value, fake_tensor)
        if not isinstance(converted, dict):
            raise CaptureError("per-parameter optimizer state must be a mapping")
        converted_state[fake_parameter] = converted
    optimizer.state = converted_state
    return optimizer, fake_names


def _complete_failed_state_discovery(
    optimizer: torch.optim.Optimizer,
    name_by_id: Mapping[int, str],
    parameter_snapshots: tuple[
        tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor | None], ...
    ],
) -> None:
    """Discover lazy state hidden behind a failing per-parameter operation.

    A CUDA-only operation can reject the CPU sandbox after its optimizer has
    initialized one parameter's state. Optimizers commonly visit parameters in
    sequence, so one failed call does not establish the complete recurrent
    tensor inventory. Retry with gradients enabled only for parameters whose
    state is still empty. Every failed attempt must leave parameter values
    unchanged; otherwise the failure boundary is not safe to use for capture.
    """

    parameters = _optimizer_parameters(optimizer)
    gradients = {id(parameter): parameter.grad for parameter in parameters}
    snapshots_by_id = {
        id(parameter): (parameter, value, gradient)
        for parameter, value, gradient in parameter_snapshots
    }
    previous_structure = _state_structure(optimizer, name_by_id)
    try:
        while True:
            pending = tuple(
                parameter
                for parameter in parameters
                if parameter.requires_grad
                and gradients[id(parameter)] is not None
                and not optimizer.state.get(parameter)
            )
            if not pending:
                return
            selected = pending[0]
            for parameter in parameters:
                parameter.grad = (
                    gradients[id(parameter)] if parameter is selected else None
                )
            try:
                with torch.no_grad():
                    optimizer.step()
            except BaseException:
                _require_unchanged_discovery_parameters(
                    (snapshots_by_id[id(selected)],)
                )
                current_structure = _state_structure(optimizer, name_by_id)
                if current_structure == previous_structure:
                    return
                previous_structure = current_structure
            else:
                return
    finally:
        for parameter in parameters:
            parameter.grad = gradients[id(parameter)]
        # One final complete audit catches optimizers that mutate tensors whose
        # gradient was absent. Per-failure checks stay linear in total tensor
        # bytes instead of rescanning the complete model for every parameter.
        _require_unchanged_discovery_parameters(parameter_snapshots)


def _require_unchanged_discovery_parameters(
    snapshots: tuple[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor | None], ...],
) -> None:
    for parameter, value, _gradient in snapshots:
        if not torch.equal(parameter, value):
            raise CaptureError("optimizer discovery failed after mutating a parameter")


def _map_optimizer_tensors(value: Any, convert: Any) -> Any:
    """Preserve optimizer state containers while replacing tensor leaves."""

    if isinstance(value, torch.Tensor):
        return convert(value)
    if isinstance(value, dict):
        return {
            key: _map_optimizer_tensors(item, convert) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_map_optimizer_tensors(item, convert) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_optimizer_tensors(item, convert) for item in value)
    return copy.deepcopy(value)


def materialize_opaque_optimizer(
    artifact: OpaqueOptimizerArtifact, *, device_ordinal: int
) -> torch.optim.Optimizer:
    """Build an isolated real-CUDA optimizer used only for task profiling."""

    optimizer = _copy_optimizer(artifact.optimizer)
    parameters = _optimizer_parameters(optimizer)
    replacements: dict[int, torch.Tensor] = {}
    device = torch.device("cuda", device_ordinal)

    def real_tensor(value: torch.Tensor, *, parameter: bool = False) -> torch.Tensor:
        existing = replacements.get(id(value))
        if existing is not None:
            return existing
        if value.layout is not torch.strided:
            raise CaptureError("opaque optimizer profiling requires strided tensors")
        if isinstance(value, torch._subclasses.fake_tensor.FakeTensor):
            raise CaptureError("opaque optimizer profiling requires concrete state")
        with torch.no_grad():
            raw = torch.empty_strided(
                tuple(value.shape),
                tuple(value.stride()),
                dtype=value.dtype,
                device=device,
            )
            raw.copy_(value)
            result: torch.Tensor
            if parameter:
                result = torch.nn.Parameter(raw, requires_grad=value.requires_grad)
            else:
                result = raw.requires_grad_(value.requires_grad)
        replacements[id(value)] = result
        return result

    real_parameters: dict[int, torch.nn.Parameter] = {}
    for value in parameters:
        converted = real_tensor(value, parameter=True)
        if not isinstance(converted, torch.nn.Parameter):
            raise AssertionError("parameter conversion changed tensor type")
        if value.grad is not None:
            converted.grad = real_tensor(value.grad)
        real_parameters[id(value)] = converted
    for group in optimizer.param_groups:
        group["params"] = [real_parameters[id(value)] for value in group["params"]]

    converted_state: defaultdict[torch.Tensor, dict[str, Any]] = defaultdict(dict)
    for parameter, value in optimizer.state.items():
        real_parameter = real_parameters.get(id(parameter))
        if real_parameter is None:
            raise CaptureError("optimizer state is keyed by an unknown parameter")
        converted = _map_optimizer_tensors(value, real_tensor)
        if not isinstance(converted, dict):
            raise CaptureError("per-parameter optimizer state must be a mapping")
        converted_state[real_parameter] = converted
    optimizer.state = converted_state
    return optimizer


def _optimizer_value_identity(value: Any) -> Any:
    """Serialize bounded optimizer options without retaining framework values."""

    if isinstance(value, torch.Tensor):
        return {
            "tensor": {
                "shape": tuple(value.shape),
                "stride": tuple(value.stride()),
                "dtype": str(value.dtype),
                "device": value.device.type,
            }
        }
    if isinstance(value, Mapping):
        return {
            str(key): _optimizer_value_identity(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return {"tuple": [_optimizer_value_identity(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_optimizer_value_identity(item) for item in value]}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return {"type": type(value).__qualname__, "value": repr(value)}


def _tensor_leaves(
    value: object, prefix: str = ""
) -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix or "value", value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _tensor_leaves(value[key], child)
    elif isinstance(value, tuple | list):
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
        # Device-side control tensors belong in the physical plan even when
        # scalar. Ordinary PyTorch optimizers intentionally keep some scalar
        # counters on the CPU; those remain bounded host-side task inputs.
        spillable = (
            role
            in {
                OptimizerTensorRole.PARAMETER,
                OptimizerTensorRole.GRADIENT,
            }
            or tensor.ndim != 0
            or tensor.device.type != "cpu"
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


def _partition_optimizer_graph(
    artifact: GraphArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
) -> tuple[OptimizerTask, ...]:
    """Split independent tensor updates without using optimizer semantics."""

    graph_module = artifact.graph_module
    placeholders = tuple(
        node for node in graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(bindings):
        raise CaptureError("optimizer placeholder inventory changed after lifting")
    operations = tuple(
        node
        for node in graph_module.graph.nodes
        if node.op not in {"placeholder", "output"}
    )
    if len(operations) < 2:
        return (
            OptimizerTask(
                artifact,
                tuple(binding.name for binding in bindings),
                tuple(binding.name for binding in bindings if binding.mutable),
            ),
        )

    operation_ids = {node: index for index, node in enumerate(operations)}
    parent = list(range(len(operations)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    placeholder_dependencies: dict[torch.fx.Node, set[int]] = {
        placeholder: set() for placeholder in placeholders
    }
    dependencies_by_operation: dict[torch.fx.Node, set[torch.fx.Node]] = {}
    for operation in operations:
        dependencies: set[torch.fx.Node] = set()
        stack = list(operation.all_input_nodes)
        while stack:
            dependency = stack.pop()
            if dependency in dependencies:
                continue
            dependencies.add(dependency)
            if dependency in operation_ids:
                union(operation_ids[operation], operation_ids[dependency])
            if dependency.op == "placeholder":
                placeholder_dependencies[dependency].add(operation_ids[operation])
            else:
                stack.extend(dependency.all_input_nodes)
        dependencies_by_operation[operation] = dependencies

    for placeholder, binding in zip(placeholders, bindings, strict=True):
        if not binding.mutable:
            continue
        consumers = sorted(placeholder_dependencies[placeholder])
        for consumer in consumers[1:]:
            union(consumers[0], consumer)

    components: dict[int, list[torch.fx.Node]] = {}
    for operation in operations:
        components.setdefault(find(operation_ids[operation]), []).append(operation)
    ordered_components = tuple(
        tuple(nodes)
        for _root, nodes in sorted(
            components.items(),
            key=lambda item: min(operation_ids[node] for node in item[1]),
        )
    )
    if len(ordered_components) == 1:
        return (
            OptimizerTask(
                artifact,
                tuple(binding.name for binding in bindings),
                tuple(binding.name for binding in bindings if binding.mutable),
            ),
        )

    tasks: list[OptimizerTask] = []
    for component in ordered_components:
        required_placeholders = {
            dependency
            for operation in component
            for dependency in dependencies_by_operation[operation]
            if dependency.op == "placeholder"
        }
        positions = tuple(
            index
            for index, placeholder in enumerate(placeholders)
            if placeholder in required_placeholders
        )
        component_graph = torch.fx.Graph()
        environment: dict[torch.fx.Node, torch.fx.Node] = {}
        for position in positions:
            original = placeholders[position]
            environment[original] = component_graph.placeholder(str(original.target))
        for operation in component:
            environment[operation] = component_graph.node_copy(
                operation, environment.__getitem__
            )
        mutable_positions = tuple(
            position for position in positions if bindings[position].mutable
        )
        component_graph.output(
            tuple(environment[placeholders[position]] for position in mutable_positions)
        )
        component_module = GraphModule(graph_module, component_graph)
        component_artifact = GraphArtifact.capture(
            kind="optimizer",
            graph_module=component_module,
            example_inputs=tuple(
                artifact.example_arguments[position] for position in positions
            ),
        )
        tasks.append(
            OptimizerTask(
                component_artifact,
                tuple(bindings[position].name for position in positions),
                tuple(bindings[position].name for position in mutable_positions),
            )
        )
    return tuple(tasks)


def _restore_binding_values(
    bindings: tuple[OptimizerTensorBinding, ...], snapshots: Mapping[int, torch.Tensor]
) -> None:
    with torch.no_grad():
        for binding in bindings:
            binding.tensor.copy_(snapshots[id(binding.tensor)])
