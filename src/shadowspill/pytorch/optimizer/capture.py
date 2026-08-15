"""Optimizer-agnostic capture with an explicit bounded opaque fallback."""

from __future__ import annotations

import copy
import inspect
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule

from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    TaskInputProvenance,
    TaskInputRole,
)
from shadowspill.pytorch.contracts import CaptureError

from .artifacts import (
    OpaqueOptimizerArtifact,
    OptimizerCapture,
    OptimizerTask,
    OptimizerTensorBinding,
    OptimizerTensorRole,
)
from .initialization import initialize_lazy_optimizer_state


@dataclass(frozen=True, slots=True)
class _OptimizerInventory:
    optimizer_type: str
    canonical_parameters: Mapping[str, torch.nn.Parameter]
    actual_parameters: tuple[torch.nn.Parameter, ...]


@dataclass(slots=True)
class _OptimizerDiscovery:
    optimizer_type: str
    sandbox: torch.optim.Optimizer
    name_by_sandbox_id: dict[int, str]
    first_step_is_opaque: bool
    created_state_names: tuple[str, ...]
    initialized_state_dict: dict[str, Any] | None
    representative_values: dict[str, torch.Tensor]
    initial_sandbox: torch.optim.Optimizer
    initial_parameter_names: dict[int, str]


@dataclass(frozen=True, slots=True)
class _DiscoveryBaseline:
    state_structure: object
    state_dict: dict[str, Any]
    parameters: tuple[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor | None], ...]


@dataclass(frozen=True, slots=True)
class _OptimizerComponent:
    nodes: tuple[torch.fx.Node, ...]
    input_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _OptimizerComponentGroup:
    completion_stage: int | None
    components: tuple[_OptimizerComponent, ...]


class _DisjointSets:
    """Minimal union-find used to find independent optimizer updates."""

    def __init__(self, size: int) -> None:
        self._parents = list(range(size))

    def find(self, index: int) -> int:
        while self._parents[index] != index:
            self._parents[index] = self._parents[self._parents[index]]
            index = self._parents[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parents[right_root] = left_root


def capture_optimizer(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None = None,
) -> OptimizerCapture:
    """Capture a recurrent tensor update without mutating the caller's state.

    Lazy Python/tensor state is discovered on a deep-copied optimizer. Its first
    semantic update remains an ordinary bounded optimizer task. Once state is
    stable, a lifted tensor-only graph is used when Dynamo can represent it;
    otherwise all steps remain bounded opaque tasks with measured workspace.
    """

    inventory = _validate_optimizer_inputs(named_parameters, optimizer)
    discovery = _discover_optimizer_state(inventory, optimizer)
    if isinstance(discovery, OptimizerCapture):
        return discovery
    preinitialized_state_names: tuple[str, ...] = ()
    if (
        discovery.created_state_names
        and initialize_lazy_optimizer_state(
            inventory.canonical_parameters,
            optimizer,
            discovery.created_state_names,
        )
    ):
        preinitialized_state_names = discovery.created_state_names
        discovery = _discover_optimizer_state(inventory, optimizer)
        if isinstance(discovery, OptimizerCapture):
            return replace(
                discovery,
                preinitialized_state_names=preinitialized_state_names,
            )
    captured = _capture_recurrent_optimizer(
        discovery,
        optimizer,
        parameter_stage_owners=parameter_stage_owners,
    )
    if discovery.created_state_names:
        spillable_names = {
            binding.name for binding in captured.bindings if binding.spillable
        }
        initial_bindings = _tensor_bindings(
            discovery.initial_sandbox,
            discovery.initial_parameter_names,
        )
        initial = OpaqueOptimizerArtifact.capture(
            discovery.initial_sandbox,
            initial_bindings,
            profile_output_names=tuple(
                name
                for name in discovery.created_state_names
                if name in spillable_names
            ),
        )
        captured = replace(captured, initial=initial)
    if preinitialized_state_names:
        captured = replace(
            captured,
            preinitialized_state_names=preinitialized_state_names,
        )
    return captured


def _validate_optimizer_inputs(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> _OptimizerInventory:
    """Validate optimizer coverage without changing model or optimizer state."""

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
    return _OptimizerInventory(optimizer_type, canonical, actual_parameters)


def _discover_optimizer_state(
    inventory: _OptimizerInventory,
    optimizer: torch.optim.Optimizer,
) -> _OptimizerDiscovery | OptimizerCapture:
    """Discover lazy tensor state on a storage-free optimizer copy."""

    copied = _copy_discovery_sandbox(inventory, optimizer)
    if isinstance(copied, OptimizerCapture):
        return copied
    sandbox, sandbox_parameters, names = copied
    actual_names = {
        id(parameter): name
        for name, parameter in inventory.canonical_parameters.items()
    }
    representative_values = _representative_optimizer_values(
        optimizer,
        actual_names,
    )
    _seed_discovery_gradients(
        inventory.actual_parameters,
        sandbox_parameters,
    )
    baseline = _discovery_baseline(sandbox, sandbox_parameters, names)
    initial_sandbox = _copy_optimizer(sandbox)
    initial_parameters = _optimizer_parameters(initial_sandbox)
    for source, copied_parameter in zip(
        sandbox_parameters,
        initial_parameters,
        strict=True,
    ):
        if source.grad is not None:
            copied_parameter.grad = source.grad.detach().clone()
    initial_parameter_names = {
        id(copied): names[id(source)]
        for source, copied in zip(
            sandbox_parameters,
            initial_parameters,
            strict=True,
        )
    }
    step = _run_discovery_step(
        inventory.optimizer_type,
        sandbox,
        names,
        baseline,
    )
    if isinstance(step, OptimizerCapture):
        return step
    sandbox, names, initialized_state, discovered_values = step
    representative_values.update(discovered_values)
    return _finish_optimizer_discovery(
        inventory,
        optimizer,
        sandbox,
        names,
        baseline,
        initialized_state,
        representative_values,
        initial_sandbox,
        initial_parameter_names,
    )


def _copy_discovery_sandbox(
    inventory: _OptimizerInventory,
    optimizer: torch.optim.Optimizer,
) -> (
    tuple[
        torch.optim.Optimizer,
        tuple[torch.nn.Parameter, ...],
        dict[int, str],
    ]
    | OptimizerCapture
):
    try:
        actual_names = {
            id(parameter): name
            for name, parameter in inventory.canonical_parameters.items()
        }
        sandbox, names = _copy_optimizer_to_meta(optimizer, actual_names)
    except BaseException as exc:
        return _empty_opaque_capture(
            inventory.optimizer_type,
            f"optimizer cannot be copied for capture: {exc}",
        )
    parameters = _optimizer_parameters(sandbox)
    if len(parameters) != len(inventory.actual_parameters):
        raise CaptureError("copied optimizer changed its parameter inventory")
    return sandbox, parameters, names


def _seed_discovery_gradients(
    actual_parameters: tuple[torch.nn.Parameter, ...],
    sandbox_parameters: tuple[torch.nn.Parameter, ...],
) -> None:
    for _actual, sandbox in zip(
        actual_parameters,
        sandbox_parameters,
        strict=True,
    ):
        if not sandbox.requires_grad or sandbox.grad is not None:
            continue
        # State discovery depends on gradient presence and geometry, not its
        # numerical payload.  Cloning the caller's real gradient here would
        # both allocate its bytes and cross from CPU to the fake CUDA device.
        # Representative profiling values are collected independently from
        # the caller before this storage-free sandbox is stepped.
        sandbox.grad = torch.ones_like(sandbox)


def _discovery_baseline(
    sandbox: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    names: Mapping[int, str],
) -> _DiscoveryBaseline:
    snapshots = tuple(
        (
            parameter,
            parameter.detach().clone(),
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for parameter in parameters
    )
    return _DiscoveryBaseline(
        _state_structure(sandbox, names),
        copy.deepcopy(sandbox.state_dict()),
        snapshots,
    )


def _run_discovery_step(
    optimizer_type: str,
    sandbox: torch.optim.Optimizer,
    names: dict[int, str],
    baseline: _DiscoveryBaseline,
) -> (
    tuple[
        torch.optim.Optimizer,
        dict[int, str],
        dict[str, Any] | None,
        dict[str, torch.Tensor],
    ]
    | OptimizerCapture
):
    try:
        # Discovery is not a semantic optimizer step.  Invoke the unwrapped
        # implementation so user/global step hooks remain reserved for real
        # callable invocations while the sandbox still exposes lazy state.
        step = inspect.unwrap(type(sandbox).step).__get__(sandbox, type(sandbox))
        with torch.no_grad():
            step()
    except BaseException as exc:
        return _recover_failed_discovery(
            optimizer_type,
            sandbox,
            names,
            baseline,
            exc,
        )
    return sandbox, names, None, {}


def _recover_failed_discovery(
    optimizer_type: str,
    sandbox: torch.optim.Optimizer,
    names: dict[int, str],
    baseline: _DiscoveryBaseline,
    failure: BaseException,
) -> (
    tuple[
        torch.optim.Optimizer,
        dict[int, str],
        dict[str, Any] | None,
        dict[str, torch.Tensor],
    ]
    | OptimizerCapture
):
    if _state_structure(sandbox, names) == baseline.state_structure:
        # A data-dependent but stateless optimizer can fail symbolic execution
        # without changing its tensor inventory.  Preserve the symbolic
        # sandbox so graph export can classify it as a bounded opaque task.
        if _is_data_dependent_failure(failure):
            return sandbox, names, None, {}
        return _empty_opaque_capture(
            optimizer_type,
            f"optimizer discovery step failed: {failure}",
        )
    if any(
        isinstance(parameter, FakeTensor)
        for parameter in _optimizer_parameters(sandbox)
    ):
        return _empty_opaque_capture(
            optimizer_type,
            "storage-free optimizer discovery step failed after changing its "
            f"tensor inventory: {failure}",
        )
    try:
        _complete_failed_state_discovery(sandbox, names, baseline.parameters)
        state = sandbox.state_dict()
        representative = _representative_optimizer_values(sandbox, names)
        fake_sandbox, fake_names = _fake_cuda_optimizer(sandbox, names)
        return fake_sandbox, fake_names, state, representative
    except BaseException as fake_failure:
        return _empty_opaque_capture(
            optimizer_type,
            (
                f"optimizer discovery step failed: {failure}; "
                "optimizer must provide valid fake/meta behavior for lazy "
                f"state discovery: {fake_failure}"
            ),
        )


def _finish_optimizer_discovery(
    inventory: _OptimizerInventory,
    optimizer: torch.optim.Optimizer,
    sandbox: torch.optim.Optimizer,
    names: dict[int, str],
    baseline: _DiscoveryBaseline,
    initialized_state: dict[str, Any] | None,
    representative_values: dict[str, torch.Tensor],
    initial_sandbox: torch.optim.Optimizer,
    initial_parameter_names: dict[int, str],
) -> _OptimizerDiscovery:
    first_step_is_opaque = _state_structure(sandbox, names) != baseline.state_structure
    actual_names = {
        id(parameter): name
        for name, parameter in inventory.canonical_parameters.items()
    }
    before_state_names = _state_tensor_names(optimizer, actual_names)
    after_state_names = _state_tensor_names(sandbox, names)
    created_state_names = tuple(
        sorted(set(after_state_names) - set(before_state_names))
    )
    if not first_step_is_opaque:
        _restore_discovery_baseline(sandbox, baseline)
    return _OptimizerDiscovery(
        optimizer_type=inventory.optimizer_type,
        sandbox=sandbox,
        name_by_sandbox_id=names,
        first_step_is_opaque=first_step_is_opaque,
        created_state_names=created_state_names,
        initialized_state_dict=initialized_state,
        representative_values=representative_values,
        initial_sandbox=initial_sandbox,
        initial_parameter_names=initial_parameter_names,
    )


def _restore_discovery_baseline(
    sandbox: torch.optim.Optimizer,
    baseline: _DiscoveryBaseline,
) -> None:
    sandbox.load_state_dict(baseline.state_dict)
    with torch.no_grad():
        for parameter, value, gradient in baseline.parameters:
            parameter.copy_(value)
            if gradient is not None and parameter.grad is not None:
                parameter.grad.copy_(gradient)


def _capture_recurrent_optimizer(
    discovery: _OptimizerDiscovery,
    optimizer: torch.optim.Optimizer,
    *,
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None,
) -> OptimizerCapture:
    """Capture the stable recurrent update or publish a bounded opaque task."""

    if _has_optimizer_step_hooks(optimizer):
        return _hooked_optimizer_capture(discovery)
    opaque = _prepare_recurrent_sandbox(discovery)
    if opaque is not None:
        return opaque
    captured = _capture_optimizer_artifact(discovery)
    if isinstance(captured, OptimizerCapture):
        return captured
    artifact, bindings = captured
    recurrent_tasks = _partition_optimizer_graph(
        artifact,
        bindings,
        parameter_stage_owners=parameter_stage_owners,
    )
    return OptimizerCapture(
        optimizer_type=discovery.optimizer_type,
        first_step_is_opaque=discovery.first_step_is_opaque,
        created_state_names=discovery.created_state_names,
        initial=None,
        recurrent=artifact,
        recurrent_tasks=recurrent_tasks,
        bindings=bindings,
        mutation_names=tuple(binding.name for binding in bindings if binding.mutable),
        initialized_state_dict=discovery.initialized_state_dict,
    )


def _hooked_optimizer_capture(
    discovery: _OptimizerDiscovery,
) -> OptimizerCapture:
    bindings = _tensor_bindings(
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


def _prepare_recurrent_sandbox(
    discovery: _OptimizerDiscovery,
) -> OptimizerCapture | None:
    if discovery.initialized_state_dict is None:
        sandbox = discovery.sandbox
        names = discovery.name_by_sandbox_id
        probe_bindings = _tensor_bindings(sandbox, names)
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
            return _opaque_optimizer_capture(
                discovery,
                opaque_artifact,
                probe_bindings,
                reason=_opaque_optimizer_reason(exc),
            )
        finally:
            torch.set_grad_enabled(probe_grad_enabled)
        _restore_binding_values(probe_bindings, probe_snapshots)
        discovery.representative_values.update(
            _representative_optimizer_values(sandbox, names)
        )
        sandbox, names = _fake_cuda_optimizer(sandbox, names)
        discovery.sandbox = sandbox
        discovery.name_by_sandbox_id = names
    return None


def _opaque_optimizer_reason(failure: BaseException) -> str:
    description = str(failure)
    if _is_data_dependent_failure(failure):
        return f"recurrent optimizer graph is data-dependent: {description}"
    return f"recurrent optimizer graph is opaque: {description}"


def _is_data_dependent_failure(failure: BaseException) -> bool:
    description = str(failure)
    return "_local_scalar_dense" in description or "Tensor.item" in description


def _capture_optimizer_artifact(
    discovery: _OptimizerDiscovery,
) -> tuple[GraphArtifact, tuple[OptimizerTensorBinding, ...]] | OptimizerCapture:
    sandbox = discovery.sandbox
    names = discovery.name_by_sandbox_id
    bindings = _tensor_bindings(sandbox, names)
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
            input_provenance=_optimizer_input_provenance(
                bindings,
                discovery.representative_values,
            ),
        )
    except BaseException as exc:
        _restore_binding_values(bindings, snapshots)
        opaque_artifact = OpaqueOptimizerArtifact.capture(sandbox, bindings)
        return _opaque_optimizer_capture(
            discovery,
            opaque_artifact,
            bindings,
            reason=_opaque_optimizer_reason(exc),
        )
    finally:
        torch.set_grad_enabled(grad_enabled)
    return artifact, bindings


def _opaque_optimizer_capture(
    discovery: _OptimizerDiscovery,
    artifact: OpaqueOptimizerArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
    *,
    reason: str,
) -> OptimizerCapture:
    mutations = tuple(binding.name for binding in bindings if binding.mutable)
    return OptimizerCapture(
        optimizer_type=discovery.optimizer_type,
        first_step_is_opaque=discovery.first_step_is_opaque,
        created_state_names=discovery.created_state_names,
        initial=None,
        recurrent=artifact,
        recurrent_tasks=(
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


def _empty_opaque_capture(optimizer_type: str, reason: str) -> OptimizerCapture:
    return OptimizerCapture(
        optimizer_type=optimizer_type,
        first_step_is_opaque=True,
        created_state_names=(),
        initial=None,
        recurrent=None,
        recurrent_tasks=(),
        bindings=(),
        mutation_names=(),
        opaque_reason=reason,
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


def _copy_optimizer_to_meta(
    optimizer: torch.optim.Optimizer,
    name_by_id: Mapping[int, str],
) -> tuple[torch.optim.Optimizer, dict[int, str]]:
    """Copy optimizer structure while replacing payload tensors by meta geometry.

    Optimizer discovery needs Python control flow, state names, and tensor
    geometry; it does not need parameter bytes.  A normal ``deepcopy`` scales
    with the complete model and can transiently duplicate parameters,
    gradients, snapshots, and lazy state.  Pre-populating ``deepcopy``'s memo
    with meta views retains the subclass structure without allocating any of
    those payloads.
    """

    parameters = _optimizer_parameters(optimizer)
    parameter_ids = {id(parameter) for parameter in parameters}
    tensors = _optimizer_protocol_tensors(optimizer)
    owners: dict[tuple[str, int], torch.Tensor] = {}
    replacements: dict[int, torch.Tensor] = {}
    for tensor in tensors:
        if tensor.layout is not torch.strided:
            raise CaptureError("optimizer meta capture requires strided tensors")
        if (
            id(tensor) not in parameter_ids
            and tensor.device.type == "cpu"
            and tensor.ndim == 0
        ):
            # Non-capturable torch.optim implementations intentionally keep
            # their scalar step counter on CPU and inspect its value in Python.
            # Retaining those few bytes follows the same control flow without
            # allocating parameter or optimizer-state payloads.
            continue
        storage = tensor.untyped_storage()
        key = (tensor.device.type, int(storage._cdata))
        owner = owners.get(key)
        if owner is None:
            owner = torch.empty(
                int(storage.nbytes()),
                dtype=torch.uint8,
                device="meta",
            )
            owners[key] = owner
        replacement = torch.empty(
            0,
            dtype=tensor.dtype,
            device="meta",
        ).set_(
            owner.untyped_storage(),
            int(tensor.storage_offset()),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        replacement.requires_grad_(bool(tensor.requires_grad))
        if id(tensor) in parameter_ids:
            replacement = torch.nn.Parameter(
                replacement,
                requires_grad=bool(tensor.requires_grad),
            )
        replacements[id(tensor)] = replacement

    copied = copy.deepcopy(optimizer, dict(replacements))
    if copied.__dict__.keys() != optimizer.__dict__.keys():
        copied = object.__new__(type(optimizer))
        copied.__dict__ = copy.deepcopy(optimizer.__dict__, dict(replacements))
    if not isinstance(copied, torch.optim.Optimizer):
        raise TypeError("copied optimizer changed its base type")
    copied_parameters = _optimizer_parameters(copied)
    fake_names = {
        id(copied_parameter): name_by_id[id(actual_parameter)]
        for actual_parameter, copied_parameter in zip(
            parameters,
            copied_parameters,
            strict=True,
        )
        if id(actual_parameter) in name_by_id
    }
    return copied, fake_names


def _optimizer_protocol_tensors(
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, ...]:
    """Return tensor leaves reachable through the optimizer's public state."""

    result: list[torch.Tensor] = []
    seen_containers: set[int] = set()
    seen_tensors: set[int] = set()

    def visit(value: object) -> None:
        if isinstance(value, torch.Tensor):
            if id(value) not in seen_tensors:
                seen_tensors.add(id(value))
                result.append(value)
            return
        identity = id(value)
        if identity in seen_containers:
            return
        seen_containers.add(identity)
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, tuple | list | set | frozenset):
            for item in value:
                visit(item)

    visit(optimizer.__dict__)
    return tuple(result)


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
    if all(isinstance(parameter, FakeTensor) for parameter in parameters):
        return optimizer, dict(name_by_id)
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

    source_parameters = _optimizer_parameters(artifact.optimizer)
    optimizer = _copy_optimizer(artifact.optimizer)
    parameters = _optimizer_parameters(optimizer)
    if len(parameters) != len(source_parameters):
        raise CaptureError("copied opaque optimizer changed its parameter inventory")
    replacements: dict[int, torch.Tensor] = {}
    device = torch.device("cuda", device_ordinal)

    def real_tensor(
        value: torch.Tensor,
        *,
        parameter: bool = False,
        synthetic_fill: str = "zero",
    ) -> torch.Tensor:
        existing = replacements.get(id(value))
        if existing is not None:
            return existing
        if value.layout is not torch.strided:
            raise CaptureError("opaque optimizer profiling requires strided tensors")
        symbolic = isinstance(value, FakeTensor) or value.device.type == "meta"
        if not symbolic and value.device.type == "cpu" and value.ndim == 0:
            scalar_copy = value.detach().clone()
            replacements[id(value)] = scalar_copy
            return scalar_copy
        with torch.no_grad():
            raw = torch.empty_strided(
                tuple(value.shape),
                tuple(value.stride()),
                dtype=value.dtype,
                device=device,
            )
            if symbolic:
                if synthetic_fill == "normal" and (
                    value.dtype.is_floating_point or value.dtype.is_complex
                ):
                    raw.normal_()
                else:
                    raw.zero_()
            else:
                raw.copy_(value)
            result: torch.Tensor
            if parameter:
                result = torch.nn.Parameter(raw, requires_grad=value.requires_grad)
            else:
                result = raw.requires_grad_(value.requires_grad)
        replacements[id(value)] = result
        return result

    real_parameters: dict[int, torch.nn.Parameter] = {}
    for source_value, value in zip(source_parameters, parameters, strict=True):
        converted = real_tensor(
            value,
            parameter=True,
            synthetic_fill="normal",
        )
        if not isinstance(converted, torch.nn.Parameter):
            raise AssertionError("parameter conversion changed tensor type")
        # ``deepcopy(Parameter)`` intentionally drops ``.grad``.  The opaque
        # optimizer would otherwise profile a no-op despite the captured task
        # requiring gradients.  Recover the storage-free captured gradient
        # from the source optimizer and materialize a representative value.
        if source_value.grad is not None:
            converted.grad = real_tensor(
                source_value.grad,
                synthetic_fill="normal",
            )
        real_parameters[id(value)] = converted
    for group in optimizer.param_groups:
        group["params"] = [real_parameters[id(value)] for value in group["params"]]

    converted_state: defaultdict[torch.Tensor, dict[str, Any]] = defaultdict(dict)
    for parameter, value in optimizer.state.items():
        real_parameter = real_parameters.get(id(parameter))
        if real_parameter is None:
            raise CaptureError("optimizer state is keyed by an unknown parameter")
        converted = _map_optimizer_tensors(
            value,
            lambda tensor: real_tensor(tensor, synthetic_fill="zero"),
        )
        if not isinstance(converted, dict):
            raise CaptureError("per-parameter optimizer state must be a mapping")
        converted_state[real_parameter] = converted
    optimizer.state = converted_state
    return optimizer


def opaque_optimizer_outputs(
    artifact: OpaqueOptimizerArtifact,
    optimizer: torch.optim.Optimizer,
    *,
    device_ordinal: int,
) -> tuple[OptimizerTensorBinding, ...]:
    """Expose first-step state using one profiling/execution storage policy.

    An opaque optimizer does not return its lazily created state from
    ``step()``.  The initial structural profile nevertheless needs those
    tensors as explicit persistent outputs so allocator ordinals can be
    reconciled with Program objects.  Names come from optimizer discovery;
    values come only from the real isolated first step.
    """

    parameters = _optimizer_parameters(optimizer)
    if len(parameters) != len(artifact.parameter_names):
        raise CaptureError("opaque optimizer changed its parameter inventory")
    names = {
        id(parameter): name
        for parameter, name in zip(
            parameters,
            artifact.parameter_names,
            strict=True,
        )
        if name is not None
    }
    by_name = {
        binding.name: binding
        for binding in _tensor_bindings(
            optimizer,
            names,
            require_gradients=False,
        )
    }
    outputs: list[OptimizerTensorBinding] = []
    target = torch.device("cuda", device_ordinal)
    for name in artifact.profile_output_names:
        binding = by_name.get(name)
        if binding is None:
            raise CaptureError(
                f"opaque optimizer did not create profiled state {name!r}"
            )
        tensor = binding.tensor
        if not binding.spillable:
            raise CaptureError(
                f"opaque optimizer output {name!r} is not spillable"
            )
        if tensor.device.type == "cpu":
            owner = torch.empty(
                tensor.untyped_storage().nbytes(),
                dtype=torch.uint8,
                device=target,
            )
            relocated = torch.empty(0, dtype=tensor.dtype, device=target).set_(
                owner.untyped_storage(),
                int(tensor.storage_offset()),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            relocated.copy_(tensor)
            tensor.data = relocated
        elif tensor.device != target:
            raise CaptureError(
                f"opaque optimizer output {name!r} was created on {tensor.device}, "
                f"expected cpu or {target}"
            )
        outputs.append(binding)
    return tuple(outputs)


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
    *,
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[OptimizerTask, ...]:
    """Partition dependency-closed updates at their backward-ready frontier."""

    placeholders, operations = _optimizer_graph_nodes(artifact, bindings)
    if len(operations) < 2:
        return (_whole_optimizer_task(artifact, bindings, parameter_stage_owners),)
    components = _optimizer_components(placeholders, operations, bindings)
    if len(components) == 1:
        return (_whole_optimizer_task(artifact, bindings, parameter_stage_owners),)
    groups = _group_optimizer_components(components, bindings, parameter_stage_owners)
    return tuple(
        _build_optimizer_component_task(
            artifact, bindings, placeholders, operations, group
        )
        for group in groups
    )


def _optimizer_graph_nodes(
    artifact: GraphArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
) -> tuple[tuple[torch.fx.Node, ...], tuple[torch.fx.Node, ...]]:
    placeholders = tuple(
        node for node in artifact.graph_module.graph.nodes if node.op == "placeholder"
    )
    if len(placeholders) != len(bindings):
        raise CaptureError("optimizer placeholder inventory changed after lifting")
    operations = tuple(
        node
        for node in artifact.graph_module.graph.nodes
        if node.op not in {"placeholder", "output"}
    )
    return placeholders, operations


def _whole_optimizer_task(
    artifact: GraphArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None,
) -> OptimizerTask:
    return OptimizerTask(
        artifact,
        tuple(binding.name for binding in bindings),
        tuple(binding.name for binding in bindings if binding.mutable),
        _completion_stage(bindings, parameter_stage_owners),
    )


def _optimizer_components(
    placeholders: tuple[torch.fx.Node, ...],
    operations: tuple[torch.fx.Node, ...],
    bindings: tuple[OptimizerTensorBinding, ...],
) -> tuple[_OptimizerComponent, ...]:
    positions = {node: index for index, node in enumerate(operations)}
    sets = _DisjointSets(len(operations))
    dependencies = _optimizer_dependencies(operations, positions, sets)
    _join_mutable_consumers(placeholders, bindings, positions, dependencies, sets)
    component_nodes = _ordered_component_nodes(operations, positions, sets)
    return tuple(
        _OptimizerComponent(
            nodes=nodes,
            input_positions=_component_input_positions(
                nodes, placeholders, dependencies
            ),
        )
        for nodes in component_nodes
    )


def _optimizer_dependencies(
    operations: tuple[torch.fx.Node, ...],
    positions: Mapping[torch.fx.Node, int],
    sets: _DisjointSets,
) -> dict[torch.fx.Node, set[torch.fx.Node]]:
    dependencies_by_operation: dict[torch.fx.Node, set[torch.fx.Node]] = {}
    for operation in operations:
        dependencies: set[torch.fx.Node] = set()
        stack = list(operation.all_input_nodes)
        while stack:
            dependency = stack.pop()
            if dependency in dependencies:
                continue
            dependencies.add(dependency)
            if dependency in positions:
                sets.union(positions[operation], positions[dependency])
            if dependency.op != "placeholder":
                stack.extend(dependency.all_input_nodes)
        dependencies_by_operation[operation] = dependencies
    return dependencies_by_operation


def _join_mutable_consumers(
    placeholders: tuple[torch.fx.Node, ...],
    bindings: tuple[OptimizerTensorBinding, ...],
    positions: Mapping[torch.fx.Node, int],
    dependencies: Mapping[torch.fx.Node, set[torch.fx.Node]],
    sets: _DisjointSets,
) -> None:
    for placeholder, binding in zip(placeholders, bindings, strict=True):
        if not binding.mutable:
            continue
        consumers = sorted(
            positions[operation]
            for operation, required in dependencies.items()
            if placeholder in required
        )
        for consumer in consumers[1:]:
            sets.union(consumers[0], consumer)


def _ordered_component_nodes(
    operations: tuple[torch.fx.Node, ...],
    positions: Mapping[torch.fx.Node, int],
    sets: _DisjointSets,
) -> tuple[tuple[torch.fx.Node, ...], ...]:
    components: dict[int, list[torch.fx.Node]] = {}
    for operation in operations:
        components.setdefault(sets.find(positions[operation]), []).append(operation)
    return tuple(
        tuple(nodes)
        for _root, nodes in sorted(
            components.items(),
            key=lambda item: min(positions[node] for node in item[1]),
        )
    )


def _component_input_positions(
    nodes: tuple[torch.fx.Node, ...],
    placeholders: tuple[torch.fx.Node, ...],
    dependencies: Mapping[torch.fx.Node, set[torch.fx.Node]],
) -> tuple[int, ...]:
    required = {
        dependency
        for operation in nodes
        for dependency in dependencies[operation]
        if dependency.op == "placeholder"
    }
    return tuple(
        index
        for index, placeholder in enumerate(placeholders)
        if placeholder in required
    )


def _group_optimizer_components(
    components: tuple[_OptimizerComponent, ...],
    bindings: tuple[OptimizerTensorBinding, ...],
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None,
) -> tuple[_OptimizerComponentGroup, ...]:
    if parameter_stage_owners is None:
        return tuple(
            _OptimizerComponentGroup(None, (component,)) for component in components
        )
    grouped: dict[int | None, list[_OptimizerComponent]] = defaultdict(list)
    for component in components:
        bound = tuple(bindings[index] for index in component.input_positions)
        grouped[_completion_stage(bound, parameter_stage_owners)].append(component)
    return tuple(
        _OptimizerComponentGroup(completion_stage, tuple(members))
        for completion_stage, members in sorted(
            grouped.items(),
            key=lambda item: (
                item[0] is None,
                -(item[0] if item[0] is not None else -1),
            ),
        )
    )


def _build_optimizer_component_task(
    artifact: GraphArtifact,
    bindings: tuple[OptimizerTensorBinding, ...],
    placeholders: tuple[torch.fx.Node, ...],
    operations: tuple[torch.fx.Node, ...],
    group: _OptimizerComponentGroup,
) -> OptimizerTask:
    positions = tuple(
        sorted(
            {
                position
                for component in group.components
                for position in component.input_positions
            }
        )
    )
    component_nodes = {
        node for component in group.components for node in component.nodes
    }
    graph = _copy_optimizer_component_graph(
        placeholders, operations, positions, component_nodes, bindings
    )
    component_artifact = GraphArtifact.capture(
        kind="optimizer",
        graph_module=GraphModule(artifact.graph_module, graph),
        example_inputs=tuple(
            artifact.example_arguments[position] for position in positions
        ),
        input_provenance=tuple(
            artifact.input_provenance[position] for position in positions
        ),
    )
    mutable_positions = tuple(
        position for position in positions if bindings[position].mutable
    )
    return OptimizerTask(
        component_artifact,
        tuple(bindings[position].name for position in positions),
        tuple(bindings[position].name for position in mutable_positions),
        group.completion_stage,
    )


def _copy_optimizer_component_graph(
    placeholders: tuple[torch.fx.Node, ...],
    operations: tuple[torch.fx.Node, ...],
    positions: tuple[int, ...],
    component_nodes: set[torch.fx.Node],
    bindings: tuple[OptimizerTensorBinding, ...],
) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    environment: dict[torch.fx.Node, torch.fx.Node] = {}
    for local_index, position in enumerate(positions):
        environment[placeholders[position]] = graph.placeholder(
            f"optimizer_tensor_{local_index:04d}"
        )
    for operation in operations:
        if operation not in component_nodes:
            continue
        copied = graph.create_node(
            operation.op,
            operation.target,
            torch.fx.map_arg(operation.args, environment.__getitem__),
            torch.fx.map_arg(operation.kwargs, environment.__getitem__),
            type_expr=operation.type,
        )
        copied.meta = copy.copy(operation.meta)
        environment[operation] = copied
    mutable_positions = tuple(
        position for position in positions if bindings[position].mutable
    )
    graph.output(
        tuple(environment[placeholders[position]] for position in mutable_positions)
    )
    return graph


def _representative_optimizer_values(
    optimizer: torch.optim.Optimizer,
    name_by_id: Mapping[int, str],
) -> dict[str, torch.Tensor]:
    """Retain occurrence-local initialized values before symbolic conversion."""

    return {
        binding.name: binding.tensor.detach()
        for binding in _tensor_bindings(
            optimizer,
            name_by_id,
            require_gradients=False,
        )
        if binding.role is not OptimizerTensorRole.GRADIENT
        and not isinstance(binding.tensor, FakeTensor)
        and binding.tensor.device.type != "meta"
    }


def _optimizer_input_provenance(
    bindings: tuple[OptimizerTensorBinding, ...],
    representative_values: Mapping[str, torch.Tensor],
) -> tuple[TaskInputProvenance, ...]:
    role_map = {
        OptimizerTensorRole.PARAMETER: TaskInputRole.PARAMETER,
        OptimizerTensorRole.GRADIENT: TaskInputRole.GRADIENT,
        OptimizerTensorRole.STATE: TaskInputRole.OPTIMIZER_STATE,
        OptimizerTensorRole.HYPERPARAMETER: (TaskInputRole.OPTIMIZER_HYPERPARAMETER),
    }
    return tuple(
        TaskInputProvenance(
            role_map[binding.role],
            binding.name,
            representative_value=representative_values.get(binding.name),
        )
        for binding in bindings
    )


def _completion_stage(
    bindings: tuple[OptimizerTensorBinding, ...],
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None,
) -> int | None:
    """Return the backward frontier after which all bound gradients are final."""

    if parameter_stage_owners is None:
        return None
    stages = {
        stage
        for binding in bindings
        if binding.role is OptimizerTensorRole.PARAMETER
        for stage in parameter_stage_owners.get(binding.name, ())
    }
    return min(stages) if stages else None


def _restore_binding_values(
    bindings: tuple[OptimizerTensorBinding, ...], snapshots: Mapping[int, torch.Tensor]
) -> None:
    with torch.no_grad():
        for binding in bindings:
            binding.tensor.copy_(snapshots[id(binding.tensor)])
