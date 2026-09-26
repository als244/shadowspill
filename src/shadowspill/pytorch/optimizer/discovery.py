"""Discovery: the lazy state an optimizer creates on its first step, found on a
storage-free copy, and the sandbox that copy becomes for tracing."""

from __future__ import annotations

import copy
import inspect
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor

from shadowspill.errors import CaptureError

from .artifacts import (
    OptimizerCapture,
)
from .bindings import (
    representative_optimizer_values,
    state_structure,
    state_tensor_names,
)
from .sandbox import (
    canonical_parameters,
    copy_optimizer,
    copy_optimizer_to_meta,
    fake_device_optimizer,
    optimizer_parameters,
)


@dataclass(frozen=True, slots=True)
class OptimizerInventory:
    optimizer_type: str
    canonical_parameters: Mapping[str, torch.nn.Parameter]
    actual_parameters: tuple[torch.nn.Parameter, ...]


@dataclass(slots=True)
class OptimizerDiscovery:
    optimizer_type: str
    sandbox: torch.optim.Optimizer
    name_by_sandbox_id: dict[int, str]
    first_step_is_opaque: bool
    created_state_names: tuple[str, ...]
    initialized_state_dict: dict[str, Any] | None
    representative_values: dict[str, torch.Tensor]
    initial_sandbox: torch.optim.Optimizer
    initial_parameter_names: dict[int, str]
    #: The sandbox before it moved onto fake tensors, and its names: what an
    #: opaque fallback is captured from, since an opaque task keeps real values.
    real_sandbox: torch.optim.Optimizer | None = None
    real_name_by_sandbox_id: dict[int, str] | None = None


@dataclass(frozen=True, slots=True)
class _DiscoveryBaseline:
    state_structure: object
    state_dict: dict[str, Any]
    parameters: tuple[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor | None], ...]


def validate_optimizer_inputs(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> OptimizerInventory:
    """Validate optimizer coverage without changing model or optimizer state."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must derive from torch.optim.Optimizer")
    canonical = canonical_parameters(named_parameters)
    actual_parameters = optimizer_parameters(optimizer)
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
    return OptimizerInventory(optimizer_type, canonical, actual_parameters)


def discover_optimizer_state(
    inventory: OptimizerInventory,
    optimizer: torch.optim.Optimizer,
    *,
    receives_gradient: Collection[str] | None = None,
) -> OptimizerDiscovery | OptimizerCapture:
    """Discover lazy tensor state on a storage-free optimizer copy.

    ``receives_gradient`` names the parameters a captured backward really
    produces a gradient for. Those it leaves out are stepped by nothing, so
    the sandbox is given no gradient for them and the optimizer declares no
    state for them -- which is what eager training does, where a parameter
    whose ``grad`` is ``None`` is skipped entirely. Left unset, every
    parameter that may be trained is treated as trained.
    """

    copied = _copy_discovery_sandbox(inventory, optimizer)
    if isinstance(copied, OptimizerCapture):
        return copied
    sandbox, sandbox_parameters, names = copied
    actual_names = {
        id(parameter): name
        for name, parameter in inventory.canonical_parameters.items()
    }
    representative_values = representative_optimizer_values(
        optimizer,
        actual_names,
    )
    _seed_discovery_gradients(
        inventory.actual_parameters,
        sandbox_parameters,
        skipped=frozenset(
            identity
            for identity, name in actual_names.items()
            if receives_gradient is not None and name not in receives_gradient
        ),
    )
    baseline = _discovery_baseline(sandbox, sandbox_parameters, names)
    initial_sandbox = copy_optimizer(sandbox)
    initial_parameters = optimizer_parameters(initial_sandbox)
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
    inventory: OptimizerInventory,
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
        sandbox, names = copy_optimizer_to_meta(optimizer, actual_names)
    except BaseException as exc:
        return _empty_opaque_capture(
            inventory.optimizer_type,
            f"optimizer cannot be copied for capture: {exc}",
        )
    parameters = optimizer_parameters(sandbox)
    if len(parameters) != len(inventory.actual_parameters):
        raise CaptureError("copied optimizer changed its parameter inventory")
    return sandbox, parameters, names


def _seed_discovery_gradients(
    actual_parameters: tuple[torch.nn.Parameter, ...],
    sandbox_parameters: tuple[torch.nn.Parameter, ...],
    *,
    skipped: frozenset[int] = frozenset(),
) -> None:
    for actual, sandbox in zip(
        actual_parameters,
        sandbox_parameters,
        strict=True,
    ):
        if id(actual) in skipped:
            # Nothing will ever hand this one a gradient, so in the sandbox
            # it is not a trained parameter at all. Saying that with the
            # flag every path already consults keeps them all agreeing: the
            # optimizer declares no state for it, the bindings carry
            # neither it nor a gradient for it, and the check that a
            # captured optimizer was given its gradients still means what
            # it says for the parameters that are trained.
            sandbox.requires_grad_(False)
            continue
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
        state_structure(sandbox, names),
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
    if state_structure(sandbox, names) == baseline.state_structure:
        # A data-dependent but stateless optimizer can fail symbolic execution
        # without changing its tensor inventory.  Preserve the symbolic
        # sandbox so graph export can classify it as a bounded opaque task.
        if is_data_dependent_failure(failure):
            return sandbox, names, None, {}
        return _empty_opaque_capture(
            optimizer_type,
            f"optimizer discovery step failed: {failure}",
        )
    if any(
        isinstance(parameter, FakeTensor) for parameter in optimizer_parameters(sandbox)
    ):
        return _empty_opaque_capture(
            optimizer_type,
            "storage-free optimizer discovery step failed after changing its "
            f"tensor inventory: {failure}",
        )
    try:
        _complete_failed_state_discovery(sandbox, names, baseline.parameters)
        state = sandbox.state_dict()
        representative = representative_optimizer_values(sandbox, names)
        fake_sandbox, fake_names = fake_device_optimizer(sandbox, names)
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
    inventory: OptimizerInventory,
    optimizer: torch.optim.Optimizer,
    sandbox: torch.optim.Optimizer,
    names: dict[int, str],
    baseline: _DiscoveryBaseline,
    initialized_state: dict[str, Any] | None,
    representative_values: dict[str, torch.Tensor],
    initial_sandbox: torch.optim.Optimizer,
    initial_parameter_names: dict[int, str],
) -> OptimizerDiscovery:
    first_step_is_opaque = state_structure(sandbox, names) != baseline.state_structure
    actual_names = {
        id(parameter): name
        for name, parameter in inventory.canonical_parameters.items()
    }
    before_state_names = state_tensor_names(optimizer, actual_names)
    after_state_names = state_tensor_names(sandbox, names)
    created_state_names = tuple(
        sorted(set(after_state_names) - set(before_state_names))
    )
    if not first_step_is_opaque:
        _restore_discovery_baseline(sandbox, baseline)
    return OptimizerDiscovery(
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
    # Not Optimizer.load_state_dict, which casts every floating-point entry
    # but "step" to its parameter's dtype: a master copy or moments kept at
    # another precision would come back at the parameter's, and the update
    # would be captured at the wrong one. The baseline is this sandbox's own
    # state_dict, so its entries go back as they were saved, to the
    # parameters its indices were taken from.
    saved = baseline.state_dict
    parameters = [
        parameter for group in sandbox.param_groups for parameter in group["params"]
    ]
    sandbox.state = defaultdict(
        dict,
        {
            parameters[index]: copy.deepcopy(entries)
            for index, entries in saved["state"].items()
        },
    )
    for group, saved_group in zip(
        sandbox.param_groups, saved["param_groups"], strict=True
    ):
        group.update(
            {
                key: copy.deepcopy(value)
                for key, value in saved_group.items()
                if key != "params"
            }
        )
    with torch.no_grad():
        for parameter, value, gradient in baseline.parameters:
            parameter.copy_(value)
            if gradient is not None and parameter.grad is not None:
                parameter.grad.copy_(gradient)


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

    parameters = optimizer_parameters(optimizer)
    gradients = {id(parameter): parameter.grad for parameter in parameters}
    snapshots_by_id = {
        id(parameter): (parameter, value, gradient)
        for parameter, value, gradient in parameter_snapshots
    }
    previous_structure = state_structure(optimizer, name_by_id)
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
                current_structure = state_structure(optimizer, name_by_id)
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


def fake_recurrent_sandbox(discovery: OptimizerDiscovery) -> None:
    """Move the sandbox onto fake device tensors, keeping its representative values."""

    sandbox = discovery.sandbox
    names = discovery.name_by_sandbox_id
    discovery.representative_values.update(
        representative_optimizer_values(sandbox, names)
    )
    # The fake conversion consumes the sandbox, so what an opaque fallback is
    # captured from is a copy of it; on meta that copy is storage-free. Names
    # follow the parameters, which is all a name map holds.
    real = copy_optimizer(sandbox)
    pairs = tuple(
        zip(optimizer_parameters(sandbox), optimizer_parameters(real), strict=True)
    )
    # A Parameter's deep copy is made from its data alone, so the gradients the
    # discovery seeded are copied across by hand.
    for original, copied in pairs:
        if original.grad is not None and copied.grad is None:
            copied.grad = original.grad.detach().clone()
    discovery.real_sandbox = real
    discovery.real_name_by_sandbox_id = {
        id(copied): names[id(original)]
        for original, copied in pairs
        if id(original) in names
    }
    sandbox, names = fake_device_optimizer(sandbox, names)
    discovery.sandbox = sandbox
    discovery.name_by_sandbox_id = names


def is_data_dependent_failure(failure: BaseException) -> bool:
    description = str(failure)
    return "_local_scalar_dense" in description or "Tensor.item" in description


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
