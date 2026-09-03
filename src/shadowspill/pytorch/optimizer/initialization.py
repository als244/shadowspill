"""Side-effect-free initialization of lazy optimizer tensor state."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Mapping

import torch


class _StopBeforeOptimizerUpdate(Exception):
    """Internal control flow raised by a one-use compiler callable."""


def initialize_lazy_optimizer_state(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    expected_state_names: tuple[str, ...],
) -> bool:
    """Initialize step-zero tensor state without executing an optimizer update.

    Some optimizers expose a per-parameter initializer.  For ordinary PyTorch
    optimizers whose Python preamble creates state before the numerical graph,
    a one-use compiler backend stops at graph entry.  Optimizers that return
    persistent state from the numerical graph are intentionally left for the
    distinct initial/recurrent-plan path.
    """

    if not expected_state_names:
        return True
    names = {id(parameter): name for name, parameter in named_parameters.items()}
    baseline = copy.deepcopy(optimizer.state_dict())
    versions = _parameter_versions(optimizer)
    try:
        initialized = _run_declared_initializer(optimizer) or (
            _initialize_at_compiler_boundary(optimizer, names, expected_state_names)
        )
        if not initialized:
            if _parameter_versions(optimizer) != versions:
                raise RuntimeError(
                    "optimizer state initialization fallback mutated a model parameter"
                )
            optimizer.load_state_dict(baseline)
            return False
        observed = frozenset(_state_tensor_names(optimizer, names))
        if not frozenset(expected_state_names).issubset(observed):
            optimizer.load_state_dict(baseline)
            return False
        if _parameter_versions(optimizer) != versions:
            raise RuntimeError(
                "optimizer state initialization mutated a model parameter"
            )
        return True
    except BaseException:
        optimizer.load_state_dict(baseline)
        raise


def _run_declared_initializer(optimizer: torch.optim.Optimizer) -> bool:
    """Use an optimizer-provided state initializer when one is available."""

    initializer = getattr(optimizer, "_initialize_parameter_state", None)
    if not callable(initializer):
        return False
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not isinstance(parameter, torch.nn.Parameter):
                return False
            if not parameter.requires_grad:
                continue
            state = optimizer.state[parameter]
            if not state:
                initializer(parameter, group, state)
    return True


def _initialize_at_compiler_boundary(
    optimizer: torch.optim.Optimizer,
    names: Mapping[int, str],
    expected_state_names: tuple[str, ...],
) -> bool:
    """Stop a traceable optimizer after its Python state preamble, before math."""

    parameters = _optimizer_parameters(optimizer)
    original_gradients = tuple(parameter.grad for parameter in parameters)
    gradient_owners: list[torch.Tensor] = []
    reached_boundary = False
    expected = frozenset(expected_state_names)

    def backend(
        _graph_module: torch.fx.GraphModule,
        _inputs: list[object],
    ) -> Callable[..., None]:
        nonlocal reached_boundary
        observed = frozenset(_state_tensor_names(optimizer, names))
        reached_boundary = expected.issubset(observed)

        def stop_before_update(*_arguments: object) -> None:
            raise _StopBeforeOptimizerUpdate

        return stop_before_update

    try:
        for parameter in parameters:
            if not parameter.requires_grad or parameter.grad is not None:
                continue
            owner = torch.ones((), dtype=parameter.dtype, device=parameter.device)
            parameter.grad = owner.expand_as(parameter)
            gradient_owners.append(owner)

        raw_step = inspect.unwrap(type(optimizer).step).__get__(
            optimizer, type(optimizer)
        )

        @torch.no_grad()
        def initialize_step() -> object:
            return raw_step()

        try:
            # This adapter frame is intentionally reused across optimizer
            # classes.  Keep Dynamo from silently falling back to eager
            # execution after its small application-default cache limit.
            with torch._dynamo.config.patch(
                recompile_limit=64,
                accumulated_recompile_limit=1024,
            ):
                torch.compile(initialize_step, backend=backend, fullgraph=False)()
        except _StopBeforeOptimizerUpdate:
            pass
        except BaseException:
            return False
        return reached_boundary
    finally:
        for parameter, gradient in zip(parameters, original_gradients, strict=True):
            parameter.grad = gradient
        gradient_owners.clear()


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if isinstance(parameter, torch.nn.Parameter)
    )


def _parameter_versions(optimizer: torch.optim.Optimizer) -> tuple[int, ...]:
    return tuple(parameter._version for parameter in _optimizer_parameters(optimizer))


def _state_tensor_names(
    optimizer: torch.optim.Optimizer,
    names: Mapping[int, str],
) -> tuple[str, ...]:
    result: list[str] = []
    for parameter in _optimizer_parameters(optimizer):
        parameter_name = names.get(id(parameter))
        if parameter_name is None:
            continue
        for path, _tensor in _tensor_leaves(optimizer.state.get(parameter, {})):
            result.append(f"optimizer.{parameter_name}.{path}")
    return tuple(result)


def _tensor_leaves(
    value: object, prefix: str = ""
) -> tuple[tuple[str, torch.Tensor], ...]:
    result: list[tuple[str, torch.Tensor]] = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, torch.Tensor):
            result.append((path or "value", item))
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                child = f"{path}.{key}" if path else str(key)
                visit(item[key], child)
        elif isinstance(item, tuple | list):
            for index, child_value in enumerate(item):
                child = f"{path}.{index}" if path else str(index)
                visit(child_value, child)

    visit(value, prefix)
    return tuple(result)


__all__ = ["initialize_lazy_optimizer_state"]
