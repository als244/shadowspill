"""Copies of an optimizer: on meta geometry for discovery, on fake device tensors for
tracing, and the walks over its tensors that every copy is made from."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor

from shadowspill.errors import CaptureError
from shadowspill.pytorch.accelerator import DEVICE_TYPE


def canonical_parameters(
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


def has_optimizer_step_hooks(optimizer: torch.optim.Optimizer) -> bool:
    return bool(
        getattr(optimizer, "_optimizer_step_pre_hooks", None)
        or getattr(optimizer, "_optimizer_step_post_hooks", None)
    )


def copy_optimizer(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """Copy complete subclass state without Optimizer.__getstate__ truncation."""

    copied = copy.deepcopy(optimizer)
    if copied.__dict__.keys() == optimizer.__dict__.keys():
        return copied
    copied = object.__new__(type(optimizer))
    copied.__dict__ = copy.deepcopy(optimizer.__dict__)
    if not isinstance(copied, torch.optim.Optimizer):
        raise TypeError("copied optimizer changed its base type")
    return copied


def copy_optimizer_to_meta(
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

    parameters = optimizer_parameters(optimizer)
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
    copied_parameters = optimizer_parameters(copied)
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


def optimizer_parameters(
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


def fake_device_optimizer(
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

    parameters = optimizer_parameters(optimizer)
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
                device=DEVICE_TYPE,
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
        converted = map_optimizer_tensors(value, fake_tensor)
        if not isinstance(converted, dict):
            raise CaptureError("per-parameter optimizer state must be a mapping")
        converted_state[fake_parameter] = converted
    optimizer.state = converted_state
    return optimizer, fake_names


def map_optimizer_tensors(value: Any, convert: Any) -> Any:
    """Preserve optimizer state containers while replacing tensor leaves."""

    if isinstance(value, torch.Tensor):
        return convert(value)
    if isinstance(value, dict):
        return {
            key: map_optimizer_tensors(item, convert) for key, item in value.items()
        }
    if isinstance(value, list):
        return [map_optimizer_tensors(item, convert) for item in value]
    if isinstance(value, tuple):
        return tuple(map_optimizer_tensors(item, convert) for item in value)
    return copy.deepcopy(value)
