"""An opaque optimizer on real device tensors, for profiling: the isolated copy and the
first-step state it exposes as outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor

from shadowspill.errors import CaptureError
from shadowspill.pytorch.accelerator import accelerator_device

from .artifacts import (
    OpaqueOptimizerArtifact,
    OptimizerTensorBinding,
)
from .bindings import (
    tensor_bindings,
)
from .sandbox import (
    copy_optimizer,
    map_optimizer_tensors,
    optimizer_parameters,
)


def materialize_opaque_optimizer(
    artifact: OpaqueOptimizerArtifact, *, device_ordinal: int
) -> torch.optim.Optimizer:
    """Build an isolated real-CUDA optimizer used only for task profiling."""

    source_parameters = optimizer_parameters(artifact.optimizer)
    optimizer = copy_optimizer(artifact.optimizer)
    parameters = optimizer_parameters(optimizer)
    if len(parameters) != len(source_parameters):
        raise CaptureError("copied opaque optimizer changed its parameter inventory")
    replacements: dict[int, torch.Tensor] = {}
    device = accelerator_device(device_ordinal)

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
        converted = map_optimizer_tensors(
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
    reconciled with ShadowSpillProgram objects.  Names come from optimizer discovery;
    values come only from the real isolated first step.
    """

    parameters = optimizer_parameters(optimizer)
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
        for binding in tensor_bindings(
            optimizer,
            names,
            require_gradients=False,
        )
    }
    outputs: list[OptimizerTensorBinding] = []
    target = accelerator_device(device_ordinal)
    for name in artifact.profile_output_names:
        binding = by_name.get(name)
        if binding is None:
            raise CaptureError(
                f"opaque optimizer did not create profiled state {name!r}"
            )
        tensor = binding.tensor
        if not binding.spillable:
            raise CaptureError(f"opaque optimizer output {name!r} is not spillable")
        if tensor.device.type == "cpu":
            owner = torch.empty(
                tensor.untyped_storage().nbytes(),
                dtype=torch.uint8,
                device=target,
            )
            target_tensor = torch.empty(0, dtype=tensor.dtype, device=target).set_(
                owner.untyped_storage(),
                int(tensor.storage_offset()),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )
            target_tensor.copy_(tensor)
            tensor.data = target_tensor
        elif tensor.device != target:
            raise CaptureError(
                f"opaque optimizer output {name!r} was created on {tensor.device}, "
                f"expected cpu or {target}"
            )
        outputs.append(binding)
    return tuple(outputs)
