"""Optimizer-agnostic capture with an explicit bounded opaque fallback: the entry
points, composing discovery, the trace and the tasks."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace

import torch

from .artifacts import (
    OpaqueOptimizerArtifact,
    OptimizerCapture,
    OptimizerTensorBinding,
)
from .bindings import (
    tensor_bindings,
)
from .discovery import (
    discover_optimizer_state,
    validate_optimizer_inputs,
)
from .phases import NoTimer, PhaseTimer
from .sandbox import (
    canonical_parameters,
)
from .store import OptimizerCaptureStore
from .trace import (
    capture_recurrent_optimizer,
)


@dataclass(frozen=True, slots=True)
class DeclaredStateEntry:
    """One optimizer-state tensor an optimizer says it will keep."""

    parameter_name: str
    entry_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


def declare_optimizer_state(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    receives_gradient: Collection[str] | None = None,
) -> tuple[DeclaredStateEntry, ...]:
    """Ask the optimizer what state it will keep, without allocating any.

    The optimizer is run on a storage-free copy whose payload tensors are meta
    geometry, so it names every entry, fixes every shape, and chooses every
    dtype at no cost. Nothing is assumed about which entries exist, how many
    there are, whether they are parameter-shaped, or whether their dtype
    matches the parameter's -- the optimizer says, and this reports.

    A parameter that is not trained is not given a gradient in the copy, so an
    optimizer skips it and declares nothing for it. `receives_gradient`
    widens what "not trained" means beyond the flag: a parameter the
    objective never reaches gets no gradient however it is flagged, and an
    optimizer that is handed one anyway keeps state it will never step.
    """

    inventory = validate_optimizer_inputs(named_parameters, optimizer)
    discovery = discover_optimizer_state(
        inventory, optimizer, receives_gradient=receives_gradient
    )
    if isinstance(discovery, OptimizerCapture):
        return ()
    declared: list[DeclaredStateEntry] = []
    for parameter, entries in discovery.sandbox.state.items():
        name = discovery.name_by_sandbox_id.get(id(parameter))
        if name is None or not isinstance(entries, Mapping):
            continue
        for entry_name, value in entries.items():
            if not isinstance(value, torch.Tensor):
                continue
            declared.append(
                DeclaredStateEntry(
                    parameter_name=name,
                    entry_name=str(entry_name),
                    shape=tuple(value.shape),
                    dtype=value.dtype,
                )
            )
    return tuple(declared)


def capture_optimizer(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    parameter_stage_owners: Mapping[str, tuple[int, ...]] | None = None,
    receives_gradient: Collection[str] | None = None,
    store: OptimizerCaptureStore | None = None,
    timer: PhaseTimer | None = None,
) -> OptimizerCapture:
    """Capture a recurrent tensor update without mutating the caller's state.

    Lazy Python/tensor state is discovered on a deep-copied optimizer. Its first
    semantic update remains an ordinary bounded optimizer task. Once state is
    stable, a lifted tensor-only graph is used when Dynamo can represent it;
    otherwise all steps remain bounded opaque tasks with measured workspace.

    With a ``store``, the traced update is served from it when the same
    optimizer, over the same tensors, split the same way, was traced before;
    the discovery of lazy state and the opaque first step run either way.
    """

    phases = timer if timer is not None else NoTimer()
    with phases.measure("optimizer_discovery"):
        inventory = validate_optimizer_inputs(named_parameters, optimizer)
        discovery = discover_optimizer_state(
            inventory, optimizer, receives_gradient=receives_gradient
        )
    if isinstance(discovery, OptimizerCapture):
        return discovery
    captured = capture_recurrent_optimizer(
        discovery,
        optimizer,
        parameter_stage_owners=parameter_stage_owners,
        store=store,
        timer=phases,
    )
    if discovery.created_state_names:
        spillable_names = {
            binding.name for binding in captured.bindings if binding.spillable
        }
        initial_bindings = tensor_bindings(
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
    return captured


def current_optimizer_bindings(
    named_parameters: Mapping[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> tuple[OptimizerTensorBinding, ...]:
    """Describe current optimizer tensors with capture-stable names."""

    canonical = canonical_parameters(named_parameters)
    name_by_id = {id(parameter): name for name, parameter in canonical.items()}
    return tensor_bindings(optimizer, name_by_id, require_gradients=False)


__all__ = [
    "DeclaredStateEntry",
    "PhaseTimer",
    "capture_optimizer",
    "current_optimizer_bindings",
    "declare_optimizer_state",
]
