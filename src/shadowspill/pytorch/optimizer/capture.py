"""Optimizer-agnostic capture with an explicit bounded opaque fallback: the entry
points, composing discovery, the trace and the tasks."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import torch

from shadowspill.errors import CaptureError

from .artifacts import (
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
from .starts import HeldStart, NoStart, StateStart
from .store import OptimizerCaptureStore
from .trace import (
    capture_optimizer_update,
)


@dataclass(frozen=True, slots=True)
class DeclaredStateEntry:
    """One optimizer-state tensor an optimizer says it will keep, and what it
    starts at."""

    parameter_name: str
    entry_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    start: StateStart


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

    How the step makes each entry says what it starts at (`starts`). An entry
    the optimizer already holds, as after loading a checkpoint into it, starts
    at what it holds.

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
        held = optimizer.state.get(inventory.canonical_parameters[name], {})
        for entry_name, value in entries.items():
            if not isinstance(value, torch.Tensor):
                continue
            start = discovery.state_starts.get(
                (name, str(entry_name)), NoStart("made before the step")
            )
            if isinstance(held.get(entry_name), torch.Tensor):
                start = HeldStart()
            declared.append(
                DeclaredStateEntry(
                    parameter_name=name,
                    entry_name=str(entry_name),
                    shape=tuple(value.shape),
                    dtype=value.dtype,
                    start=start,
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
    compute_copies: Mapping[str, torch.Tensor] | None = None,
    gradient_dtype: torch.dtype | None = None,
) -> OptimizerCapture:
    """Capture the optimizer's update without mutating the caller's state.

    The update is found on a deep-copied optimizer, which must already hold the
    state the update reads: planning creates what the optimizer's first step
    would before capturing, and state that step would still create is refused.
    A lifted tensor-only graph is used when Dynamo can represent the update;
    otherwise every step runs it as a bounded opaque task with measured
    workspace.

    With a ``store``, the traced update is served from it when the same
    optimizer, over the same tensors, split the same way, was traced before.

    ``compute_copies`` names the parameters in ``named_parameters`` that are
    master copies, each with the weights the model computes with at another
    precision, and the update writes each copy from its master once it has
    stepped. Every gradient reaches the update as the step produces it: at
    ``gradient_dtype``, or at its weights' dtype when that is ``None``; the
    update casts one that is not at its parameter's dtype. Only a traced
    update does either; an optimizer whose update cannot be traced is refused
    them.
    """

    phases = timer if timer is not None else NoTimer()
    with phases.measure("optimizer_discovery"):
        inventory = validate_optimizer_inputs(named_parameters, optimizer)
        discovery = discover_optimizer_state(
            inventory, optimizer, receives_gradient=receives_gradient
        )
    casts = bool(compute_copies) or any(
        parameter.requires_grad and parameter.dtype != gradient_dtype
        for parameter in named_parameters.values()
        if gradient_dtype is not None
    )
    if isinstance(discovery, OptimizerCapture):
        return _refuse_opaque_casts(discovery, casts)
    if discovery.created_state_names:
        created = ", ".join(discovery.created_state_names[:4])
        raise CaptureError(
            f"the optimizer's first step creates state ({created}) that planning "
            "did not create before it; import the optimizer's state before planning"
        )
    return _refuse_opaque_casts(
        capture_optimizer_update(
            discovery,
            optimizer,
            parameter_stage_owners=parameter_stage_owners,
            store=store,
            timer=phases,
            compute_copies=compute_copies,
            gradient_dtype=gradient_dtype,
        ),
        casts,
    )


def _refuse_opaque_casts(captured: OptimizerCapture, casts: bool) -> OptimizerCapture:
    if casts and captured.update_is_opaque:
        raise CaptureError(
            "an optimizer over master copies or over gradients kept at another "
            "dtype needs an update that can be traced, which casts the gradients "
            f"and writes the copies; this one cannot be: {captured.opaque_reason}"
        )
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
