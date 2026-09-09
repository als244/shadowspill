"""Public optimizer-state import and export operations."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence

import torch

from shadowspill.pytorch.optimizer.capture import declare_optimizer_state
from shadowspill.pytorch.runtime_adapter.runtime import MemoryPool, Runtime

from .storage import (
    NamedTensor,
    export_tensors,
    import_state_from_file,
    import_tensors,
    persistent_state,
    pool_backed_tensor,
    read_state,
    release_persistent_tensors,
)


def import_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
    pool: str,
    release_source: bool = True,
) -> torch.optim.Optimizer:
    """Copy optimizer tensors into runtime objects and release sources by default."""

    import_tensors(
        optimizer,
        _optimizer_tensors(optimizer),
        runtime=runtime,
        pool=pool,
        release_source=release_source,
    )
    return optimizer


def export_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
    release_runtime: bool = False,
) -> torch.optim.Optimizer:
    """Copy optimizer tensors back into ordinary CPU allocations."""

    export_tensors(
        optimizer,
        runtime=runtime,
        release_runtime=release_runtime,
    )
    return optimizer


def import_optimizer_state_from_file(
    optimizer: torch.optim.Optimizer,
    path: str | os.PathLike[str],
    *,
    runtime: Runtime,
    pool: str,
) -> None:
    """Fill the optimizer's state in ``pool`` from a checkpoint on disk.

    Keyed by the paths :func:`import_optimizer_state` enumerates, which is
    what :func:`read_optimizer_state` writes, so a checkpoint that came from
    one reads back through the other. The optimizer must already have the
    state the checkpoint names.
    """

    import_state_from_file(
        optimizer, _optimizer_tensors(optimizer), path, runtime=runtime, pool=pool
    )


def read_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
    copy: bool = True,
) -> dict[str, torch.Tensor]:
    """Return the optimizer's current values without rebinding its tensors.

    Keyed by the same paths ``import_optimizer_state`` enumerates, so it is a
    flat mapping rather than an optimizer ``state_dict`` shape. ``copy`` has
    the meaning it has in :func:`read_state`.
    """

    return read_state(
        optimizer, _optimizer_tensors(optimizer), runtime=runtime, copy=copy
    )


def _held_in_tensor(value: object) -> object:
    """A number becomes a scalar tensor; a sequence of them, one each.

    Each is held in the widest type of its own kind -- float64 for a float,
    int64 for an int -- because that is what a Python number already is, so
    holding it loses nothing and a constant an optimizer derives from it comes
    out as it would have.

    They stay on the host. A scalar the update reads is a handful of bytes that
    never belongs on the device: keeping it here means setting it is a host
    write, with nothing copied and nothing synchronized.

    A ``bool`` is refused, and is the reason this checks exact types rather than
    ``isinstance``. A bool in an optimizer selects behaviour rather than scaling
    it -- ``amsgrad``, ``maximize``, ``nesterov`` -- and behaviour is what the
    capture *is*. Holding one in a tensor would not make the update follow it;
    it would either be read as a value the capture fixed anyway, or change what
    the update computes. Two behaviours are two captures, and two plans.
    """

    if isinstance(value, torch.Tensor):
        return value
    if type(value) is float:
        return torch.tensor(value, dtype=torch.float64)
    if type(value) is int:
        return torch.tensor(value, dtype=torch.int64)
    if isinstance(value, tuple | list) and value:
        held = tuple(_held_in_tensor(item) for item in value)
        if all(isinstance(item, torch.Tensor) for item in held):
            return list(held) if isinstance(value, list) else held
    return None


def declare_varying_hyperparams(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    names: Sequence[str],
) -> tuple[str, ...]:
    """Hold the named values in tensors, so a step can set them afterwards.

    A capture fixes whatever it read as a Python number: a float learning rate
    becomes a constant in the lowered step, and assigning to the parameter
    group afterwards changes nothing. A tensor is read when the step runs, and
    a tensor the capture was given is an input to it rather than a value folded
    into it, so one capture serves every value it takes.

    Only the names given here are held that way, and the naming is the point.
    Promoting every number would be shorter to write and wrong: an optimizer is
    entitled to require one -- to branch on it, or to derive a constant at the
    precision it carried -- and turning that into a tensor behind its back
    either changes what it computes or fails inside it, far from the line that
    caused it. Naming a value is the caller saying this one is safe to hold.

    A name addresses an entry in the optimizer's parameter groups or a buffer
    on the model, which are the two registries of named values that already
    exist. An entry holding several values, as ``betas`` does, has each of them
    held. Every group carrying the name is held, so one schedule reaches an
    optimizer with several groups.
    """

    requested = tuple(dict.fromkeys(names))
    buffers = dict(model.named_buffers())
    declared: list[str] = []
    for name in requested:
        groups = [group for group in optimizer.param_groups if name in group]
        if groups and name in buffers:
            raise ValueError(
                f"{name!r} names both an optimizer value and a model buffer, "
                "so which one a step would set is ambiguous; rename one"
            )
        if not groups and name not in buffers:
            raise ValueError(
                f"no optimizer value or model buffer named {name!r}, so there "
                "is nothing for a step to set"
            )
        if name in buffers:
            declared.append(name)
            continue
        for group in groups:
            held = _held_in_tensor(group[name])
            if held is None:
                held_type = type(group[name]).__name__
                reason = (
                    "a bool selects what the update does, and what it does is "
                    "what was captured, so changing one means planning again"
                    if held_type == "bool"
                    else "only a number, or a sequence of them, can be held "
                    "in a tensor"
                )
                raise TypeError(
                    f"{name!r} is {held_type}, which a step cannot set: {reason}"
                )
            group[name] = held
        declared.append(name)
    return tuple(declared)


def install_declared_optimizer_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
    pool: MemoryPool,
    initialize: Callable[[str, torch.Tensor, torch.nn.Parameter], None] | None,
) -> int:
    """Give the optimizer its state in the pool, and let the caller fill it.

    The optimizer declares what state it keeps by being run on meta
    parameters, which allocates nothing and fixes every entry's shape and
    dtype -- including entries that are not parameter-shaped, and dtypes it
    chose rather than inherited. Each declared entry is then allocated in
    ``pool`` and handed to ``initialize``, which writes its values in place.

    Nothing is assumed about what those values are: a default would be one
    optimizer family's convention, and an optimizer whose state starts
    elsewhere would train subtly wrong rather than fail. Returns how many
    entries were installed.

    State the caller already imported for this optimizer is left alone, since
    the caller owns it and it outlives the plan.
    """

    if persistent_state(runtime, optimizer) is not None:
        return 0
    named = dict(model.named_parameters())
    declared = declare_optimizer_state(named, optimizer)
    if not declared:
        return 0
    if initialize is None:
        entries = ", ".join(sorted({item.entry_name for item in declared})[:4])
        raise RuntimeError(
            f"this optimizer keeps state ({entries}) and nothing says what it "
            "starts at. Pass optimizer_state_init to fill each entry in place, or "
            "import the optimizer's state yourself before planning."
        )
    for entry in declared:
        parameter = named[entry.parameter_name]
        view, _allocation = pool_backed_tensor(
            runtime, pool, shape=entry.shape, dtype=entry.dtype
        )
        initialize(entry.entry_name, view, parameter)
        optimizer.state.setdefault(parameter, {})[entry.entry_name] = view
    return len(declared)


def adopt_optimizer_state_for_plan(
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
    pool: str,
    owning_plan: int,
) -> bool:
    """Give one plan the optimizer state it needs, and say whether it owns it.

    The counterpart of :func:`adopt_model_state_for_plan`, and the same rule:
    state the caller imported is adopted as it stands and outlives the plan,
    state the caller did not import is imported here and belongs to the plan.

    planning took it from there, by storage identity; that state is adopted
    where it is instead of being copied into a second object.
    """

    existing = persistent_state(runtime, optimizer)
    if existing is not None:
        if existing.pool != pool:
            raise RuntimeError(
                f"optimizer state is in pool {existing.pool!r}, not requested {pool!r}"
            )
        return False
    import_tensors(
        optimizer,
        _optimizer_tensors(optimizer),
        runtime=runtime,
        pool=pool,
        release_source=True,
        owning_plan=owning_plan,
        _allow_in_progress_plan=True,
    )
    return True


def release_optimizer_state_from_plan(
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
) -> bool:
    """Drop optimizer state a plan created, and say whether it dropped any.

    State the caller imported is left alone: the plan was lent it and does not
    get to end it.
    """

    state = persistent_state(runtime, optimizer)
    if state is None or state.owning_plan is None:
        return False
    release_persistent_tensors(optimizer, runtime=runtime)
    return True


def _optimizer_tensors(
    optimizer: torch.optim.Optimizer,
) -> tuple[NamedTensor, ...]:
    parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
        if isinstance(parameter, torch.Tensor)
    }
    result: list[NamedTensor] = []
    for ordinal, (_parameter, state) in enumerate(optimizer.state.items()):
        result.extend(_walk_tensors(state, f"state.{ordinal}", parameters))
    for ordinal, group in enumerate(optimizer.param_groups):
        values = {
            key: value
            for key, value in group.items()
            # A scalar in a group is a setting the update reads and the caller
            # writes between steps. Importing it would move the value into a
            # pool, where the caller's next write would not reach it, to save
            # eight bytes. Anything larger is state and is imported.
            if key != "params" and not _is_host_scalar(value)
        }
        result.extend(_walk_tensors(values, f"param_groups.{ordinal}", parameters))
    return tuple(result)


def _is_host_scalar(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return value.ndim == 0 and value.device.type == "cpu"
    if isinstance(value, tuple | list):
        return bool(value) and all(_is_host_scalar(item) for item in value)
    return False


def _walk_tensors(
    value: object,
    path: str,
    excluded: set[int],
) -> Iterable[NamedTensor]:
    if isinstance(value, torch.Tensor):
        if id(value) not in excluded:
            yield NamedTensor(path, value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_tensors(item, f"{path}.{key}", excluded)
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk_tensors(item, f"{path}.{index}", excluded)


__all__ = [
    "export_optimizer_state",
    "import_optimizer_state",
    "import_optimizer_state_from_file",
    "read_optimizer_state",
]
