"""Public optimizer-state import and export operations."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

import torch

from shadowspill.pytorch.runtime_adapter.runtime import Runtime

from .storage import (
    NamedTensor,
    export_tensors,
    import_state_from_file,
    import_tensors,
    persistent_state,
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
        values = {key: value for key, value in group.items() if key != "params"}
        result.extend(_walk_tensors(values, f"param_groups.{ordinal}", parameters))
    return tuple(result)


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
