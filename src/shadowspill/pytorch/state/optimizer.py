"""Public relocation operations for materialized optimizer state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from shadowspill.pytorch.runtime_adapter.runtime import Runtime

from .storage import (
    NamedTensor,
    externalize_tensors,
    release_persistent_tensors,
    relocate_tensors,
)


def relocate_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
    pool: str,
    release_source: bool = True,
) -> torch.optim.Optimizer:
    """Copy optimizer tensors into runtime objects and release sources by default."""

    relocate_tensors(
        optimizer,
        _optimizer_tensors(optimizer),
        runtime=runtime,
        pool=pool,
        release_source=release_source,
    )
    return optimizer


def externalize_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
    release_runtime: bool = False,
) -> torch.optim.Optimizer:
    """Copy optimizer tensors back into ordinary CPU allocations."""

    externalize_tensors(
        optimizer,
        runtime=runtime,
        release_runtime=release_runtime,
    )
    return optimizer


def relocate_optimizer_state_for_plan(
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
    pool: str,
) -> None:
    """Move initialized state into spill storage owned by an active plan build."""

    relocate_tensors(
        optimizer,
        _optimizer_tensors(optimizer),
        runtime=runtime,
        pool=pool,
        release_source=True,
        _allow_in_progress_plan=True,
    )


def release_optimizer_state_from_plan(
    optimizer: torch.optim.Optimizer,
    *,
    runtime: Runtime,
) -> None:
    """Drop internal spill ownership after views are external or abandoned."""

    release_persistent_tensors(optimizer, runtime=runtime)


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


__all__ = ["externalize_optimizer_state", "relocate_optimizer_state"]
