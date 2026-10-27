"""Rebind task inputs, publish results, and retire views at admitted boundaries."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from shadowspill.runtime.failures import RuntimeExecutionError


@dataclass(frozen=True, slots=True)
class PublishedStorage:
    """One concrete task result matched to a predecoded publication."""

    tensor: torch.Tensor
    alias_id: str
    publication_ordinal: int


@dataclass(frozen=True, slots=True)
class ReplacementStorageViews:
    alias_id: str
    tensors: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if not self.alias_id or not self.tensors:
            raise ValueError("replacement must name an alias and its frontend views")


def before_task_and_acquire(
    task_handle: int,
    device_ordinal: int,
    tensors: Sequence[torch.Tensor],
) -> None:
    """Acquire one predecoded storage-only input vector."""

    if tensors:
        torch.ops.shadowspill._before_task_storages(
            tensors, task_handle, device_ordinal
        )
    else:
        # Tensor-free tasks have no tensor dispatch key.
        torch.ops.shadowspill._before_task_boundary(task_handle, device_ordinal)


def wait_task_allocations(task_handle: int, device_ordinal: int) -> None:
    """Resolve the range-reuse dependencies of this task's allocations."""

    torch.ops.shadowspill._wait_task_allocations(task_handle, device_ordinal)


def after_task_and_update(
    task_handle: int,
    device_ordinal: int,
    adopted: Sequence[PublishedStorage],
    dematerialized: Sequence[torch.Tensor],
    *,
    replacements: Sequence[ReplacementStorageViews] = (),
) -> None:
    """Overwrite logical objects and publish one admitted task boundary."""

    if not (adopted or dematerialized or replacements):
        torch.ops.shadowspill._after_task_boundary(task_handle, device_ordinal)
        return
    replacement_tensors: list[torch.Tensor] = []
    replacement_target_indices: list[int] = []
    if replacements:
        replacement_by_alias = {item.alias_id: item for item in replacements}
        if len(replacement_by_alias) != len(replacements):
            raise RuntimeExecutionError("task replacement aliases are not unique")
        unknown = replacement_by_alias.keys() - {item.alias_id for item in adopted}
        if unknown:
            raise RuntimeExecutionError(
                f"task replacement has no adopted output: {sorted(unknown)}"
            )
        for index, item in enumerate(adopted):
            replacement = replacement_by_alias.get(item.alias_id)
            if replacement is not None:
                replacement_tensors.extend(replacement.tensors)
                replacement_target_indices.extend([index] * len(replacement.tensors))
    torch.ops.shadowspill._after_task_storages(
        tuple(item.tensor for item in adopted),
        tuple(item.publication_ordinal for item in adopted),
        replacement_tensors,
        replacement_target_indices,
        dematerialized,
        task_handle,
        device_ordinal,
    )
