"""The values the training executor's phases pass between them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from shadowspill.diagnostics.timing import (
    ArmedTaskTiming as _ArmedTaskTiming,
)
from shadowspill.pytorch.materialization.replacement import ReplacementStorageViews
from shadowspill.pytorch.runtime_adapter.boundaries import PublishedStorage

from ..records import (
    ExecutionTaskRecord as _ExecutionTaskRecord,
)
from ..records import (
    PlanRun as _PlanRun,
)


@dataclass(frozen=True, slots=True)
class TensorLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class ExposedOptimizerTensor:
    tensor: torch.Tensor
    device_placeholder: torch.Tensor


@dataclass(slots=True)
class PreparedTask:
    run: _PlanRun
    record: _ExecutionTaskRecord
    stream: torch.cuda.Stream | None
    arguments: Sequence[object]
    function: Callable[..., object] | None
    eager_optimizer: bool
    timing: _ArmedTaskTiming | None
    runtime_scope_open: bool = True


@dataclass(frozen=True, slots=True)
class TaskCall:
    arguments: Sequence[object]
    function: Callable[..., object] | None
    eager_optimizer: bool


@dataclass(frozen=True, slots=True)
class ProcessedTaskOutputs:
    outputs: tuple[torch.Tensor, ...]
    adopted: tuple[PublishedStorage, ...]
    replacements: tuple[ReplacementStorageViews, ...]
    optimizer_bindings: tuple[tuple[str, torch.Tensor, str], ...] = ()

    @property
    def replacement_aliases(self) -> frozenset[str]:
        return frozenset(item.alias_id for item in self.replacements)


def same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors name the same bytes with the same geometry."""

    return bool(
        left.untyped_storage()._cdata == right.untyped_storage()._cdata
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype == right.dtype
    )


__all__ = [
    "ExposedOptimizerTensor",
    "PreparedTask",
    "ProcessedTaskOutputs",
    "TaskCall",
    "TensorLayout",
    "same_tensor_view",
]
