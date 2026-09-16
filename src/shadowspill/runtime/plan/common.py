"""Values and helpers every part of the bridge shares."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shadowspill.errors import PlanningError
from shadowspill.ir import MemoryAction, MemoryActionKind
from shadowspill.runtime.abi import RuntimeAction
from shadowspill.runtime.failures import (
    RuntimeExecutionError,
    generic_runtime_error,
    read_allocator_failure,
)

if TYPE_CHECKING:
    from shadowspill.task.allocations import TaskAllocationContract

#: How the runtime numbers each memory action kind.
ACTION_KIND = {
    MemoryActionKind.RELEASE: 0,
    MemoryActionKind.EVICT: 1,
    MemoryActionKind.FETCH: 2,
    MemoryActionKind.WRITE_BACK: 3,
}


@dataclass(frozen=True, slots=True)
class TaskMemoryEnvelope:
    """Conservative bounds on one task's anonymous allocator behavior."""

    maximum_requested_allocation_bytes: int = 0
    maximum_charged_allocation_bytes: int = 0
    live_requested_allocation_limit_bytes: int = 0
    live_charged_allocation_limit_bytes: int = 0
    dynamic_scratch_maximum_allocation_bytes: int = 0
    dynamic_scratch_live_limit_bytes: int = 0
    allocation_path_digests: tuple[str, ...] = ()
    allocation_contract: TaskAllocationContract | None = None

    def __post_init__(self) -> None:
        values = (
            self.maximum_requested_allocation_bytes,
            self.maximum_charged_allocation_bytes,
            self.live_requested_allocation_limit_bytes,
            self.live_charged_allocation_limit_bytes,
            self.dynamic_scratch_maximum_allocation_bytes,
            self.dynamic_scratch_live_limit_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("task memory envelope bounds must be non-negative")
        if (
            self.live_requested_allocation_limit_bytes
            and self.maximum_requested_allocation_bytes
            > self.live_requested_allocation_limit_bytes
        ):
            raise ValueError("requested allocation maximum exceeds live limit")
        if (
            self.live_charged_allocation_limit_bytes
            and self.maximum_charged_allocation_bytes
            > self.live_charged_allocation_limit_bytes
        ):
            raise ValueError("charged allocation maximum exceeds live limit")
        if (
            self.dynamic_scratch_live_limit_bytes
            and self.dynamic_scratch_maximum_allocation_bytes
            > self.dynamic_scratch_live_limit_bytes
        ):
            raise ValueError("scratch allocation maximum exceeds scratch live limit")
        if any(len(value) != 64 for value in self.allocation_path_digests):
            raise ValueError("allocation path digests must be SHA-256")


@dataclass(frozen=True, slots=True)
class TaskPublication:
    """One stable logical object that an admitted task may publish."""

    alias_id: str
    replace_lease: bool = False


def plan_local_id(value: str, prefix: str) -> int:
    """The number behind a plan-local identity such as ``alias_000017``."""

    if not value.startswith(prefix):
        raise PlanningError(
            f"plan-local identity {value!r} does not start with {prefix!r}"
        )
    suffix = value.removeprefix(prefix)
    if not suffix.isdigit():
        raise PlanningError(f"plan-local identity {value!r} has a nonnumeric suffix")
    return int(suffix)


def action_labels(
    actions: tuple[MemoryAction, ...],
    labels: tuple[str, ...] | None,
) -> tuple[str, ...]:
    resolved = ("",) * len(actions) if labels is None else labels
    if len(resolved) != len(actions):
        raise ValueError("action trace labels must align with ordered actions")
    return resolved


def runtime_action(
    object_id: int,
    kind: int,
    *,
    trace_label: bytes | None = None,
) -> RuntimeAction:
    """Construct one ABI action without relying on ctypes field ordering."""

    return RuntimeAction(
        object_id=object_id,
        kind=kind,
        trace_label=trace_label,
    )


def require_status(library: Any, raw_status: Any, operation: str) -> None:
    """Raise for a nonzero adapter status, with the latched failure if any."""

    status = int(raw_status)
    if status == 0:
        return
    diagnostics = read_allocator_failure(library, operation)
    if diagnostics is not None:
        raise generic_runtime_error(diagnostics)
    raise RuntimeExecutionError(f"{operation} failed with status {status}")


def actions_by_task(
    actions: Iterable[MemoryAction],
) -> Mapping[str, tuple[MemoryAction, ...]]:
    """Preserve schedule ordering while grouping actions by trigger task."""

    result: dict[str, list[MemoryAction]] = {}
    for action in actions:
        result.setdefault(action.trigger_task_id, []).append(action)
    return {task_id: tuple(values) for task_id, values in result.items()}


__all__ = [
    "ACTION_KIND",
    "TaskMemoryEnvelope",
    "TaskPublication",
    "action_labels",
    "actions_by_task",
    "plan_local_id",
    "require_status",
    "runtime_action",
]
