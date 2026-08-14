"""Structured runtime failures surfaced by the PyTorch frontend."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure

_NO_ID = (1 << 64) - 1
_OUT_OF_MEMORY = 3
_NO_PROGRESS = 4
_TASK_ALLOCATION_ENVELOPE_EXCEEDED = 10
_STATUS_NAMES = {
    0: "ok",
    1: "invalid_argument",
    2: "allocation_failure",
    _OUT_OF_MEMORY: "out_of_memory",
    _NO_PROGRESS: "no_progress",
    5: "invalid_state",
    6: "plan_violation",
    7: "backend_failure",
    8: "worker_failure",
    9: "closed",
    _TASK_ALLOCATION_ENVELOPE_EXCEEDED: "task_allocation_envelope_exceeded",
}


@dataclass(frozen=True, slots=True)
class ExecutionTaskIdentity:
    """User-facing and canonical identities for one execution task."""

    execution_task_id: str
    semantic_name: str
    canonical_task_id: str


@dataclass(frozen=True, slots=True)
class RuntimeFailureDiagnostics:
    """Immutable allocator/runtime failure details retained after an exception."""

    operation: str
    status: int
    status_name: str
    device_ordinal: int
    requested_bytes: int
    free_bytes: int
    largest_free_range_bytes: int
    object_id: int | None
    allocation_id: int | None
    task_id: int | None = None
    task_live_requested_bytes: int = 0
    task_live_charged_bytes: int = 0
    task_live_requested_limit_bytes: int = 0
    task_live_charged_limit_bytes: int = 0
    task_maximum_requested_allocation_bytes: int = 0
    task_maximum_charged_allocation_bytes: int = 0
    task: ExecutionTaskIdentity | None = None

    @property
    def is_allocator_oom(self) -> bool:
        """Whether the failure is the null-allocation case we translate."""

        return self.status in {_OUT_OF_MEMORY, _NO_PROGRESS} and (
            self.requested_bytes > 0
        )

    @property
    def is_recoverable_no_progress(self) -> bool:
        """Whether synchronized teardown may clear the native failure latch."""

        return self.status == _NO_PROGRESS and self.requested_bytes > 0

    @property
    def is_shadowspill_contract_failure(self) -> bool:
        """Whether ShadowSpill should replace a secondary provider exception."""

        return self.status in {
            6,  # plan_violation
            _TASK_ALLOCATION_ENVELOPE_EXCEEDED,
        }

    def as_dict(self) -> dict[str, object]:
        task = self.task
        return {
            "operation": self.operation,
            "status": self.status,
            "status_name": self.status_name,
            "device_ordinal": self.device_ordinal,
            "requested_bytes": self.requested_bytes,
            "free_bytes": self.free_bytes,
            "largest_free_range_bytes": self.largest_free_range_bytes,
            "object_id": self.object_id,
            "allocation_id": self.allocation_id,
            "task_id": self.task_id,
            "task_live_requested_bytes": self.task_live_requested_bytes,
            "task_live_charged_bytes": self.task_live_charged_bytes,
            "task_live_requested_limit_bytes": (
                self.task_live_requested_limit_bytes
            ),
            "task_live_charged_limit_bytes": self.task_live_charged_limit_bytes,
            "task_maximum_requested_allocation_bytes": (
                self.task_maximum_requested_allocation_bytes
            ),
            "task_maximum_charged_allocation_bytes": (
                self.task_maximum_charged_allocation_bytes
            ),
            "execution_task_id": None if task is None else task.execution_task_id,
            "semantic_name": None if task is None else task.semantic_name,
            "canonical_task_id": None if task is None else task.canonical_task_id,
        }


class RuntimeExecutionError(RuntimeError):
    """A ShadowSpill runtime operation rejected planned execution."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: RuntimeFailureDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        self._cleanup_prepared = False

    def _begin_cleanup(self) -> bool:
        """Claim fault cleanup once across nested planning/execution boundaries."""

        if self._cleanup_prepared:
            return False
        self._cleanup_prepared = True
        return True


def read_allocator_failure(
    library: Any,
    operation: str,
    *,
    task: ExecutionTaskIdentity | None = None,
) -> RuntimeFailureDiagnostics | None:
    """Read the adapter's immutable first-failure snapshot, if one exists."""

    failure = AdapterFailure()
    status = int(library.shadowspill_pytorch_allocator_failure(ctypes.byref(failure)))
    if status == 0:
        return None
    requested = max(int(failure.requested_bytes), int(failure.runtime.requested_bytes))
    return RuntimeFailureDiagnostics(
        operation=operation,
        status=status,
        status_name=_STATUS_NAMES.get(status, f"unknown_{status}"),
        device_ordinal=int(failure.device_ordinal),
        requested_bytes=requested,
        free_bytes=int(failure.runtime.free_bytes),
        largest_free_range_bytes=int(failure.runtime.largest_free_range_bytes),
        object_id=_optional_id(int(failure.runtime.object_id)),
        allocation_id=_optional_id(int(failure.runtime.allocation_id)),
        task_id=_optional_id(int(failure.runtime.task_id)),
        task_live_requested_bytes=int(failure.runtime.task_live_requested_bytes),
        task_live_charged_bytes=int(failure.runtime.task_live_charged_bytes),
        task_live_requested_limit_bytes=int(
            failure.runtime.task_live_requested_limit_bytes
        ),
        task_live_charged_limit_bytes=int(
            failure.runtime.task_live_charged_limit_bytes
        ),
        task_maximum_requested_allocation_bytes=int(
            failure.runtime.task_maximum_requested_allocation_bytes
        ),
        task_maximum_charged_allocation_bytes=int(
            failure.runtime.task_maximum_charged_allocation_bytes
        ),
        task=task,
    )


def allocator_oom_error(
    diagnostics: RuntimeFailureDiagnostics,
) -> RuntimeExecutionError:
    """Build the precise public error for a failed allocator callback."""

    if not diagnostics.is_allocator_oom:
        raise ValueError("allocator OOM error requires an OOM diagnostic")
    title = (
        "ShadowSpill no-progress OOM"
        if diagnostics.status == _NO_PROGRESS
        else "ShadowSpill OOM"
    )
    lines = [title]
    task = diagnostics.task
    if task is not None:
        lines.extend(
            (
                f"execution_task: {task.execution_task_id}",
                f"semantic_task: {task.semantic_name}",
                f"canonical_task: {task.canonical_task_id}",
            )
        )
    else:
        lines.append(f"operation: {diagnostics.operation}")
    lines.extend(
        (
            f"device: {diagnostics.device_ordinal}",
            f"requested: {diagnostics.requested_bytes}",
            f"free: {diagnostics.free_bytes}",
            f"largest_free_range: {diagnostics.largest_free_range_bytes}",
        )
    )
    if diagnostics.object_id is not None:
        lines.append(f"object: {diagnostics.object_id}")
    if diagnostics.allocation_id is not None:
        lines.append(f"allocation: {diagnostics.allocation_id}")
    return RuntimeExecutionError("\n".join(lines), diagnostics=diagnostics)


def generic_runtime_error(
    diagnostics: RuntimeFailureDiagnostics,
) -> RuntimeExecutionError:
    """Build an error for an explicit runtime API rejection."""

    lines = [
        f"{diagnostics.operation} failed with status {diagnostics.status} "
        f"({diagnostics.status_name})"
    ]
    if diagnostics.task is not None:
        lines.extend(
            (
                f"execution_task: {diagnostics.task.execution_task_id}",
                f"semantic_task: {diagnostics.task.semantic_name}",
                f"canonical_task: {diagnostics.task.canonical_task_id}",
            )
        )
    elif diagnostics.task_id is not None:
        lines.append(f"runtime_task: {diagnostics.task_id}")
    lines.extend(
        (
            f"device: {diagnostics.device_ordinal}",
            f"object: {diagnostics.object_id}",
            f"allocation: {diagnostics.allocation_id}",
            f"requested: {diagnostics.requested_bytes}",
            f"free: {diagnostics.free_bytes}",
            f"largest_free_range: {diagnostics.largest_free_range_bytes}",
        )
    )
    if diagnostics.status == _TASK_ALLOCATION_ENVELOPE_EXCEEDED:
        lines.extend(
            (
                "reason: TASK_ALLOCATION_ENVELOPE_EXCEEDED",
                f"task_live_requested: {diagnostics.task_live_requested_bytes}",
                f"task_live_charged: {diagnostics.task_live_charged_bytes}",
                "task_live_requested_limit: "
                f"{diagnostics.task_live_requested_limit_bytes}",
                "task_live_charged_limit: "
                f"{diagnostics.task_live_charged_limit_bytes}",
                "task_maximum_requested_allocation: "
                f"{diagnostics.task_maximum_requested_allocation_bytes}",
                "task_maximum_charged_allocation: "
                f"{diagnostics.task_maximum_charged_allocation_bytes}",
            )
        )
    return RuntimeExecutionError("\n".join(lines), diagnostics=diagnostics)


def raise_if_allocator_failed(library: Any, operation: str) -> None:
    """Raise a latched allocator failure before issuing dependent CUDA work."""

    diagnostics = read_allocator_failure(library, operation)
    if diagnostics is None:
        return
    if diagnostics.is_allocator_oom:
        raise allocator_oom_error(diagnostics)
    raise generic_runtime_error(diagnostics)


def _optional_id(value: int) -> int | None:
    return None if value == _NO_ID else value


__all__ = [
    "ExecutionTaskIdentity",
    "RuntimeExecutionError",
    "RuntimeFailureDiagnostics",
    "allocator_oom_error",
    "generic_runtime_error",
    "raise_if_allocator_failed",
    "read_allocator_failure",
]
