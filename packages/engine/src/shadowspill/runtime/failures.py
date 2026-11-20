"""Structured runtime failures surfaced by a frontend."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

from shadowspill.status import Status

from .abi import (
    AdapterFailure,
    AdapterStatistics,
    runtime_library,
)
from .admission_capi import optional_id

_OUT_OF_MEMORY = Status.OUT_OF_MEMORY
_NO_PROGRESS = Status.NO_PROGRESS
_TASK_ALLOCATION_ENVELOPE_EXCEEDED = Status.TASK_ALLOCATION_ENVELOPE_EXCEEDED
_TASK_ALLOCATION_CONTRACT_MISMATCH = Status.TASK_ALLOCATION_CONTRACT_MISMATCH
_STATUS_NAMES: dict[int, str] = {
    Status.OK: "ok",
    Status.INVALID_ARGUMENT: "invalid_argument",
    Status.INTERNAL_FAILURE: "internal_failure",
    Status.OUT_OF_MEMORY: "out_of_memory",
    Status.NO_PROGRESS: "no_progress",
    Status.INVALID_STATE: "invalid_state",
    Status.PLAN_VIOLATION: "plan_violation",
    Status.BACKEND_FAILURE: "backend_failure",
    Status.WORKER_FAILURE: "worker_failure",
    Status.CLOSED: "closed",
    Status.TASK_ALLOCATION_ENVELOPE_EXCEEDED: ("task_allocation_envelope_exceeded"),
    Status.TASK_ALLOCATION_CONTRACT_MISMATCH: ("task_allocation_contract_mismatch"),
}


def _reason_name(reason: int) -> str:
    """The runtime's own name for why it refused."""

    if reason == 0:
        return ""
    rendered = runtime_library().shadowspill_failure_reason_string(reason)
    return "" if rendered is None else bytes(rendered).decode()


def format_bytes(value: int) -> str:
    """Bytes as a person reads them, with the exact count kept.

    Reports are read by someone deciding what to change, and "16273899520"
    does not tell them it exceeds a 16 GiB budget.
    """

    scaled = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if unit == "B":
            if scaled < 1024.0:
                return f"{value} B"
        elif scaled < 1024.0:
            return f"{scaled:.2f} {unit} ({value} bytes)"
        scaled /= 1024.0
    return f"{scaled * 1024.0:.2f} PiB ({value} bytes)"


_ALLOCATION_OPERATIONS = {0: "allocate", 1: "free", 255: "end_of_task"}


@dataclass(frozen=True, slots=True)
class RuntimeFailureDiagnostics:
    """What the runtime recorded about its first failure.

    `RuntimeExecutionError.diagnostics` carries it. The runtime keeps only
    the first failure it latches, so later refusals report that one. Byte
    counts describe the pool when the failure was latched.

    Attributes
    ----------
    operation : str
        The call that failed, as its caller names it.
    status : int
        The runtime's status code.
    status_name : str
        The status's name, such as ``"out_of_memory"``, ``"no_progress"`` or
        ``"plan_violation"``; ``"unknown_<code>"`` for a code it does not
        name.
    device_ordinal : int
        The device the failure happened on.
    requested_bytes : int
        Bytes the failing request asked for; zero when no request was
        involved.
    free_bytes : int
        Free bytes in the pool.
    largest_free_range_bytes : int
        The pool's largest free range, in bytes.
    object_id : int or None
        The runtime object involved, if any.
    allocation_id : int or None
        The allocation involved, if any.
    task_id : int or None
        The runtime's id for the task the failure happened in, if any.
    task_live_requested_bytes : int
        For an exceeded task allocation envelope, the bytes the task's live
        allocations asked for, the refused request included; zero otherwise.
    task_live_charged_bytes : int
        The same, in charged bytes.
    task_live_requested_limit_bytes : int
        The task's limit on the bytes its live allocations ask for; zero for
        no limit.
    task_live_charged_limit_bytes : int
        Its limit on their charged bytes; zero for no limit.
    task_maximum_requested_allocation_bytes : int
        Its limit on the bytes one allocation asks for; zero for no limit.
    task_maximum_charged_allocation_bytes : int
        Its limit on one allocation's charged bytes; zero for no limit.
    task_allocation_operation_index : int
        For a task allocation contract mismatch, the position in the task's
        allocation contract where it occurred; zero otherwise.
    task_allocation_expected_ordinal : int or None
        The allocation ordinal the contract expected there; None without a
        mismatch.
    task_allocation_actual_ordinal : int or None
        The allocation ordinal the task made there; None without a mismatch.
    task_allocation_expected_requested_bytes : int
        Bytes the contract expected that allocation to ask for.
    task_allocation_actual_requested_bytes : int
        Bytes it asked for.
    task_allocation_expected_charged_bytes : int
        Bytes the contract expected it to be charged.
    task_allocation_actual_charged_bytes : int
        Bytes it was charged.
    task_allocation_expected_alignment_bytes : int
        Alignment the contract expected, in bytes.
    task_allocation_actual_alignment_bytes : int
        Alignment it asked for, in bytes.
    task_allocation_expected_operation : int
        The operation the contract expected: 0 to allocate, 1 to free, 255
        for the end of the task. 255 without a mismatch.
    task_allocation_actual_operation : int
        The operation the task made, coded the same way.
    reason : int
        The runtime's code for why it refused; zero when the status says it
        all.
    reason_name : str
        The runtime's name for ``reason``; empty for zero.
    pool_id : int or None
        The pool the failure concerns, if any.
    """

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
    task_allocation_operation_index: int = 0
    task_allocation_expected_ordinal: int | None = None
    task_allocation_actual_ordinal: int | None = None
    task_allocation_expected_requested_bytes: int = 0
    task_allocation_actual_requested_bytes: int = 0
    task_allocation_expected_charged_bytes: int = 0
    task_allocation_actual_charged_bytes: int = 0
    task_allocation_expected_alignment_bytes: int = 0
    task_allocation_actual_alignment_bytes: int = 0
    task_allocation_expected_operation: int = 255
    task_allocation_actual_operation: int = 255
    reason: int = 0
    reason_name: str = ""
    pool_id: int | None = None

    @property
    def is_allocator_oom(self) -> bool:
        """Whether an allocation request found no memory.

        True when the status is ``out_of_memory`` or ``no_progress`` and the
        failure concerns a request for at least one byte.
        """

        return self.status in {_OUT_OF_MEMORY, _NO_PROGRESS} and (
            self.requested_bytes > 0
        )

    @property
    def is_shadowspill_contract_failure(self) -> bool:
        """Whether execution departed from its admitted plan or a task's contract.

        True for ``plan_violation``, ``task_allocation_envelope_exceeded`` and
        ``task_allocation_contract_mismatch``. Such a failure, not an error the
        framework raises after it, is the cause to report.
        """

        return self.status in {
            Status.PLAN_VIOLATION,
            _TASK_ALLOCATION_ENVELOPE_EXCEEDED,
            _TASK_ALLOCATION_CONTRACT_MISMATCH,
        }

    def as_dict(self) -> dict[str, object]:
        """Return the record as a JSON-compatible dictionary.

        Returns
        -------
        dict
            Every field by name; each value is a str, an int or None.
        """
        return {
            "operation": self.operation,
            "status": self.status,
            "status_name": self.status_name,
            "reason": self.reason,
            "reason_name": self.reason_name,
            "device_ordinal": self.device_ordinal,
            "pool_id": self.pool_id,
            "requested_bytes": self.requested_bytes,
            "free_bytes": self.free_bytes,
            "largest_free_range_bytes": self.largest_free_range_bytes,
            "object_id": self.object_id,
            "allocation_id": self.allocation_id,
            "task_id": self.task_id,
            "task_live_requested_bytes": self.task_live_requested_bytes,
            "task_live_charged_bytes": self.task_live_charged_bytes,
            "task_live_requested_limit_bytes": (self.task_live_requested_limit_bytes),
            "task_live_charged_limit_bytes": self.task_live_charged_limit_bytes,
            "task_maximum_requested_allocation_bytes": (
                self.task_maximum_requested_allocation_bytes
            ),
            "task_maximum_charged_allocation_bytes": (
                self.task_maximum_charged_allocation_bytes
            ),
            "task_allocation_operation_index": (self.task_allocation_operation_index),
            "task_allocation_expected_ordinal": (self.task_allocation_expected_ordinal),
            "task_allocation_actual_ordinal": self.task_allocation_actual_ordinal,
            "task_allocation_expected_requested_bytes": (
                self.task_allocation_expected_requested_bytes
            ),
            "task_allocation_actual_requested_bytes": (
                self.task_allocation_actual_requested_bytes
            ),
            "task_allocation_expected_charged_bytes": (
                self.task_allocation_expected_charged_bytes
            ),
            "task_allocation_actual_charged_bytes": (
                self.task_allocation_actual_charged_bytes
            ),
            "task_allocation_expected_alignment_bytes": (
                self.task_allocation_expected_alignment_bytes
            ),
            "task_allocation_actual_alignment_bytes": (
                self.task_allocation_actual_alignment_bytes
            ),
            "task_allocation_expected_operation": (
                self.task_allocation_expected_operation
            ),
            "task_allocation_actual_operation": (self.task_allocation_actual_operation),
        }


class RuntimeExecutionError(RuntimeError):
    """Raised when the runtime refuses a call made against it.

    Parameters
    ----------
    message : str
        The error message.
    diagnostics : RuntimeFailureDiagnostics, optional
        What the runtime latched about the failure, when it latched anything.
    """

    def __init__(
        self,
        message: str,
        *,
        diagnostics: RuntimeFailureDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def read_allocator_failure(
    library: Any, operation: str
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
        reason=int(failure.runtime.reason),
        reason_name=_reason_name(int(failure.runtime.reason)),
        device_ordinal=int(failure.device_ordinal),
        pool_id=(
            None
            if int(failure.runtime.pool_id) == (1 << 32) - 1
            else int(failure.runtime.pool_id)
        ),
        requested_bytes=requested,
        free_bytes=int(failure.runtime.free_bytes),
        largest_free_range_bytes=int(failure.runtime.largest_free_range_bytes),
        object_id=optional_id(int(failure.runtime.object_id)),
        allocation_id=optional_id(int(failure.runtime.allocation_id)),
        task_id=optional_id(int(failure.runtime.task_id)),
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
        task_allocation_operation_index=int(
            failure.runtime.task_allocation_operation_index
        ),
        task_allocation_expected_ordinal=optional_id(
            int(failure.runtime.task_allocation_expected_ordinal)
        ),
        task_allocation_actual_ordinal=optional_id(
            int(failure.runtime.task_allocation_actual_ordinal)
        ),
        task_allocation_expected_requested_bytes=int(
            failure.runtime.task_allocation_expected_requested_bytes
        ),
        task_allocation_actual_requested_bytes=int(
            failure.runtime.task_allocation_actual_requested_bytes
        ),
        task_allocation_expected_charged_bytes=int(
            failure.runtime.task_allocation_expected_charged_bytes
        ),
        task_allocation_actual_charged_bytes=int(
            failure.runtime.task_allocation_actual_charged_bytes
        ),
        task_allocation_expected_alignment_bytes=int(
            failure.runtime.task_allocation_expected_alignment_bytes
        ),
        task_allocation_actual_alignment_bytes=int(
            failure.runtime.task_allocation_actual_alignment_bytes
        ),
        task_allocation_expected_operation=int(
            failure.runtime.task_allocation_expected_operation
        ),
        task_allocation_actual_operation=int(
            failure.runtime.task_allocation_actual_operation
        ),
    )


def generic_runtime_error(
    diagnostics: RuntimeFailureDiagnostics,
) -> RuntimeExecutionError:
    """Build an error for an explicit runtime API rejection."""

    lines = [
        f"{diagnostics.operation} failed with status {diagnostics.status} "
        f"({diagnostics.status_name})"
    ]
    if diagnostics.task_id is not None:
        lines.append(f"runtime_task: {diagnostics.task_id}")
    if diagnostics.pool_id is not None:
        lines.append(f"pool: {diagnostics.pool_id}")
    if diagnostics.reason_name:
        lines.append(f"reason: {diagnostics.reason_name}")
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
                f"task_live_charged_limit: {diagnostics.task_live_charged_limit_bytes}",
                "task_maximum_requested_allocation: "
                f"{diagnostics.task_maximum_requested_allocation_bytes}",
                "task_maximum_charged_allocation: "
                f"{diagnostics.task_maximum_charged_allocation_bytes}",
            )
        )
    if diagnostics.status == _TASK_ALLOCATION_CONTRACT_MISMATCH:
        lines.extend(
            (
                "reason: TASK_ALLOCATION_CONTRACT_MISMATCH",
                "task_allocation_operation_index: "
                f"{diagnostics.task_allocation_operation_index}",
                "expected_operation: "
                f"{_allocation_operation_name(diagnostics.task_allocation_expected_operation)}",
                "actual_operation: "
                f"{_allocation_operation_name(diagnostics.task_allocation_actual_operation)}",
                f"expected_ordinal: {diagnostics.task_allocation_expected_ordinal}",
                f"actual_ordinal: {diagnostics.task_allocation_actual_ordinal}",
                "expected_requested: "
                f"{diagnostics.task_allocation_expected_requested_bytes}",
                "actual_requested: "
                f"{diagnostics.task_allocation_actual_requested_bytes}",
                "expected_charged: "
                f"{diagnostics.task_allocation_expected_charged_bytes}",
                f"actual_charged: {diagnostics.task_allocation_actual_charged_bytes}",
                "expected_alignment: "
                f"{diagnostics.task_allocation_expected_alignment_bytes}",
                "actual_alignment: "
                f"{diagnostics.task_allocation_actual_alignment_bytes}",
            )
        )
    return RuntimeExecutionError("\n".join(lines), diagnostics=diagnostics)


def wait_allocator_idle(
    library: Any, runtime_handle: int, *, problem: str
) -> str | None:
    """Drain the runtime, and say what stayed outstanding if it will not.

    Returns None once the runtime is idle. Callers raise their own error
    rather than sharing one, because the phases that drain -- installing,
    profiling, planning -- report failure in their own terms. What they do
    share is the description, since the counters that explain a runtime that
    will not settle are the same wherever it happens. The drain is the neutral
    runtime's; the adapter library is only asked for those counters.
    """

    status = int(runtime_library().shadowspill_runtime_wait_idle(runtime_handle))
    if status == 0:
        return None
    detail = f"status={status}"
    if hasattr(library, "shadowspill_pytorch_allocator_statistics"):
        statistics = AdapterStatistics()
        if (
            int(
                library.shadowspill_pytorch_allocator_statistics(
                    ctypes.byref(statistics)
                )
            )
            == 0
        ):
            runtime = statistics.runtime
            detail = (
                f"{detail} pending={runtime.pending_retirements} "
                f"fenced={runtime.retirement_records_fenced} "
                f"evented={runtime.retirement_records_evented} "
                f"preparing={runtime.retirement_records_preparing} "
                f"unfenced={runtime.retirement_records_unfenced} "
                f"actions={runtime.queued_actions}"
            )
    return f"the runtime did not become idle during {problem}: {detail}"


def _allocation_operation_name(value: int) -> str:
    return _ALLOCATION_OPERATIONS.get(value, f"unknown_{value}")


__all__ = [
    "RuntimeExecutionError",
    "RuntimeFailureDiagnostics",
    "generic_runtime_error",
    "read_allocator_failure",
]
