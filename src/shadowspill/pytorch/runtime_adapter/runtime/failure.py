"""What a failed call leaves on the runtime, and how teardown is made safe.

A failure is latched once, as `RuntimeFailureDiagnostics`, and a runtime that
cannot be trusted after it is marked unusable with the reason; `close` and every
later call read both. `prepare_failure_cleanup` is the step every failing
planning call and callable runs before rolling back: record the failure, drain
the device, recover the no-progress latch where that is the failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from shadowspill.pytorch.runtime_adapter.failures import (
    RuntimeExecutionError,
    RuntimeFailureDiagnostics,
    read_allocator_failure,
)

if TYPE_CHECKING:
    from .core import Runtime


def prepare_failure_cleanup(
    runtime: Runtime,
    error: BaseException,
    *,
    operation: str,
    synchronize_unlatched: bool,
) -> None:
    """Record handle failure state and safely prepare runtime teardown."""

    if isinstance(error, RuntimeExecutionError) and not error._begin_cleanup():
        return
    diagnostics = (
        error.diagnostics if isinstance(error, RuntimeExecutionError) else None
    )
    if diagnostics is None:
        diagnostics = read_allocator_failure(runtime._installed.library, operation)
    if diagnostics is not None:
        record_failure(runtime, diagnostics)
    elif not synchronize_unlatched:
        return
    try:
        torch.cuda.synchronize(int(runtime._installed.admission.device_ordinal))
    except BaseException as synchronize_error:
        error.add_note(
            "Failed to synchronize the execution device during fault cleanup: "
            f"{synchronize_error}"
        )
        mark_unusable(runtime, "execution-device synchronization failed")
        return
    if diagnostics is None or not diagnostics.is_recoverable_no_progress:
        return
    status = int(runtime._installed.library.shadowspill_pytorch_recover_no_progress())
    if status != 0:
        error.add_note(
            f"Failed to recover the no-progress latch for teardown: status {status}"
        )
        mark_unusable(
            runtime, f"no-progress teardown recovery failed with status {status}"
        )


def record_failure(runtime: Runtime, diagnostics: RuntimeFailureDiagnostics) -> None:
    with runtime._lock:
        runtime._last_failure = diagnostics


def mark_unusable(runtime: Runtime, reason: str) -> None:
    with runtime._lock:
        if runtime._unusable_reason is None:
            runtime._unusable_reason = reason


__all__ = ["mark_unusable", "prepare_failure_cleanup", "record_failure"]
