"""Fresh-process diagnostic OOM canary for the PyTorch callback boundary."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch

from shadowspill.pytorch.runtime_adapter.abi import AdapterFailure, TaskDescription
from shadowspill.pytorch.runtime_adapter.allocator import install_allocator
from shadowspill.pytorch.runtime_adapter.failures import (
    ExecutionTaskIdentity,
    RuntimeExecutionError,
    allocator_oom_error,
    read_allocator_failure,
)
from tests.integration.pytorch.runtime_helpers import begin_task

NO_PROGRESS = 4
REQUEST_BYTES = 128 << 20


def _admit_task(library: object, description: TaskDescription) -> tuple[int, int]:
    plan = ctypes.c_size_t()
    status = int(library.shadowspill_pytorch_plan_create(0, 1, ctypes.byref(plan)))
    if status != 0 or plan.value == 0:
        raise AssertionError(f"failed to create OOM canary plan: status={status}")
    task = ctypes.c_size_t()
    status = int(
        library.shadowspill_pytorch_plan_admit_task(
            plan.value, ctypes.byref(description), ctypes.byref(task)
        )
    )
    if status != 0 or task.value == 0:
        raise AssertionError(f"failed to admit OOM canary task: status={status}")
    return int(plan.value), int(task.value)


def main() -> int:
    installed = install_allocator(
        Path(sys.argv[1]).resolve(),
        device_ordinal=0,
        device_budget_bytes=1 << 30,
        provider_headroom_bytes=512 << 20,
        spill_pool_bytes=1 << 20,
        worker_poll_nanoseconds=10_000,
    )
    task_id = 17
    description = TaskDescription(
        task_id=task_id,
        trace_label=b"execution_000017.dummy_model.stage_0003.forward",
        input_object_ids=None,
        input_count=0,
        updates=None,
        update_count=0,
        actions=None,
        action_count=0,
        allocation_contract_steps=None,
        allocation_contract_step_count=0,
        enforce_allocation_contract=0,
        maximum_requested_allocation_bytes=0,
        maximum_charged_allocation_bytes=0,
        live_requested_allocation_limit_bytes=0,
        live_charged_allocation_limit_bytes=0,
    )
    plan_handle, task_handle = _admit_task(installed.library, description)
    stream = torch.cuda.current_stream()
    begin_task(
        installed.library,
        task_handle,
        task_id,
        stream.cuda_stream,
        expected_bindings=0,
    )
    try:
        torch.empty((REQUEST_BYTES,), dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError as error:
        message = str(error)
        for expected in (
            "ShadowSpill no-progress OOM",
            "execution_task: execution_000017",
            "semantic_task: dummy_model.stage_0003.forward",
            "canonical_task: task_000017",
            f"requested: {REQUEST_BYTES}",
        ):
            if expected not in message:
                raise AssertionError(
                    f"direct OOM omitted {expected!r}: {message}"
                ) from error
        status = int(
            installed.library.shadowspill_pytorch_abort_task_handle(task_handle)
        )
        if status != 0:
            raise AssertionError(f"failed to abort OOM task: status={status}") from None
    else:
        raise AssertionError("failed callback returned a tensor to its caller")
    failure = AdapterFailure()
    status = int(
        installed.library.shadowspill_pytorch_allocator_failure(ctypes.byref(failure))
    )
    if status != NO_PROGRESS or failure.status != NO_PROGRESS:
        raise AssertionError(f"unexpected adapter failure status: {status}")
    if failure.requested_bytes != REQUEST_BYTES:
        raise AssertionError("adapter lost the requested allocation size")
    if failure.runtime.status != NO_PROGRESS:
        raise AssertionError("adapter did not preserve the runtime's first cause")
    if failure.runtime.free_bytes != installed.admission.execution_pool_bytes:
        raise AssertionError("diagnostic free-space accounting is incorrect")
    task = ExecutionTaskIdentity(
        execution_task_id="execution_000017",
        semantic_name="dummy_model.stage_0003.forward",
        canonical_task_id="task_000017",
    )
    diagnostics = read_allocator_failure(
        installed.library,
        "allocate dummy-model task workspace",
        task=task,
    )
    if diagnostics is None:
        raise AssertionError("public failure translation lost the allocator failure")
    try:
        raise allocator_oom_error(diagnostics)
    except RuntimeExecutionError as error:
        message = str(error)
        for expected in (
            "ShadowSpill no-progress OOM",
            "execution_task: execution_000017",
            "semantic_task: dummy_model.stage_0003.forward",
            "canonical_task: task_000017",
            f"requested: {REQUEST_BYTES}",
        ):
            if expected not in message:
                raise AssertionError(
                    f"structured OOM omitted {expected!r}: {message}"
                ) from error
        if error.diagnostics is not diagnostics:
            raise AssertionError(
                "structured OOM did not retain its diagnostics"
            ) from error
    torch.cuda.synchronize()
    if int(installed.library.shadowspill_pytorch_recover_no_progress()) != 0:
        raise AssertionError("failed to recover no-progress for teardown")
    if (
        int(
            installed.library.shadowspill_pytorch_allocator_failure(
                ctypes.byref(failure)
            )
        )
        != 0
    ):
        raise AssertionError("adapter failure remained latched after recovery")
    probe = torch.empty((1024,), dtype=torch.uint8, device="cuda")
    if probe.data_ptr() == 0:
        raise AssertionError("allocator remained unusable after recovery")
    if int(installed.library.shadowspill_pytorch_plan_close(plan_handle)) != 0:
        raise AssertionError("failed to close OOM canary plan")
    installed.library.shadowspill_pytorch_plan_destroy(plan_handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
