"""Fresh-process diagnostic OOM canary for the PyTorch callback boundary."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch

from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.runtime.abi import (
    AdapterFailure,
    AdapterStatistics,
    PlanDescription,
    TaskDescription,
    runtime_library,
)
from shadowspill.runtime.bootstrap import install_runtime
from shadowspill.runtime.failures import format_bytes, read_allocator_failure
from shadowspill.status import Status
from tests.integration.pytorch.runtime_helpers import begin_task, two_pool_topology

NO_PROGRESS = Status.NO_PROGRESS
# Above the whole device budget, so it cannot fit whatever the context baseline costs.
REQUEST_BYTES = 4 << 30
LABEL = "example_0000.stage_0003.save.forward"


def _runtime_handle(library: object) -> int:
    handle = ctypes.c_size_t()
    status = int(library.shadowspill_pytorch_runtime_handle(ctypes.byref(handle)))
    if status != 0 or handle.value == 0:
        raise AssertionError(f"runtime handle unavailable (status {status})")
    return int(handle.value)


def _plan_description(library: object) -> PlanDescription:
    """A description naming an id this runtime issued.

    Plan creation refuses an id it did not hand out, so the id is taken rather
    than written in, and every field is named: the description's first member is
    the plan id, so a positional construction silently shifts the pool roles.
    """

    plan_id = ctypes.c_uint64()
    status = int(
        runtime_library().shadowspill_runtime_next_plan_id(
            _runtime_handle(library), ctypes.byref(plan_id)
        )
    )
    if status != 0 or plan_id.value == 0:
        raise AssertionError(f"taking a plan id failed with status {status}")
    return PlanDescription(
        plan_id=int(plan_id.value),
        execution_pool_id=0,
        spill_pool_id=1,
        fetch_route_id=0,
        evict_route_id=1,
    )


def _admit_task(library: object, description: TaskDescription) -> tuple[int, int]:
    plan = ctypes.c_size_t()
    status = int(
        runtime_library().shadowspill_plan_create(
            _runtime_handle(library),
            ctypes.byref(_plan_description(library)),
            ctypes.byref(plan),
        )
    )
    if status != 0 or plan.value == 0:
        raise AssertionError(f"failed to create OOM canary plan: status={status}")
    task = ctypes.c_size_t()
    status = int(
        runtime_library().shadowspill_plan_admit_task(
            plan.value, ctypes.byref(description), ctypes.byref(task)
        )
    )
    if status != 0 or task.value == 0:
        raise AssertionError(f"failed to admit OOM canary task: status={status}")
    return int(plan.value), int(task.value)


def main() -> int:
    installed = install_runtime(
        Path(sys.argv[1]).resolve(),
        frontend=PyTorchFrontend(),
        device_ordinal=0,
        device_budget_bytes=2 << 30,
        external_headroom_bytes=512 << 20,
        **two_pool_topology(1 << 20),
        worker_poll_nanoseconds=10_000,
    )
    task_id = 17
    description = TaskDescription(
        task_id=task_id,
        trace_label=LABEL.encode(),
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
    # In a pool large enough for provider initialization, install leaves the
    # cuBLAS workspace resident, so the free space to preserve is the
    # runtime's own measurement, not the raw pool capacity.
    statistics = AdapterStatistics()
    if (
        int(
            installed.library.shadowspill_pytorch_allocator_statistics(
                ctypes.byref(statistics)
            )
        )
        != 0
    ):
        raise AssertionError("pre-request statistics query failed")
    free_before_request = int(statistics.allocator_pool.free_bytes)
    try:
        torch.empty((REQUEST_BYTES,), dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError as error:
        message = str(error)
        lines = message.splitlines()
        if "out of memory" not in message:
            raise AssertionError(f"direct OOM is not an OOM: {message}") from error
        # The label is one name: printed whole, never split at its first dot.
        for expected in (
            f"semantic_task: {LABEL}",
            "canonical_task: task_000017",
            f"requested: {format_bytes(REQUEST_BYTES)}",
        ):
            if expected not in lines:
                raise AssertionError(
                    f"direct OOM omitted the line {expected!r}: {message}"
                ) from error
        if any(line.startswith("execution_task:") for line in lines):
            raise AssertionError(
                f"direct OOM split the task label: {message}"
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
    if failure.runtime.free_bytes != free_before_request:
        raise AssertionError("diagnostic free-space accounting is incorrect")
    diagnostics = read_allocator_failure(
        installed.library, "allocate dummy-model task workspace"
    )
    if diagnostics is None:
        raise AssertionError("public failure translation lost the allocator failure")
    if (
        diagnostics.status != NO_PROGRESS
        or diagnostics.task_id != task_id
        or diagnostics.requested_bytes != REQUEST_BYTES
    ):
        raise AssertionError(
            f"public failure translation lost the task or the request: {diagnostics}"
        )
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
    if int(runtime_library().shadowspill_plan_close(plan_handle)) != 0:
        raise AssertionError("failed to close OOM canary plan")
    runtime_library().shadowspill_plan_destroy(plan_handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
