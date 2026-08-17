"""Fresh-process fail-fast allocation-contract mismatch canary."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch

from shadowspill.pytorch.runtime_adapter.abi import (
    TaskAllocationContractStep,
    TaskDescription,
)
from shadowspill.pytorch.runtime_adapter.allocator import install_allocator
from shadowspill.pytorch.runtime_adapter.failures import (
    ExecutionTaskIdentity,
    read_allocator_failure,
)

TASK_ID = 17
CONTRACT_MISMATCH = 11


def _admit_task(library: object, description: TaskDescription) -> int:
    plan = ctypes.c_size_t()
    status = int(library.shadowspill_pytorch_plan_create(0, 1, ctypes.byref(plan)))
    if status != 0 or plan.value == 0:
        raise AssertionError(
            f"allocation-contract plan creation failed with status {status}"
        )
    task = ctypes.c_size_t()
    status = int(
        library.shadowspill_pytorch_plan_admit_task(
            plan.value, ctypes.byref(description), ctypes.byref(task)
        )
    )
    if status != 0 or task.value == 0:
        raise AssertionError(f"execution admission failed with status {status}")
    return int(task.value)


def main() -> int:
    installed = install_allocator(
        Path(sys.argv[1]).resolve(),
        device_ordinal=0,
        device_budget_bytes=1 << 30,
        provider_headroom_bytes=512 << 20,
        spill_pool_bytes=1 << 20,
        worker_poll_nanoseconds=1_000,
    )
    library = installed.library
    expected = (TaskAllocationContractStep * 1)(
        TaskAllocationContractStep(
            allocation_ordinal=0,
            requested_bytes=4096,
            charged_bytes=4096,
            alignment_bytes=256,
            operation=0,
            required=1,
        )
    )
    description = TaskDescription(
        task_id=TASK_ID,
        input_object_ids=None,
        input_count=0,
        updates=None,
        update_count=0,
        actions=None,
        action_count=0,
        allocation_contract_steps=expected,
        allocation_contract_step_count=1,
        enforce_allocation_contract=1,
        maximum_requested_allocation_bytes=8192,
        maximum_charged_allocation_bytes=8192,
        live_requested_allocation_limit_bytes=8192,
        live_charged_allocation_limit_bytes=8192,
    )
    task_handle = _admit_task(library, description)
    labels = (ctypes.c_char_p * (TASK_ID + 1))()
    labels[TASK_ID] = b"execution_000017.canary.stage_0000.forward"
    status = int(library.shadowspill_pytorch_task_labels_configure(labels, TASK_ID + 1))
    if status != 0:
        raise AssertionError(f"task label configuration failed with status {status}")
    stream = torch.cuda.current_stream()
    status = int(
        library.shadowspill_pytorch_before_task_handle(
            task_handle, TASK_ID, stream.cuda_stream, None, 0
        )
    )
    if status != 0:
        raise AssertionError(f"task entry failed with status {status}")

    try:
        torch.empty((24,), dtype=torch.uint8, device="cuda")
    except RuntimeError as cause:
        message = str(cause)
        for expected_text in (
            "ShadowSpill allocator callback failed",
            "status: 11 (task allocation contract mismatch)",
            "reason: TASK_ALLOCATION_CONTRACT_MISMATCH",
            "execution_task: execution_000017",
            "semantic_task: canary.stage_0000.forward",
            "canonical_task: task_000017",
            "expected_requested: 4096",
            "actual_requested: 24",
        ):
            if expected_text not in message:
                raise AssertionError(
                    f"direct contract failure omitted {expected_text!r}: {message}"
                ) from cause
        status = int(
            library.shadowspill_pytorch_abort_task_handle(
                task_handle, TASK_ID
            )
        )
        if status != 0:
            raise AssertionError(
                f"failed to abort allocation-contract task: status={status}"
            ) from None
        diagnostics = read_allocator_failure(
            library,
            "execute mismatched allocation task",
            task=ExecutionTaskIdentity(
                "execution_000017",
                "canary.stage_0000.forward",
                "task_000017",
            ),
        )
        if diagnostics is None or diagnostics.status != CONTRACT_MISMATCH:
            raise AssertionError(
                "allocation contract mismatch was not preserved"
            ) from cause
    else:
        raise AssertionError(
            "allocation contract mismatch returned invalid storage to its caller"
        )

    # No kernel consumed a null pointer, so the CUDA context remains healthy
    # and ordinary teardown cannot turn the structured error into SIGABRT.
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
