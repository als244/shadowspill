"""Fresh-process CUDA transfer/compute-overlap and multi-wait canary."""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import torch

from shadowspill.pytorch.runtime_adapter.abi import (
    AdapterStatistics,
    ExecutionDescription,
    ObjectBinding,
    ObjectSnapshot,
    RuntimeAction,
)
from shadowspill.pytorch.runtime_adapter.allocator import install_allocator

ELEMENTS = 16 << 20
BYTES_PER_OBJECT = ELEMENTS * 4


def _require_ok(status: int, operation: str) -> None:
    if status != 0:
        raise AssertionError(f"{operation} failed with status {status}")


def _promote(library: object, tensor: torch.Tensor, object_id: int) -> ObjectBinding:
    binding = ObjectBinding()
    _require_ok(
        int(
            library.shadowspill_pytorch_promote_allocation(
                object_id,
                tensor.data_ptr(),
                tensor.untyped_storage().nbytes(),
                ctypes.byref(binding),
            )
        ),
        "allocation promotion",
    )
    return binding


def _snapshot(library: object, object_id: int) -> ObjectSnapshot:
    snapshot = ObjectSnapshot()
    _require_ok(
        int(
            library.shadowspill_pytorch_object_snapshot(
                object_id, ctypes.byref(snapshot)
            )
        ),
        "object snapshot",
    )
    return snapshot


def _create_plan(library: object) -> int:
    handle = ctypes.c_size_t()
    _require_ok(
        int(library.shadowspill_pytorch_plan_create(0, 1, ctypes.byref(handle))),
        "plan creation",
    )
    if handle.value == 0:
        raise AssertionError("plan creation returned a null handle")
    return int(handle.value)


def _bind_plan_object(library: object, plan: int, object_id: int) -> None:
    handle = ctypes.c_size_t()
    _require_ok(
        int(
            library.shadowspill_pytorch_object_handle_acquire(
                object_id, ctypes.byref(handle)
            )
        ),
        "object handle acquisition",
    )
    try:
        _require_ok(
            int(
                library.shadowspill_pytorch_plan_bind_object(
                    plan, object_id, handle.value, 0
                )
            ),
            "plan object binding",
        )
    finally:
        library.shadowspill_pytorch_object_handle_release(handle.value)


def _submit_actions(
    library: object,
    plan: int,
    batch_id: int,
    stream: int,
    actions: tuple[RuntimeAction, ...],
) -> None:
    for object_id in dict.fromkeys(action.object_id for action in actions):
        _bind_plan_object(library, plan, object_id)
    action_array = (RuntimeAction * len(actions))(*actions)
    handle = ctypes.c_size_t()
    _require_ok(
        int(
            library.shadowspill_pytorch_plan_admit_action_batch(
                plan,
                batch_id,
                action_array,
                len(actions),
                ctypes.byref(handle),
            )
        ),
        "action admission",
    )
    _require_ok(
        int(
            library.shadowspill_pytorch_submit_action_batch_handle(
                handle.value, stream
            )
        ),
        "action submission",
    )


def _admit_task(
    library: object,
    plan: int,
    task_id: int,
    inputs: tuple[int, ...],
    actions: tuple[RuntimeAction, ...],
) -> int:
    for object_id in dict.fromkeys(
        (*inputs, *(action.object_id for action in actions))
    ):
        _bind_plan_object(library, plan, object_id)
    input_array = (ctypes.c_uint64 * len(inputs))(*inputs)
    action_array = (RuntimeAction * len(actions))(*actions)
    description = ExecutionDescription(
        task_id=task_id,
        input_object_ids=input_array,
        input_count=len(inputs),
        actions=action_array,
        action_count=len(actions),
    )
    handle = ctypes.c_size_t()
    _require_ok(
        int(
            library.shadowspill_pytorch_plan_admit_task(
                plan, ctypes.byref(description), ctypes.byref(handle)
            )
        ),
        "task admission",
    )
    if handle.value == 0:
        raise AssertionError("task admission returned a null handle")
    return int(handle.value)


def _action(
    object_id: int,
    kind: int,
    semantic_label: str,
) -> RuntimeAction:
    return RuntimeAction(
        object_id=object_id,
        kind=kind,
        trace_label=semantic_label.encode("utf-8"),
    )


def main() -> int:
    adapter_path = Path(sys.argv[1]).resolve()
    if torch.cuda.is_initialized():
        raise AssertionError("canary must start before PyTorch CUDA initialization")
    installed = install_allocator(
        adapter_path,
        device_ordinal=0,
        device_budget_bytes=2 << 30,
        provider_headroom_bytes=512 << 20,
        spill_pool_bytes=256 << 20,
        worker_poll_nanoseconds=10_000,
    )
    library = installed.library
    plan = _create_plan(library)
    _require_ok(
        int(library.shadowspill_pytorch_profiler_annotations_set(1)),
        "enable profiler annotations",
    )
    first = torch.full((ELEMENTS,), 1.0, device="cuda")
    second = torch.full((ELEMENTS,), 2.0, device="cuda")
    third = torch.full((ELEMENTS,), 3.0, device="cuda")
    first_binding = _promote(library, first, 2001)
    second_binding = _promote(library, second, 2002)
    third_binding = _promote(library, third, 2003)
    compute = torch.cuda.current_stream()
    stream_address = compute.cuda_stream
    torch.cuda._sleep(1_000)
    compute.synchronize()

    torch.ops.shadowspill._rebind_storage(
        second, 0, second_binding.object_id, second_binding.generation
    )
    torch.ops.shadowspill._rebind_storage(
        third, 0, third_binding.object_id, third_binding.generation
    )
    _submit_actions(
        library,
        plan,
        200,
        stream_address,
        (
            _action(
                second_binding.object_id,
                1,
                "shadowspill.runtime.transfer.evict.second_tensor."
                "role_activation.bytes_67108864.from_output."
                "execution_000200.initial_offload",
            ),
            _action(
                third_binding.object_id,
                1,
                "shadowspill.runtime.transfer.evict.third_tensor."
                "role_activation.bytes_67108864.from_output."
                "execution_000200.initial_offload",
            ),
        ),
    )
    _require_ok(
        int(library.shadowspill_pytorch_allocator_wait_idle()),
        "initial offload drain",
    )
    baseline = AdapterStatistics()
    _require_ok(
        int(library.shadowspill_pytorch_allocator_statistics(ctypes.byref(baseline))),
        "baseline statistics",
    )

    torch.ops.shadowspill._rebind_storage(
        first, 0, first_binding.object_id, first_binding.generation
    )
    overlap_compute = torch.cuda.Stream()
    overlap_compute.wait_stream(compute)
    with torch.cuda.stream(overlap_compute):
        torch.cuda._sleep(3_000_000_000)
    torch.cuda._sleep(500_000_000)
    _submit_actions(
        library,
        plan,
        201,
        stream_address,
        (
            _action(
                first_binding.object_id,
                1,
                "shadowspill.runtime.transfer.evict.first_tensor."
                "role_activation.bytes_67108864.from_output."
                "execution_000201.overlap_producer",
            ),
            _action(
                second_binding.object_id,
                2,
                "shadowspill.runtime.transfer.fetch.second_tensor."
                "role_activation.bytes_67108864.for_input."
                "execution_000202.two_input_consumer",
            ),
            _action(
                third_binding.object_id,
                2,
                "shadowspill.runtime.transfer.fetch.third_tensor."
                "role_activation.bytes_67108864.for_input."
                "execution_000202.two_input_consumer",
            ),
        ),
    )

    release_actions = (
        RuntimeAction(object_id=second_binding.object_id, kind=0),
        RuntimeAction(object_id=third_binding.object_id, kind=0),
    )
    consumer = _admit_task(
        library,
        plan,
        202,
        (second_binding.object_id, third_binding.object_id),
        release_actions,
    )
    rebound = (ObjectBinding * 2)()
    _require_ok(
        int(
            library.shadowspill_pytorch_before_task_handle(
                consumer,
                202,
                stream_address,
                rebound,
                2,
            )
        ),
        "two-input acquisition",
    )
    torch.ops.shadowspill._rebind_storage(
        second,
        rebound[0].pointer,
        rebound[0].object_id,
        rebound[0].generation,
    )
    torch.ops.shadowspill._rebind_storage(
        third,
        rebound[1].pointer,
        rebound[1].object_id,
        rebound[1].generation,
    )

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        first_state = _snapshot(library, first_binding.object_id)
        second_state = _snapshot(library, second_binding.object_id)
        third_state = _snapshot(library, third_binding.object_id)
        if (
            first_state.residency == 0
            and second_state.residency == 1
            and third_state.residency == 1
        ):
            break
        time.sleep(0.001)
    else:
        raise AssertionError("concurrent transfer actions did not complete")
    if overlap_compute.query():
        raise AssertionError("transfers completed only after unrelated compute drained")

    overlap = AdapterStatistics()
    _require_ok(
        int(library.shadowspill_pytorch_allocator_statistics(ctypes.byref(overlap))),
        "overlap statistics",
    )
    if overlap.cuda.stream_synchronizations != baseline.cuda.stream_synchronizations:
        raise AssertionError("steady task actions synchronized a transfer stream")
    inserted_waits = (
        overlap.runtime.wait_events_inserted - baseline.runtime.wait_events_inserted
    )
    if inserted_waits != 2:
        raise AssertionError(
            "two unavailable inputs did not produce two stream waits: "
            f"observed {inserted_waits}"
        )
    if overlap.runtime.evict_transfers - baseline.runtime.evict_transfers != 1:
        raise AssertionError("overlap interval did not perform one EVICT transfer")
    if overlap.runtime.fetch_transfers - baseline.runtime.fetch_transfers != 2:
        raise AssertionError("overlap interval did not perform two FETCH transfers")

    compute.synchronize()
    overlap_compute.synchronize()
    torch.testing.assert_close(second[:1024].cpu(), torch.full((1024,), 2.0))
    torch.testing.assert_close(third[-1024:].cpu(), torch.full((1024,), 3.0))
    torch.ops.shadowspill._rebind_storage(
        second, 0, rebound[0].object_id, rebound[0].generation
    )
    torch.ops.shadowspill._rebind_storage(
        third, 0, rebound[1].object_id, rebound[1].generation
    )
    _require_ok(
        int(
            library.shadowspill_pytorch_after_task_handle(
                consumer, 202, stream_address
            )
        ),
        "consumer publication",
    )
    _require_ok(
        int(library.shadowspill_pytorch_allocator_wait_idle()),
        "final release drain",
    )
    _require_ok(
        int(library.shadowspill_pytorch_profiler_annotations_set(0)),
        "disable profiler annotations",
    )
    _require_ok(int(library.shadowspill_pytorch_plan_close(plan)), "plan close")
    library.shadowspill_pytorch_plan_destroy(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
