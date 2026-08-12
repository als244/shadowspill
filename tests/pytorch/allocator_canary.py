"""Fresh-process CUDA pluggable-allocator canary."""

from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path

import torch

from shadowspill.pytorch._abi import (
    AdapterCapabilities,
    AdapterStatistics,
    Allocation,
    ObjectBinding,
    ObjectSnapshot,
    PhysicalMemory,
    RuntimeAction,
)
from shadowspill.pytorch._allocator import install_allocator, resize_host_arena


def main() -> int:
    adapter_path = Path(sys.argv[1]).resolve()
    if torch.cuda.is_initialized():
        raise AssertionError("canary must start before PyTorch CUDA initialization")
    installed = install_allocator(
        adapter_path,
        device_ordinal=0,
        device_budget_bytes=2 << 30,
        provider_headroom_bytes=512 << 20,
        host_arena_bytes=16 << 20,
        worker_poll_nanoseconds=10_000,
    )
    resize_host_arena(installed, host_arena_bytes=32 << 20, host_budget_bytes=64 << 20)
    if installed.admission.host_arena_bytes != 32 << 20:
        raise AssertionError("planning-time host admission did not grow")
    capabilities = AdapterCapabilities()
    installed.library.shadowspill_pytorch_adapter_capabilities(
        ctypes.byref(capabilities)
    )
    if capabilities.storage_rebinding != 1:
        raise AssertionError("canary requires the version-pinned storage adapter")
    if capabilities.runtime_trace != 1:
        raise AssertionError("adapter lacks bounded runtime tracing")
    source = torch.full((1024, 1024), 3.0, device="cuda")
    warm = source + 1.0
    torch.cuda.synchronize()
    del warm
    gc.collect()
    source_pointer = source.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        result = source + 2.0
    source.record_stream(stream)
    del source
    gc.collect()
    before_replacement = AdapterStatistics()
    installed.library.shadowspill_pytorch_allocator_statistics(
        ctypes.byref(before_replacement)
    )
    stream_complete_before = stream.query()
    if stream_complete_before or before_replacement.runtime.pending_retirements != 1:
        raise AssertionError("stream-retirement precondition was not established")
    replacement = torch.empty((1024, 1024), device="cuda", dtype=torch.float32)
    if replacement.data_ptr() == source_pointer:
        debug = AdapterStatistics()
        installed.library.shadowspill_pytorch_allocator_statistics(ctypes.byref(debug))
        raise AssertionError(
            "allocator reused storage before its recorded stream: "
            f"before_complete={stream_complete_before} "
            f"before_pending={before_replacement.runtime.pending_retirements} "
            f"before_free={before_replacement.runtime.free_bytes} "
            f"stream_complete={stream.query()} "
            f"record_callbacks={debug.record_stream_callbacks} "
            f"pending={debug.runtime.pending_retirements} "
            f"slab={debug.runtime.slab_bytes} "
            f"allocated={debug.runtime.allocated_bytes} "
            f"free={debug.runtime.free_bytes} "
            f"allocations={debug.allocation_callbacks} "
            f"frees={debug.free_callbacks}"
        )
    stream.synchronize()
    torch.testing.assert_close(result.cpu(), torch.full((1024, 1024), 5.0))
    empty = torch.empty((0,), device="cuda")
    if empty.numel() != 0:
        raise AssertionError("zero-size tensor changed shape")
    del empty, replacement, result, stream
    gc.collect()
    torch.cuda.synchronize()
    library = installed.library
    library.shadowspill_pytorch_allocator_wait_idle.restype = ctypes.c_uint32
    if int(library.shadowspill_pytorch_allocator_wait_idle()) != 0:
        raise AssertionError("allocator did not become idle")
    statistics = AdapterStatistics()
    if (
        int(library.shadowspill_pytorch_allocator_statistics(ctypes.byref(statistics)))
        != 0
    ):
        raise AssertionError("statistics query failed")
    if statistics.allocation_callbacks == 0:
        raise AssertionError("PyTorch issued no allocation callbacks")
    if statistics.free_callbacks == 0:
        raise AssertionError("PyTorch issued no free callbacks")
    if statistics.record_stream_callbacks == 0:
        raise AssertionError("record-stream callback was not installed")
    if statistics.pointer_lookup_failures != 0 or statistics.callback_failures != 0:
        raise AssertionError("a PyTorch callback missed the runtime allocation table")
    if statistics.cuda.device_allocations != 1:
        raise AssertionError("CUDA backend did not use exactly one device slab")

    same_stream = torch.full((1024, 1024), 6.0, device="cuda")
    same_stream.add_(1.0)
    same_stream_pointer = same_stream.data_ptr()
    del same_stream
    gc.collect()
    same_stream_replacement = torch.empty(
        (1024, 1024), dtype=torch.float32, device="cuda"
    )
    if same_stream_replacement.data_ptr() != same_stream_pointer:
        raise AssertionError("same-stream logical free was not immediately reusable")
    same_stream_replacement.fill_(8.0)
    torch.testing.assert_close(
        same_stream_replacement.cpu(), torch.full((1024, 1024), 8.0)
    )
    del same_stream_replacement
    gc.collect()

    expected = torch.arange(2 << 20, dtype=torch.float32)
    parameter = torch.nn.Parameter(expected.cuda())
    view = parameter.view(1024, -1)
    parameter_identity = id(parameter)
    storage_identity = parameter.untyped_storage()._cdata
    address = parameter.data_ptr()
    size_bytes = parameter.untyped_storage().nbytes()
    binding = ObjectBinding()
    status = int(
        library.shadowspill_pytorch_promote_allocation(
            1001, address, size_bytes, ctypes.byref(binding)
        )
    )
    if status != 0 or binding.pointer != address:
        raise AssertionError("ordinary PyTorch allocation promotion failed")
    try:
        torch.ops.shadowspill._rebind_storage(
            parameter, address, binding.object_id, binding.generation + 1
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("storage adapter accepted a stale generation")
    if parameter.data_ptr() != address:
        raise AssertionError("failed rebinding validation mutated the storage")
    try:
        torch.ops.shadowspill._rebind_storages(
            [parameter, parameter],
            [0, 0],
            [binding.object_id, binding.object_id],
            [binding.generation, binding.generation + 1],
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("batch adapter accepted a stale generation")
    if parameter.data_ptr() != address:
        raise AssertionError("failed batch validation partially mutated storage")
    try:
        torch.ops.shadowspill._acquire_storages(
            [parameter, parameter], [address, address + 256]
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("storage acquisition accepted a stale address")
    if parameter.data_ptr() != address:
        raise AssertionError("failed storage acquisition partially mutated storage")
    torch.ops.shadowspill._rebind_storages(
        [parameter],
        [address],
        [binding.object_id],
        [binding.generation],
    )

    compute_stream = torch.cuda.current_stream().cuda_stream
    torch.cuda._sleep(100_000_000)
    offload = (RuntimeAction * 1)(RuntimeAction(binding.object_id, 1))
    status = int(
        library.shadowspill_pytorch_after_task(
            100,
            compute_stream,
            None,
            0,
            offload,
            1,
        )
    )
    if status != 0:
        raise AssertionError(f"offload submission failed with status {status}")
    torch.ops.shadowspill._rebind_storage(
        parameter, 0, binding.object_id, binding.generation
    )
    if parameter.data_ptr() != 0 or view.data_ptr() != 0:
        raise AssertionError("alias storage was not dematerialized together")
    if int(library.shadowspill_pytorch_allocator_wait_idle()) != 0:
        raise AssertionError("offload did not complete")
    snapshot = ObjectSnapshot()
    if (
        int(
            library.shadowspill_pytorch_object_snapshot(
                binding.object_id, ctypes.byref(snapshot)
            )
        )
        != 0
        or snapshot.residency != 0
    ):
        raise AssertionError("offload did not leave a host-only object")

    blocker = torch.empty(2 << 20, dtype=torch.float32, device="cuda")
    if blocker.data_ptr() != address:
        raise AssertionError("canary failed to occupy the object's former slab range")
    prefetch = (RuntimeAction * 1)(RuntimeAction(binding.object_id, 2))
    status = int(
        library.shadowspill_pytorch_after_task(
            101,
            compute_stream,
            None,
            0,
            prefetch,
            1,
        )
    )
    if status != 0:
        raise AssertionError(f"prefetch submission failed with status {status}")
    object_ids = (ctypes.c_uint64 * 1)(binding.object_id)
    rebound = (ObjectBinding * 1)()
    status = int(
        library.shadowspill_pytorch_before_task(
            102,
            compute_stream,
            object_ids,
            1,
            rebound,
            1,
        )
    )
    if status != 0:
        raise AssertionError(
            f"prefetched input acquisition failed with status {status}"
        )
    if rebound[0].pointer in (None, 0, address, blocker.data_ptr()):
        raise AssertionError("prefetch did not lease a different slab range")
    if rebound[0].generation == binding.generation:
        raise AssertionError("prefetch did not advance the allocation generation")
    torch.ops.shadowspill._rebind_storage(
        parameter,
        rebound[0].pointer,
        rebound[0].object_id,
        rebound[0].generation,
    )
    if id(parameter) != parameter_identity:
        raise AssertionError("Parameter identity changed during storage rebinding")
    if parameter.untyped_storage()._cdata != storage_identity:
        raise AssertionError("Storage identity changed during rebinding")
    if view.untyped_storage()._cdata != storage_identity:
        raise AssertionError("view alias identity changed during rebinding")
    torch.testing.assert_close(parameter.cpu(), expected)

    torch.cuda._sleep(100_000_000)
    release = (RuntimeAction * 1)(RuntimeAction(binding.object_id, 0))
    status = int(
        library.shadowspill_pytorch_after_task(
            103,
            compute_stream,
            None,
            0,
            release,
            1,
        )
    )
    if status != 0:
        raise AssertionError(f"release submission failed with status {status}")
    torch.ops.shadowspill._rebind_storage(
        parameter, 0, rebound[0].object_id, rebound[0].generation
    )
    if int(library.shadowspill_pytorch_allocator_wait_idle()) != 0:
        raise AssertionError("planned release did not complete")
    del blocker, view, parameter
    gc.collect()
    if int(library.shadowspill_pytorch_allocator_wait_idle()) != 0:
        raise AssertionError("canary cleanup did not complete")
    final_statistics = AdapterStatistics()
    if (
        int(
            library.shadowspill_pytorch_allocator_statistics(
                ctypes.byref(final_statistics)
            )
        )
        != 0
    ):
        raise AssertionError("final statistics query failed")
    if final_statistics.runtime.transfers_to_host != 1:
        raise AssertionError("canary did not execute exactly one D2H transfer")
    if final_statistics.runtime.transfers_to_device != 1:
        raise AssertionError("canary did not execute exactly one H2D transfer")
    if final_statistics.runtime.wait_events_inserted != 1:
        raise AssertionError("prefetched input did not insert exactly one stream wait")
    if (
        final_statistics.pointer_lookup_failures != 0
        or final_statistics.callback_failures != 0
    ):
        raise AssertionError("storage relocation caused an allocator callback failure")

    host_source = torch.arange(4096, dtype=torch.uint8)
    status = int(
        library.shadowspill_pytorch_register_host_object(
            3001,
            host_source.untyped_storage().nbytes(),
            1,
            host_source.untyped_storage().data_ptr(),
        )
    )
    if status != 0:
        raise AssertionError(f"direct host registration failed with status {status}")
    host_tensor = torch.empty_like(host_source, device="cuda")
    host_binding = ObjectBinding()
    status = int(
        library.shadowspill_pytorch_bind_registered_allocation(
            3001,
            host_tensor.untyped_storage().data_ptr(),
            host_tensor.untyped_storage().nbytes(),
            ctypes.byref(host_binding),
        )
    )
    if status != 0:
        raise AssertionError(f"registered allocation bind failed with status {status}")
    host_release = (RuntimeAction * 1)(RuntimeAction(3001, 0))
    if (
        int(
            library.shadowspill_pytorch_after_task(
                300, compute_stream, None, 0, host_release, 1
            )
        )
        != 0
    ):
        raise AssertionError("initial placeholder release failed")
    torch.ops.shadowspill._rebind_storage(
        host_tensor, 0, host_binding.object_id, host_binding.generation
    )
    if int(library.shadowspill_pytorch_allocator_wait_idle()) != 0:
        raise AssertionError("host placeholder release did not drain")
    host_prefetch = (RuntimeAction * 1)(RuntimeAction(3001, 2))
    if (
        int(
            library.shadowspill_pytorch_after_task(
                301, compute_stream, None, 0, host_prefetch, 1
            )
        )
        != 0
    ):
        raise AssertionError("direct host prefetch failed")
    host_ids = (ctypes.c_uint64 * 1)(3001)
    host_rebound = (ObjectBinding * 1)()
    if (
        int(
            library.shadowspill_pytorch_before_task(
                302, compute_stream, host_ids, 1, host_rebound, 1
            )
        )
        != 0
    ):
        raise AssertionError("direct host object acquisition failed")
    torch.ops.shadowspill._rebind_storage(
        host_tensor,
        host_rebound[0].pointer,
        host_rebound[0].object_id,
        host_rebound[0].generation,
    )
    torch.testing.assert_close(host_tensor.cpu(), host_source)
    if (
        int(
            library.shadowspill_pytorch_after_task(
                302, compute_stream, None, 0, host_release, 1
            )
        )
        != 0
    ):
        raise AssertionError("direct host final release failed")
    torch.ops.shadowspill._rebind_storage(
        host_tensor, 0, host_rebound[0].object_id, host_rebound[0].generation
    )
    if int(library.shadowspill_pytorch_allocator_wait_idle()) != 0:
        raise AssertionError("direct host final release did not drain")

    caller_output = torch.arange(1024, dtype=torch.float32, device="cuda")
    caller_binding = ObjectBinding()
    if (
        int(
            library.shadowspill_pytorch_promote_allocation(
                3002,
                caller_output.untyped_storage().data_ptr(),
                caller_output.untyped_storage().nbytes(),
                ctypes.byref(caller_binding),
            )
        )
        != 0
    ):
        raise AssertionError("caller output promotion failed")
    caller_allocation = Allocation()
    if (
        int(
            library.shadowspill_pytorch_transfer_output_to_caller(
                caller_binding.object_id, ctypes.byref(caller_allocation)
            )
        )
        != 0
    ):
        raise AssertionError("output ownership transfer failed")
    if caller_allocation.pointer != caller_output.untyped_storage().data_ptr():
        raise AssertionError("caller output changed allocation during transfer")
    del caller_output, host_tensor
    gc.collect()
    if int(library.shadowspill_pytorch_allocator_wait_idle()) != 0:
        raise AssertionError("caller-owned output did not free normally")

    physical = PhysicalMemory()
    if int(library.shadowspill_pytorch_physical_memory(ctypes.byref(physical))) != 0:
        raise AssertionError("physical memory query failed")
    if physical.process_bytes > installed.admission.device_budget_bytes:
        raise AssertionError("process exceeded the physical device cap")
    if physical.process_bytes < installed.admission.slab_bytes:
        raise AssertionError("physical ledger does not include the complete slab")
    if (
        int(
            library.shadowspill_pytorch_seal_physical_budget(
                installed.admission.provider_headroom_bytes,
                64,
            )
        )
        != 0
    ):
        raise AssertionError("physical budget seal failed")
    if int(library.shadowspill_pytorch_check_physical_budget()) != 0:
        raise AssertionError("sealed physical budget check failed")
    sealed_statistics = AdapterStatistics()
    if (
        int(
            library.shadowspill_pytorch_allocator_statistics(
                ctypes.byref(sealed_statistics)
            )
        )
        != 0
    ):
        raise AssertionError("sealed statistics query failed")
    if sealed_statistics.physical_budget_sealed != 1:
        raise AssertionError("adapter did not retain the physical seal")
    if int(library.shadowspill_pytorch_resize_host_arena(48 << 20)) == 0:
        raise AssertionError("sealed adapter accepted pinned-host growth")
    if sealed_statistics.physical_checks < 3:
        raise AssertionError("adapter did not record physical reconciliation")
    if (
        sealed_statistics.peak_process_physical_bytes
        > installed.admission.device_budget_bytes
    ):
        raise AssertionError("physical peak exceeded the device cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
