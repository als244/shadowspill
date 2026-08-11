"""Fresh-process CUDA pluggable-allocator canary."""

from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path

import torch

from shadowspill.pytorch._abi import AdapterStatistics
from shadowspill.pytorch._allocator import install_allocator


def main() -> int:
    adapter_path = Path(sys.argv[1]).resolve()
    if torch.cuda.is_initialized():
        raise AssertionError("canary must start before PyTorch CUDA initialization")
    installed = install_allocator(
        adapter_path,
        device_ordinal=0,
        device_slab_bytes=256 << 20,
        host_arena_bytes=16 << 20,
        progress_poll_nanoseconds=10_000,
    )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
