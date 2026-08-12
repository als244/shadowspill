# CUDA backend C API

`libshadowspill_backend_cuda.so` implements the public neutral backend vtable
declared in `backend.h`. Vendor-facing construction, capability, diagnostics,
and stream-wrapping declarations live in `backend_cuda.h`.

The backend retains the selected CUDA primary context and rejects a different
already-current context. The backend object must outlive every runtime that
borrows its vtable. Destroying it releases only ShadowSpill's primary-context
reference; it neither resets the device nor owns PyTorch's context reference.

One runtime creation results in one conventional `cuMemAlloc` for the complete
slab and one `cuMemHostAlloc` for the complete pinned arena. Stream and event
operations use nonblocking CUDA streams and timing-disabled events. Copies,
event recording/query, and stream waits are asynchronous. Only the runtime's
explicit close path calls stream synchronization.

`shadowspill_cuda_backend_profiler()` returns the neutral profiler vtable
implemented with NVTX. It names the runtime worker and names the two transfer
streams `shadowspill_fetch` and `shadowspill_evict` in NSYS. NVTX headers and
symbols do not enter the neutral runtime or PyTorch storage-adapter source.

The operation ledger is intended for admission and qualification. It reports
conventional allocation/free counts, pinned allocation/free counts, transfers,
events, waits, synchronizations, and context activations. A successful sealed
steady state must not increase either allocation count.

`shadowspill_cuda_physical_memory` uses NVML's per-process accounting and
device ledger. The process value includes the CUDA context, complete physical
slab, and allocations made directly by providers. It is independent of which
slab ranges are logically live. ShadowSpill treats missing per-process
accounting as an admission failure rather than silently weakening the public
physical cap.
