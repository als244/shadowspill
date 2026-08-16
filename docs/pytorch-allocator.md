# PyTorch allocator integration

ShadowSpill installs through PyTorch's supported
`torch.cuda.memory.CUDAPluggableAllocator` interface before PyTorch initializes
CUDA. It does not replace Python functions, patch tensor methods, or install
allocator hooks.

The private `libshadowspill_pytorch.so` connector owns one process-lifetime CUDA
backend and neutral runtime. Its three callbacks translate PyTorch's device,
address, and stream arguments into neutral allocation IDs and opaque stream
tokens. It contains no allocation policy: slab admission, blocking, retirement,
and failures remain in `libshadowspill_runtime.so`.

The connector is process-global because PyTorch cannot safely replace its CUDA
allocator after initialization. Public planning will calculate the physical
slab and host-arena admission before calling this internal installer. A later
plan may reuse the installed runtime but cannot silently resize its physical
reservation.

Bootstrap accepts the physical device budget plus explicit provider headroom,
not a requested slab size. Before allocating the slab, it measures the current
process's CUDA-context bytes through the backend NVML ledger. The slab is the
remaining budget rounded down to 2 MiB. Bootstrap then re-queries physical use
and fails before allocator installation if the process exceeds the cap. The
immutable bootstrap admission and current physical reading are available
through the private adapter ABI for `PlanReport` and seal checks.

The default provider headroom is 1,280 MiB. It is an explicit subdivision of
the physical cap, not extra memory. A conventional CUDA slab cannot be resized
after compiler or provider code retains even one allocator-owned pointer, so
this allowance must exist before CUDA initialization. Users with a measured
smaller or larger provider footprint may set `provider_headroom=` on
`shadowspill.memory.device()`. Final sealing still measures actual external
use and fails closed if the configured allowance is insufficient.

Sealing compares the provider high-water learned during profiling with the
headroom reserved at bootstrap; it cannot enlarge the reservation. A physical
check reads current per-process bytes and derives external provider use as
`process - context - slab`. Exceeding either the provider reservation or total
cap latches a plan violation independently of allocator callback failures.
Public callables check at planning seal and lifecycle boundaries; supported
fixed-shape task ABIs must have already exposed all direct provider growth
during warmup.

The C callback remains the source of the structured first-failure ledger. The
C++ PyTorch adapter checks its return value synchronously and throws before a
failed nonzero request can become a null-backed tensor. No-progress and
physical-cap failures become PyTorch `OutOfMemoryError`; task-allocation
envelope or ABI violations become `RuntimeError`. Both include the active
execution and semantic task identities plus the native pool geometry. The
executor catches failures only long enough to abort the active native task;
unrelated compiler, provider, and CUDA errors retain their original type and
traceback. Free and record-stream first resolve the exact live address and
generation in the neutral allocator table; a missing address is a
qualification failure rather than an ignored foreign allocation.

## Storage rebinding

The only PyTorch-specific C++ operation replaces a CUDA tensor storage's
`DataPtr` with a non-owning current slab address or a null CUDA placeholder. It
does not allocate, transfer, schedule work, or own runtime state. Before a swap,
the adapter validates the logical object ID, current slab address, and
generation against the neutral object table.

Release progress may finish between `after_task` and frontend
dematerialization. The runtime therefore retains one retired binding token;
the adapter accepts it only when replacing that exact stale non-owning pointer
with a null placeholder. This closes the timing race without delaying the
release, synchronizing a stream, or changing a planner directive.

All views of one storage observe the swap together. The existing `Parameter`,
TensorImpl, StorageImpl, sizes, strides, offsets, registrations, and ties remain
unchanged. The initial owning allocation is promoted to plan ownership before
its owning `DataPtr` is cleared, so the normal PyTorch free callback cannot
prematurely return it. Subsequent addresses are non-owning; annotated runtime
actions own their physical lifetime.

The private task-boundary bridge accepts borrowed CUDA stream addresses and
delegates exact input acquisition and annotated action submission to the
neutral runtime. It does not infer, reorder, or repair schedule directives.
`before_task` returns the current address and generation after inserting every
required FETCH event wait; the frontend rebinds storages before argument lookup.

The storage operation is compiled only when CMake finds the exact installed
PyTorch package. Development and wheel builds therefore pass
`torch.utils.cmake_prefix_path` as `CMAKE_PREFIX_PATH`. The release installer
will make that handshake mandatory; an allocator-only build advertises
`storage_rebinding = 0` and cannot construct a planned callable.

Task actions do not synchronize transfer streams. Qualification holds two FETCH
inputs behind a compute event, verifies that `before_task` inserts both waits,
and then observes those copies plus a simultaneous EVICT finishing while an
independent compute stream remains busy. Explicit `wait_idle`, checkpoint, and
close boundaries are synchronizing by contract.

Provider annotations are disabled by default and are independent of bounded
runtime tracing. Passing `profiler_annotations=True` to a planned call enables
them for that call and its asynchronous terminal actions.

Nsight ranges follow the documented namespace:

- `shadowspill.pytorch.task.<dense-id>` spans frontend task dispatch;
- `shadowspill.pytorch.storage_rebind` identifies each storage swap;
- `shadowspill.runtime.allocate` identifies slab-range admission;
- `shadowspill.runtime.transfer.{fetch,evict}.<alias>.<relationship>` identifies
  copy submission. Each label includes the alias bundle, object role, byte
  count, trigger execution, and either the next input consumer (FETCH) or the
  latest output/mutation/last-input source (EVICT). Labels are constructed once
  during execution admission; the worker performs no lookup or formatting;
- `shadowspill.runtime.wait_event` identifies stream dependency insertion.

The Phase-5 qualification trace is reproducibly checked with
`qualification/check_cuda_trace.py`. It requires one conventional slab
allocation, one pinned arena, real bidirectional copy overlap with compute, no
VMM entry point, and no device/context-wide synchronization.
