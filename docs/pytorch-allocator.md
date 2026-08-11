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

Every callback is non-throwing. Allocation failure returns null through the
ordinary PyTorch allocation path and latches the first structured runtime
failure for diagnostics. PyTorch 2.13 may return an unmaterialized nonzero
tensor with a null `data_ptr()` from `torch.empty` after that callback instead
of raising immediately. Public planning therefore checks the latched failure
at task/API boundaries rather than assuming the construction call raised. Free
and record-stream first resolve the exact live address and generation in the
neutral allocator table; a missing address is a qualification failure rather
than an ignored foreign allocation.

## Storage rebinding

The only PyTorch-specific C++ operation replaces a CUDA tensor storage's
`DataPtr` with a non-owning current slab address or a null CUDA placeholder. It
does not allocate, transfer, schedule work, or own runtime state. Before a swap,
the adapter validates the logical object ID, current slab address, and
generation against the neutral object table.

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
required H2D event wait; the frontend rebinds storages before argument lookup.

The storage operation is compiled only when CMake finds the exact installed
PyTorch package. Development and wheel builds therefore pass
`torch.utils.cmake_prefix_path` as `CMAKE_PREFIX_PATH`. The release installer
will make that handshake mandatory; an allocator-only build advertises
`storage_rebinding = 0` and cannot construct a planned callable.

Task actions do not synchronize transfer streams. Qualification holds two H2D
inputs behind a compute event, verifies that `before_task` inserts both waits,
and then observes those copies plus a simultaneous D2H finishing while an
independent compute stream remains busy. Explicit `wait_idle`, checkpoint, and
close boundaries are synchronizing by contract.
