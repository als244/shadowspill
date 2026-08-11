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
