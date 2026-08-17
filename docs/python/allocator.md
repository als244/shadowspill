# PyTorch allocator integration

`Runtime` installs the ShadowSpill allocator through the compiled PyTorch
adapter. Allocator selection is process-global and cannot be reversed after
PyTorch initializes the accelerator, so construct exactly one runtime before
any accelerator tensor allocation.

## Callback contract

PyTorch allocation callbacks enter the adapter and then the neutral C runtime:

- allocate a strict fixed-layout slot or bounded dynamic range;
- free logically and retire behind the last recorded stream;
- record additional stream use;
- look up the allocation owning a pointer;
- promote task outputs into logical object generations.

A nonzero allocation failure raises `RuntimeExecutionError` from the adapter.
No nonzero request returns a null pointer to compiled code. Structured
diagnostics distinguish no-progress OOM, task-envelope violation,
allocation-contract mismatch, worker failure, and backend failure.

Zero-byte requests are tracked separately in diagnostics. They do not acquire
a physical lease and are not counted as ordinary allocations requiring a
matching free.

## Storage rebinding

Logical objects keep identity while their current execution address changes.
Before a task, the executor acquires current object generations, inserts
stream waits for unfinished fetches, and batch-rebinds PyTorch storages. After
the compiled call, returned allocations are classified as outputs, mutations,
or anonymous temporaries and published through the runtime.

Views preserve shape, stride, storage offset, and alias relationships. The
runtime never asks PyTorch to copy an object solely because its residency
generation changed.

## Streams and readiness

Host code does not synchronize on normal transfer readiness. A fetch owns a
completion event, and the compute stream waits on that event. Allocator
capacity waits are permitted only when a known pending retirement or transfer
can satisfy the request; otherwise the runtime reports no progress.

## Provider annotations and tracing

`profiler_annotations=True` enables backend profiler ranges such as task,
compiled-call, fetch, evict, and allocation labels. It is independent of
`runtime_trace=True`, which records the structured data returned through
`StepDiagnostics`. Both are off by default.

The C worker is a provider-independent native thread and does not execute
Python or acquire the GIL.
