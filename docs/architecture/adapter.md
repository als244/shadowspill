# PyTorch adapter

The adapter is the compiled component that speaks PyTorch on one side and the
neutral runtime on the other. It is the only place framework conventions and
the process-global allocator live, and since the backend contract it holds no
provider knowledge either.

## Why it exists

PyTorch's pluggable allocator calls three C functions -- malloc, free,
record_stream -- with no pointer of the caller's to carry state in, and its
storages are rebound through libtorch's C++ API. Both need compiled code that
knows PyTorch, and nothing else in ShadowSpill may: planning-only callers must
not carry libtorch, and the runtime must stay usable from any framework. So
the adapter is the one library that links both, and it holds exactly what
needs PyTorch -- the callbacks, the storage views, the stream wrapping and
profiler ranges at task boundaries, and loading the backend by name -- and no
policy: no planning, no pools, routes or lanes of its own, no provider code.
Anything reachable with the runtime handle it publishes is called on the
neutral library instead.

## What it is made of

```text
torch allocator hooks     objects and storage views     task boundaries
 (malloc / free /          (import, export, bindings,     (before_task,
  record_stream)            storage validation)            after_task, traces)
          \                        |                          /
           +------------  shadowspill_pytorch  ------------+
                                   |
                        neutral runtime handle
                                   |
                     backend table (loaded by name)
```

- **Allocator hooks** implement PyTorch's pluggable allocator over the
  runtime's device pool, so every tensor the framework creates on the
  accelerator is a runtime allocation with an identity the plan can reason
  about.
- **Objects and storage** import model state into pools, publish bindings for
  a plan, and validate that PyTorch storage views match the objects they
  claim.
- **Task boundaries** wrap each compiled task with the runtime's readiness and
  completion protocol, profiler ranges, and the trace's timing markers.

## How the source is laid out

`csrc/adapter/pytorch/` has the runtime's shape: one `internal.h` per
directory saying what it holds, one file per concern.

- `lifecycle/` — bootstrap from a config, close, the process-exit hook, and
  the physical-memory ledger.
- `allocator/` — the three callbacks PyTorch's pluggable allocator makes, and
  the C++ wrapper that turns a failed one into a typed exception.
- `failure/` — what a failed call latches, and the report a person reads.
- `tasks/` — the task boundary on the dispatching thread: the range a task
  opens, allocation scopes, before, after, abort, and the pre-task action
  batch.
- `storage/` — PyTorch storages over runtime leases: the C primitives, and
  the torch operators over them, one file per dispatch key.
- `internal.h`, `adapter.c` and `profiler.c` at the top: the one
  process-global instance PyTorch's callback signature forces, the calls
  that describe the process, and the profiler every directory opens ranges
  through.

## What it requires of a backend

Exactly the [backend contract](../c/backends.md): the flat table of
driver-level calls. The adapter loads the library named by
`Runtime(backend=...)` at bootstrap, checks the table with
`shadowspill_backend_is_valid()`, reads the provider's alignment and memory
accounting from it, and hands it to the runtime, which builds the pools,
routes, and lanes; see [backends](backends.md).

## What it exposes upward

The C entry points in `<shadowspill/pytorch_adapter.h>`, grouped in the
[adapter C API](../c/pytorch-adapter.md): bootstrap and physical admission,
the allocator hooks, object and storage operations, execution boundaries,
profiling and tracing, and failure reporting. The Python layer wraps them in
`shadowspill.pytorch.runtime_adapter`; anything reachable through a runtime
handle the adapter already published is called on the neutral library
directly rather than restated here.
