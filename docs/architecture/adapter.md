# PyTorch adapter

The adapter is the shared object that speaks PyTorch on one side and the
neutral runtime on the other. It is the only place framework conventions and
the process-global allocator live, and it holds no provider knowledge: the
backend it loads is reached through the same flat table the runtime uses.

## Why it exists

PyTorch's pluggable allocator calls three C functions -- malloc, free,
record_stream -- with no pointer of the caller's to carry state in, and its
storages are rebound through libtorch's C++ API. Both need a library that
links libtorch, and nothing else in ShadowSpill may: planning-only callers must
not carry libtorch, and the runtime must stay usable from any framework. So
the adapter is the one library that links both, and it holds exactly what
needs PyTorch -- the callbacks, the storage views, the stream wrapping and
profiler ranges at task boundaries, and loading the backend by name -- and no
policy: no planning, no pools, routes or lanes of its own, no provider code.
Anything reachable with the runtime handle it publishes is called on the
neutral library instead.

**It is also the only C++ in the tree, and only barely.** ShadowSpill is C;
libtorch's headers are C++, so a translation unit that includes them must be
too. Three files in this directory do -- the allocator's exception wrapper and
the two storage files -- and everything else here, including the bootstrap that
builds the runtime's pools, routes and lanes, is C like the rest.

That boundary is not free, because those three files include
`shadowspill/runtime.h` and so compile every header the umbrella pulls in as
C++. Anything in that set must therefore stay valid C++, which rules out
`_Atomic`. Where a header's contents are of no use to a caller -- a layout only
an implementation needs -- the answer is to keep it out of the umbrella rather
than to write it twice: [`lane_base.h`](lanes.md) is the case that forced the
question, and the rule it set.

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
  completion protocol, and open the runtime's profiler ranges and timing
  markers around it.

## How the source is laid out

`csrc/adapter/pytorch/` has the runtime's shape: one `internal.h` per
directory saying what it holds, one file per concern.

- `lifecycle/` — bootstrap from a config, loading the backend library by
  name, close, the process-exit hook, and the physical-memory ledger.
- `allocator/` — the three callbacks PyTorch's pluggable allocator makes, and
  the C++ wrapper that turns a failed one into a typed exception.
- `failure/` — what a failed call latches, and the report a person reads.
- `tasks/` — the task boundary on the dispatching thread: the range a task
  opens, allocation scopes, before, after and abort. What a scope owes the
  allocations it made, and what actually releases one, is [what a scope owes
  when it ends](task-boundaries.md#what-a-scope-owes-when-it-ends).
- `storage/` — PyTorch storages over runtime leases: the C primitives, and
  the torch operators over them, one file per dispatch key.
- `internal.h` and `adapter.c` at the top: the one process-global instance
  PyTorch's callback signature forces, and the calls that describe the
  process. Profiler ranges are not here — the runtime owns them, and every
  directory opens one on the runtime it is bound to.

## What it requires of a backend

Exactly the [backend contract](../c/backends.md): the flat table of
driver-level calls. The adapter loads the library named by
`Runtime(backend=...)` at bootstrap, checks the table with
`shadowspill_backend_is_valid()`, reads the provider's alignment and memory
accounting from it, and hands it to the runtime, which builds the pools,
routes, and lanes; see [backends](backends.md).

## What it exposes upward

The C entry points in `<shadowspill/pytorch_adapter.h>`, grouped in the
[adapter C API](../c/pytorch-adapter.md): bootstrap, physical admission and
close; the allocator callbacks; objects and storage; task boundaries and
allocation scopes; and failure and recovery. Profiling is absent on purpose:
the ranges are the runtime's. The Python layer wraps
them in two places, along the same line this header draws. What needs the
framework -- the allocator install, the storage bridge at a task boundary -- is
wrapped in `shadowspill.pytorch`; everything reachable through the runtime handle
the adapter publishes is called from `shadowspill.runtime`, which imports no
framework at all, rather than restated in the frontend. The rule is the same one
stated for the header: a call the neutral runtime already owns is made there.

Previous: [Step boundaries](step-boundaries.md). Next: [Timelines: how a step
is measured](timelines.md).
