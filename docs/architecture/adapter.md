# PyTorch adapter

The adapter is the compiled component that speaks PyTorch on one side and the
neutral runtime on the other. It is the only place framework conventions and
the process-global allocator live, and since the backend contract it holds no
provider knowledge either.

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
