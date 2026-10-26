# Provider resources

Some custom operators share buffers or backend handles that cannot be saved as
model tensors. Builders accept an optional `resources` mapping for these objects.
It is generic: lookup tables, communication buffers and other provider-owned
resources use the same description. Ordinary models do not need this argument.

```python
from shadowspill import Resource, build_step

program = build_step(
    model, optimizer=optimizer, objective=objective,
    example_args=(features,), parallelism=parallelism,
    resources={
        "exchange_buffer": Resource(
            factory="my_backend.resources:initialize",
            options={"capacity": 4096},
            groups={"exchange": parallelism.group("exchange")},
        ),
    },
)
```

`groups` is optional. Its keys are provider argument names, and its values are
typed references from the central parallelism specification. `options` contains
finite JSON values. The factory is an importable function, not a closure or a
live object. Operators refer to stable resource names instead of process-local
handles. Parameter values remain model state, initialized separately by Executor.

The factory has this interface:

```python
def initialize(*, name, device, groups, options):
    # groups contains resolved ProcessGroups; options is ordinary Python data.
    return BackendResource(name, device, groups, options)  # exposes close()
```

The provider owns its internal implementation and must clean up partial factory
failures. Its operator completion on the calling stream must cover private-stream
work and buffer use. ShadowSpill does not schedule those private streams. Backend
setup must reproduce the saved task implementation without choosing new kernels.

Build, profiling and execution each recreate resources for their own lifetime.
Build records the factory source digest alongside its descriptions; profiling
and execution check it before calling the factory. Setup uses ordinary allocations
during build/profiling and precedes memory admission in Executor. Profiling counts
retained PyTorch-visible resource allocations as fixed device memory. Allocations
outside PyTorch require the same external-headroom allowance as other libraries.

Resource groups require caller-owned distributed rendezvous and a complete
`rank_map` when ranks are reassigned. Pass that mapping to the builder and
`measure_program`, as well as Executor. Model tasks that communicate must run
on all participants during finalization/profiling; ordinary DP compute still
needs only one representative GPU. Profiling coordinates iteration counts,
requires symmetric allocation contracts, and retains maximum measured costs.
Communication inside a model task is already included in that task's duration.

Plans contain descriptions and source identities, never live groups or handles.
Archive loading, planning and reporting do not import or initialize the provider.
Identically named resources are shared by admitted training/evaluation Plans;
conflicting descriptions or source identities are rejected. Resources close in
reverse initialization order before their process groups are destroyed.

The generic lifecycle is qualified with a non-distributed custom operator through
all three builders and fresh Executor workers. Each distributed backend still
needs its own target-hardware qualification; this interface alone does not certify
EP transport or a provider's transitive native dependencies.
