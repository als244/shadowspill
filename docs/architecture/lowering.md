# Capture, compilation and lowering

The public builders produce an untimed `Program`. Capture identifies logical
values; compilation establishes executable storage geometry; lowering binds
those facts to cross-task objects and dependencies. Profiling and planning are
separate phases.

```text
Export / AOT graphs + save/recompute alternatives
    → semantic task contracts
    → compiled code + physical storage and visible workspace
    → untimed Program with reusable stage templates

Program + measured or explicit profiles → ProfiledProgram
ProfiledProgram + budgets + communication → Plan
```

## Semantic contracts

`TaskStorageContract` records input/fresh storage roots, output views and
mutations. Views retain shape, stride, dtype and their offset within a root.
Their byte span describes reachable storage, which can differ from tensor
`numel`: broadcast views occupy less storage and strided views can span more.
Duplicate leaves share a root.

Capture derives these identities from FX provenance, operator schemas and graph
signatures. Incidental FakeTensor storage addresses do not decide whether
logical outputs alias. Ambiguous alias or mutation schemas fail with the task,
node and operator involved.

## Compilation

`build/compilation/` packages each explicit task through PyTorch's standalone
AOT compiler. Forward/backward graphs already describe their saved values, so
compilation detaches examples and does not introduce another autograd save
boundary. PyTorch functionalizes mutations before Inductor optimization;
regressions cover overlapping input views, mutated temporaries and opaque
mutable operators. Shape padding is disabled to preserve declared geometry.

The version-specific adapter in `build/compilation/inductor/` reads the actual
GraphLowering result. Its executable contract records optimized output aliases,
allocation extents and offsets. Build also records compiler-visible workspace
and unresolved external allocation requirements. These physical facts validate
logical identities; they do not replace them.

Code, contracts and memory facts share one AOT cache entry. A foreign entry
missing build facts is recompiled in a temporary compiler cache. There is no
separate manifest sidecar cache. Finalization warms complete tasks with valid
inputs, resolves provider choices and packages the selected kernels. Isolated
stage tracing uses this same task compiler.

## Cross-task binding

`build/lowering/` owns this work:

- `objects.py`, `registration.py` and `task_binding.py` assign canonical tensor
  identities and bind task roots/views/mutations.
- `forward/` emits forward tasks and their initial/final residency.
- `training/` registers gradients and optimizer storage, binds stage boundaries
  and all backward forms, and saves reusable task templates.

The build emits one canonical topological inventory. It does not select a
microbatch traversal or optimizer ordering. CPU planning binds saved templates
to an ordering and selects save/recompute alternatives without recapture,
compilation or measurement. Creating and accumulating backward forms both
remain available.

Compiled replacement outputs become the mutated object's next generation at its
publication boundary. No copy is added just to preserve its identity. Cotangent
handoffs explicitly connect distinct storage roots; direct producer edges also
cover values that skip stages. See [task boundaries](task-boundaries.md).

## Memory evidence

Build uses compiler geometry and observed output placement to exclude host-only
leaves from device storage. Such values still have identities and dependencies;
they simply require no GPU residency or transfer.

`measure_program` runs finalized tasks with the ordinary device allocator and
supplies missing memory evidence and durations. Its inputs preserve aliases and
geometry. Supported floating-point inputs are prepared automatically; unknown
controls require a caller override or their recorded producer. Backward uses
its matching forward's actual saved tensors.

Three warmed observations must agree on allocation contracts and output/input
aliases. First-use retained memory is recorded separately; unexplained retained
allocations fail measurement. Execution validates requests, sizes, alignment and
lifetimes against the admitted contract. Compiler facts alone do not establish
opaque backend scratch requirements, and estimated memory permits simulation
only.

Lowering contains no model-family policy. Custom operations need correct
fake/meta behavior and alias/mutation schemas. See [graph-pair
construction](graph-pair-construction.md), [programs](program.md) and
[planning](planning-pipeline.md).
