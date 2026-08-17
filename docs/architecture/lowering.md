# PyTorch capture and lowering

ShadowSpill separates semantic identity from compiled allocation geometry.
This prevents incidental FakeTensor storage or allocator callback identity
from merging or splitting logical objects. The output is the canonical
[framework-neutral Program](ir.md), not a PyTorch execution trace.

For training, stage partitioning is followed by the dedicated
[graph-pair construction](graph-pair-construction.md) phase. That phase creates
the forward/backward alternatives whose individual task contracts, compiled
layouts, and measurements are consumed by the lowering described here.

```text
Export/AOT FX semantics
        |
        v
TaskStorageContract
        |
        +------ semantic roots, views, aliases, mutations
        |
        v
compiled physical profile
        |
        +------ extents, allocation paths, workspace, timing
        |
        v
ObjectCatalog + TaskBindingResolver
        |
        v
canonical Program
```

## Semantic contract

Each compiled task has a deterministic `TaskStorageContract`:

- input roots alias declared task-input groups;
- fresh roots are identified by producer and result index;
- output views retain shape, stride, dtype, layout, and relative offset;
- mutations bind an output root to an explicit state input;
- duplicate leaves reference the same root directly.

The extractor uses FX provenance, dispatcher schemas, graph signatures, and a
fresh fake/meta interpretation for geometry. FakeTensor storage identity is
never an object-identity authority. Ambiguous alias or mutation schemas fail
closed with the task, node, operator, and result involved.

## Executable storage contract

Inductor has already converted functional ATen semantics into executable
storage behavior. ShadowSpill's compiler adapter captures a normalized task
manifest and the callable at that boundary. Private PyTorch
knowledge is confined to the frontend compilation package; the rest of the
system consumes stable ShadowSpill contracts.

The executable contract reconciles semantic roots with actual output
allocations and views. Physical sizes and offsets validate that views fit and
that distinct live roots are not accidentally merged. They do not redefine
semantic object identity.

## Allocation behavior

Task allocation profiling is a physical contract, not a semantic lowering
fallback. Each structural task is warmed and probed with representative
inputs. Floating anonymous values are deterministic standard-normal samples;
registered state and caller inputs retain authentic values; integer and
boolean control values come from callers or their producing tasks when
available.

The resulting `TaskAllocationABI` contains a strict core allocation path and a
bounded optional path:

- required outputs and mutations must match their expected ordinal, size,
  alignment, and ownership;
- optional anonymous/provider operations may be inserted or omitted within a
  measured dynamic-scratch allowance;
- ambiguous paths fail during profiling or immediately at runtime;
- a request exceeding its task envelope raises a task-attributed runtime
  error before an invalid pointer reaches the backend.

`allocation_probe_seeds` controls independent randomized activation probes.
`allocation_probe_repetitions` repeats each seed to expose first-use paths.
The defaults are one seed and two repetitions.

## Cross-task lowering

`ObjectCatalog` owns canonical model state, inputs, saved values, gradients,
optimizer state, and public outputs. `TaskBindingResolver` applies the same
root/view/mutation rules to forward, backward, recomputation, inference, and
optimizer tasks. Mode-specific helpers only classify public results and state
roles.

No output copy is added merely to preserve mutation identity. When compiled
code returns a replacement allocation for a mutated object, `after_task()`
publishes the new lease generation, retires the prior lease behind the task
completion fence, and rebinds registered PyTorch storage.

## Generality boundary

Lowering performs no model-family or operator-specific policy. Custom
operations work when their fake/meta behavior and alias/mutation schemas are
correct. Opaque external workspace may still require measurement because it
is not fully represented in FX or Inductor's visible buffer graph.

Previous: [Intermediate representation](ir.md). Next:
[Graph-pair construction](graph-pair-construction.md).
