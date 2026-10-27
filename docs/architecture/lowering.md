# PyTorch capture and lowering

ShadowSpill separates semantic identity from compiled allocation geometry.
This prevents incidental FakeTensor storage or allocator callback identity
from merging or splitting logical objects. The output is the canonical
[framework-neutral program](program.md), not a PyTorch execution trace.

For training, stage partitioning is followed by [graph-pair
construction](graph-pair-construction.md), which creates the forward/backward
alternatives whose task contracts, compiled layouts and measurements this
lowering consumes.

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
canonical ShadowSpillProgram
```

## Semantic contract

Each compiled task has a deterministic `TaskStorageContract`:

- input roots name the canonical argument position of an input alias group;
- fresh roots name one producer node and result index;
- output views retain shape, stride, dtype, layout, and offset within their
  root;
- a view's span is the reach of its locations from where it starts, which is
  what its root must hold and what moving it costs; it is not how many
  elements the view has. A broadcast reads one location many times and so has
  more elements than bytes, and a view that steps over its storage has more
  bytes than elements. An object is worth its span;
- a mutation names the input position a task updates, and the output leaf that
  replaces it where the compiled form returns one;
- duplicate leaves reference the same root directly.

The extractor uses FX provenance, dispatcher schemas, graph signatures, and a
fresh fake/meta interpretation for geometry. FakeTensor storage identity is
never an object-identity authority. Ambiguous alias or mutation schemas fail
closed with the task, node, operator, and result involved.

## Executable storage contract

Before invoking Inductor, direct task compilation functionalizes intermediate
mutations through PyTorch's compiler dispatcher. Inductor's clone elimination
and buffer-reuse passes require this functional form. Input updates remain part
of the task contract and are published by terminal copies. Inputs that share
storage use a shared base during normalization, preserving writes through
overlapping views. Opaque mutable operators use PyTorch's automatic
functionalization. This normalization runs during compilation, not execution.

The profiling environment identifies this functional task compiler separately
from the earlier direct-mutation compiler. Their captured graphs can match
while Inductor's allocation sizes and storage reuse differ, so cached manifests
and measurements must be rebuilt under the new identity. Artifact schemas stay
at version 1.

Inductor has already converted functional ATen semantics into executable
storage behavior. ShadowSpill's compiler adapter captures a normalized task
manifest and the callable at that boundary. Private PyTorch
knowledge is confined to the frontend compilation package; the rest of the
system consumes stable ShadowSpill contracts.

The executable contract reconciles semantic roots with actual output
allocations and views. Physical sizes and offsets validate that views fit and
that distinct live roots are not accidentally merged. They do not redefine
semantic object identity.

### Storage the plan does not manage

A contract is written from a trace, and a trace is taken over values that only
describe the real ones. A describing value can be wrong about where a result
lives: an operator whose result is a scalar allocated off the execution device
is traced as device memory of that scalar's size. No device allocation is ever
made for it, so none can be found for it either.

Where a result lives is therefore observed rather than traced. Profiling runs
the compiled task and records which of its output leaves came back off the
execution device, and reconciliation narrows the contract to match before it
looks for anything: those leaves give up their span, and a root all of whose
leaves gave up theirs gives up its own. One root is one allocation and so is
in one place, and a root named for some of its leaves and not others is
refused rather than reconciled.

What is left is how the rest of the system already says *this needs no
storage*: zero bytes, no allocation, no residency, no transfer. The object
still exists, is still produced by one task and read by another, and is still
passed between them; only the plan's memory has nothing to do with it.

## Allocation behavior

Build derives compiler-visible storage structure. `measure_program` observes
allocation requests and releases while running each finalized task on the
ordinary device allocator. This supplies missing physical memory evidence;
execution later validates it against the admitted layout.

Automatic input preparation preserves tensor geometry and shared storage. It
uses small deterministic floating-point samples, zero gradient/optimizer inputs,
and known statistic values where supported. It does not load checkpoint weights.
Callers can supply real task inputs when their semantics require them. Control
values must be embedded in the Program, supplied by the caller, or obtained by
replaying their recorded producer on the captured device. Backward saved tensors
always come from the matching forward.

Three warmed observations must have identical allocation contracts and
output/input aliases. Retained anonymous allocations fail measurement rather
than silently becoming unbudgeted state. First-use retained memory is recorded
separately. The contract identifies required allocations, requested/charged bytes,
alignment, ownership and lifetimes. Runtime rejects allocations that disagree
with that contract or exceed the admitted task envelope.

The lower-level contract also supports explicitly supplied optional paths and
bounded dynamic scratch. The public measurement phase does not infer such paths
from inconsistent observations; it requires a stable warmed path.

## Cross-task lowering

`ObjectCatalog` owns canonical model state, inputs, saved values, gradients,
optimizer state, and public outputs. `TaskBindingResolver` applies the same
root/view/mutation rules to forward, backward, recomputation, inference, and
optimizer tasks. Mode-specific helpers only classify public results and state
roles.

A training step is lowered once per [microbatch
ordering](planning-pipeline.md#the-walk-through-the-microbatches), and the
orderings share everything that comes before the walk matters.
`TrainingLowering` registers the step's objects and binds its stage boundaries
once, on the first program it is asked for, and gives each ordering's program
its own copy of the object and profile catalogs to continue from. Identifiers
are numbered in the order they are first used, so a program built from the copy
is the program its ordering lowers to alone. What each ordering does for itself
is what the walk decides: whether each microbatch's stage runs its creating or
its accumulating backward form, the order of the tasks, and the residency at
the step's boundaries.

No output copy is added merely to preserve mutation identity. When compiled
code returns a replacement allocation for a mutated object, the replacement
becomes the object's next generation at the task boundary that publishes it;
see [task boundaries](task-boundaries.md) for what that boundary does.

## Generality boundary

Lowering performs no model-family or operator-specific policy. Custom
operations work when their fake/meta behavior and alias/mutation schemas are
correct. Opaque external workspace may still require measurement because it
is not fully represented in FX or Inductor's visible buffer graph.

Previous: [Intermediate representation](ir.md). Next: [Graph-pair
construction](graph-pair-construction.md).
