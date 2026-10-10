# Importing state

How a caller's model and optimizer state come to live in the runtime's pools,
what the caller must guarantee for that to stay bounded, how dtype is decided,
and how the values come back out.

## The problem

State that a plan will place has to live in a pool the runtime owns. The
obvious way to arrange that — build the state the ordinary way and copy it in
— means the state exists twice at once, so ordinary memory must hold a full
copy of something that was only ever meant to live in the pool. For state that
is large relative to the memory available, that transient is what decides
whether the model can be loaded at all, and it is paid on every run.

ShadowSpill allocates state in its destination pool before running its
initializer. Constructing on `meta` therefore avoids a complete temporary host
model. Addressable pools expose their bytes directly. Non-addressable pools
(such as SSD and remote memory) stage only the storage roots used by the current
initialization operation, write mutations back, and keep metadata-only tensors
between operations. The model's initialization code is unchanged.

## Three paths in

| Path | Values come from | Host payload outside the pool |
|---|---|---|
| **Meta model, then initialize** | `reset_parameters()` or an explicit `initialize(model)` callback | Initializer scratch; non-addressable pools also stage roots touched by one operation |
| **Import an existing model** | Already initialized registered tensors | The caller's source until its references are dropped |
| **Import a checkpoint** | A memory-mapped file | Reclaimable file pages, plus per-operation staging for non-addressable pools |

A meta model is materialized in place and returned as the same module. An
already initialized model is copied into a distinct module hierarchy, preserving
ties, views and values; assign the returned model back to the input variable if
the source is no longer needed. Checkpoint import fills the supplied model in
place. No full CPU payload is retained for a non-addressable pool: attempting to
read such a tensor directly raises an error explaining that its bytes belong
to the pool.

`import_model_state(model, runtime=runtime, pool="spill", initialize=fn)` accepts
an optional in-place initializer. With a meta model and no callback, each module
that owns state must implement `reset_parameters()`. Trainer and Forward retain
their explicit `prepare(..., initialize=fn)` contract and execute that callback
after the ShadowSpill backend has allocated pool state. Quickstart uses the same
import path. Optimizer moments and optional master parameters are also declared
without payloads and initialized after pool allocation.

The entry points and return values are in the
[frontend API](../python/api/frontend.md#persistent-state).

## The contract

### Tensor representations

A registered tensor can use PyTorch's `__tensor_flatten__` /
`__tensor_unflatten__` protocol to represent one logical value with several
ordinary tensors, such as quantized bytes, scales and a transposed payload.
ShadowSpill imports and accounts for those physical components, preserving
shared storage, views and tied parameter identities. It rebuilds the logical
wrapper for capture and state reads; the wrapper's nominal dtype does not
determine its storage size. Nested representations use the same traversal.

Setup-time operations also walk those components. Reading or converting a
wrapper backed by an SSD/remote pool stages its physical roots; mutating it
writes those roots back before staging is reused. Returned views refer to pool
metadata, not temporary staging storage. Authentic control-value derivation
during planning uses the same bounded access, so it does not require copying
the whole model into host memory.

All physical state must be named by that protocol. Its metadata must be
serializable, data-free configuration. Allocations hidden in a library or a
communication handle remain external memory. The layer's logical backward
defines the parameter gradient; integer payloads and scale components do not
become separate trainable parameters just because they hold the bytes.

For dense optimizer masters, the representation implements ordinary PyTorch
`dense.copy_(wrapped)` for initialization and `wrapped.copy_(dense)` for
publication and restore. Publication must be traceable, including every
component it changes. A custom operation can expose an external kernel's tensor
inputs and mutations. The tensor library supplies the conversion math;
ShadowSpill neither chooses quantization nor recognizes particular libraries.

This state protocol does not imply support for arbitrary wrapper-valued public
inputs or task outputs. The supported boundary is registered state whose
captured operations produce ordinary tensors and logical dense gradients.

Profiling also records allocations retained by a library during task warmup.
Their measured bytes reserve execution-pool capacity once; they are separate
from the workspace allocated on each invocation. Allocations outside the
framework allocator remain covered by external-memory headroom.

Reusable library resources should be initialized outside individual task
invocations. An allocation made inside a plan's task scope belongs to that plan
and is reclaimed when it closes; a process-global library cache must not retain
it for a later plan. Setup allocations made through the framework allocator are
included in the runtime's fixed reserve before planning.

### Construction and initialization

Constructing into the pool asks three things of a model. They are not
ShadowSpill inventions: they are the standard recipe for a model too large to
build on the host, and a model that satisfies them works with other systems
that build on `meta` for the same reason.

1. **Constructible on `meta`.** `__init__` allocates no data and reads no
   values — nothing that inspects a tensor's contents. On `meta` the model is
   structure alone and costs nothing.
2. **Dtype fixed at construction.** Parameters and buffers are created in the
   dtype they will be used in. A cast afterwards allocates, and a cast is
   exactly what defeats any scheme that placed the state carefully.
3. **Initialization writes in place.** Each state-owning module implements
   `reset_parameters()`, or the caller supplies an initializer for the whole
   model. It writes existing tensors instead of replacing them or their storage.

The rule behind all three: **`__init__` declares shape and dtype;
`reset_parameters` produces values.** A module that fuses allocation and
initialisation cannot have its storage placed anywhere but where it happened
to be allocated.

Stock library modules already satisfy this, so most models comply for the
parts they did not write.

### Two kinds of owned state, one rule

A module should not have to know which kind it holds, and nothing here is
special-cased by kind.

- **Learned parameters** are drawn from a distribution.
- **Derived constants** are computed deterministically from configuration:
  rotary tables, positional encodings, causal masks, precomputed index
  tensors. They are usually registered non-persistent, because a checkpoint
  can recompute them rather than store them.

Both are initialised in place, by the same hook. Derived constants are the
common accidental violation, because the natural way to write one computes
the value and registers the result in a single line — which allocates its own
storage and pins it where it was computed. Written to the contract, the
buffer is registered empty with a shape and dtype, and the value is computed
and copied in when the hook runs.

```python
class RotaryTables(nn.Module):
    def __init__(self, width: int, *, capacity: int) -> None:
        super().__init__()
        self.width, self.capacity = width, capacity
        table = torch.empty(capacity, width, dtype=torch.float32)
        self.register_buffer("cosine", table, persistent=False)
        self.register_buffer("sine", table.clone(), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        angles = self._angles()          # scratch: one table, not one model
        with torch.no_grad():
            self.cosine.copy_(angles.cos())
            self.sine.copy_(angles.sin())
```

### Temporary memory bounds

Host-addressable pools require no second parameter allocation. An initializer
may still allocate its own scratch. For a non-addressable pool, extra CPU
payload is bounded by the storage roots touched by one tensor operation plus
that operation's scratch. Staging buffers are reused across operations and
released at the end of setup. This is a largest-operation bound, not a promise
of constant bytes independent of tensor dimensions.

Initialize tensors individually. Passing every parameter to one bulk operation,
or retaining clones of every parameter, defeats that bound. A partial view may
stage its entire underlying storage to preserve the untouched bytes. Library
initializers must support the device on which their setup operations run; the
pool mechanism does not implement library-specific quantization.

Explicit `read_model_state()` and `export_model_state()` intentionally return
ordinary host values and can require a full model copy. Use `release_model_state()`
to discard pool state without reading it out. `PlannedTrainStep.save()` writes
pool objects incrementally into a standard `torch.load`-compatible checkpoint,
flushing mapped ranges as it goes. Restoring model and optimizer values also
processes one parameter's storage at a time. Distributed checkpoint restoration
may additionally need communication buffers for rebuilding replicated weights.

## What is refused

The contract is enforced rather than assumed, because every way of breaking
it produces a plausible-looking wrong answer rather than an error.

- A meta model without an explicit initializer whose modules own state without
  a `reset_parameters` is refused,
  naming them. Materialising it would leave whatever the pool memory
  contained, which reads as values.
- A model that mixes `meta` and materialised tensors is refused: which of its
  values are real is then unclear, and guessing is worse than stopping.

## Dtype

**The caller decides what the state is, including every dtype. ShadowSpill
decides only where it lives.** Nothing on the state path names a dtype: what a
tensor declares is what gets allocated, per tensor.

Four decisions are easy to run together and are better kept apart:

| decision | whose | where it is written |
|---|---|---|
| a tensor's dtype | the model's | the module that declares the tensor |
| the precision a value is *drawn* in | the module's | inside its `reset_parameters` |
| the optimizer state's dtype | the optimizer's | wherever the optimizer creates it |
| whether a higher-precision master exists | the optimizer's | as ordinary optimizer state |

### One dtype per tensor, not per model

A model is not required to be uniform. Each tensor is materialised in the
dtype it declares, so a module that needs more precision than its neighbours
simply says so:

```python
class PreciseNorm(nn.Module):
    """Normalisation kept in fp32 inside an otherwise bf16 model."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)
```

Nothing else in the model changes, and nothing in the import path inspects
this. Derived constants work the same way -- rotary tables are commonly fp32
inside a low-precision model, and they are declared fp32 where they are
registered.

A caller that wants a *default* for tensors which do not say applies one while
constructing, which is the caller's own setup step and not something
ShadowSpill supplies:

```python
def build(config: object) -> nn.Module:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            return MyModel(config)
    finally:
        torch.set_default_dtype(previous)
```

Tensors that declare a dtype keep it; the rest take the default. The model
handed over is then fully described by itself.

### Drawing more precisely than you store

The dtype a value is *drawn* in is a separate question from the dtype it is
*kept* in, and it is the module's business. A module that wants a
high-precision draw uses scratch the size of that one tensor:

```python
class WidelyDrawn(nn.Module):
    def reset_parameters(self) -> None:
        scratch = torch.empty_like(self.weight, dtype=torch.float32)
        nn.init.normal_(scratch, std=self.width**-0.5)
        with torch.no_grad():
            self.weight.copy_(scratch)
```

This costs one tensor of host memory at a time, never two copies of the model,
and needs nothing from ShadowSpill.

### Optimizer state at its own dtype

An optimizer's state dtype is the optimizer's decision, and it is routinely
not the parameter's -- fp32 moments over low-precision parameters are the
usual mixed-precision arrangement. Nothing needs to be declared for that to
work: whatever dtype the optimizer creates is the dtype that is allocated in
the pool, because the state's shape and dtype are discovered from the
optimizer itself rather than assumed from the parameter.

A frozen parameter is still model state and still lives in the pool, because
it is read every step; that it carries no *optimizer* state is
[the optimizer's page](optimizer.md#frozen-parameters-carry-no-state).

### A higher-precision master copy

Whether a master copy exists at all is the optimizer's decision, and what
exists differs by case. This is worth being explicit about, because the answer
is not the same:

| master dtype | what exists |
|---|---|
| same as the parameter | **one object.** The optimizer mutates the parameter in place. There is no separate master, so no duplicate pool capacity and no second fetch. |
| different from the parameter | **two objects**, each with its own dtype and residency. The optimizer reads and updates the master and writes the parameter the forward pass consumes. |

The first case is the default and costs nothing: a same-dtype master *is* the
parameter, so nothing is duplicated and no aliasing machinery is involved --
it is identity, not aliasing.

The second needs no new concept either. The master is ordinary optimizer
state: pool-resident, checkpointed, updated by the task the optimizer capture
already produces. Because it is a distinct object, the planner schedules it
independently -- its first use is the optimizer's task, so it need not occupy
device memory during forward and backward at all, and a task that requires the
higher precision simply reads the object that has it.

Two dtypes are therefore always two objects. An alias holds the same bytes on
both sides of a transfer -- which is what the residency model, the planner's
transfer accounting and the byte count the runtime hands a lane to copy all
assume -- so a
conversion on the way to the device is not something a single alias can
express.

## And back out, the same way

Training checkpoint files encode tensor wrappers as their ordinary component
tensors, plus a description of the representation. They remain readable with
`torch.load(..., weights_only=True)`. Loading through ShadowSpill reconstructs
the wrapper using the receiving model's class and metadata, after checking the
description and component geometry. No quantizer object is unpickled from the
checkpoint. Compute checkpoints preserve both quantized orientations and scales
exactly; master checkpoints still save only the dense master where one exists.
In-memory `state_dict()` results retain the model's logical tensor types.

State leaves a pool by copying, exactly as it enters. `read_model_state()` and
its optimizer counterpart answer with ordinary host memory -- one buffer per
storage root, with the target's views laid over it, so entries that shared a
root still share one. `export_model_state()` does the same and hands ownership
back, rebinding the target's own tensors.

Neither hands out a pool address, and there is deliberately no second path that
reads the pool in place. A pool whose memory is not in this address space has
no address to view, so an in-place read is available only sometimes -- and a
cheaper path that silently stops applying is the one a caller comes to rely on.
The same trade as the import side above, decided the same way.

How the bytes actually cross the edge is the kind's business: a pool this
process can address leaves the [pool-memory contract](../c/pool-memory.md)'s
`read` entry out and the runtime copies directly, and one whose memory is
elsewhere implements it. The caller makes the same call either way.

## See also

- [The ShadowSpillProgram](program.md) for persistence: how long a piece of
  state must survive, which is a different question from where it lives.
- [Plan search](search.md) for how residency decides when state is on the
  device.

Previous: [The step artifacts](step.md). Next: [The optimizer](optimizer.md).
