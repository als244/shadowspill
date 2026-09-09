# Importing state

How a caller's model and optimizer state come to live in the runtime's pools,
what the caller must guarantee for that to cost nothing, and how dtype is
decided.

## The problem

State that a plan will place has to live in a pool the runtime owns. The
obvious way to arrange that — build the state the ordinary way and copy it in
— means the state exists twice at once, so the host must hold a full copy of
something that was only ever meant to live in the pool. For state that is
large relative to host memory, that transient is what decides whether the
model can be loaded at all, and it is paid on every run.

The way out is not to copy less. It is to never allocate the state anywhere
else: give it pool storage before it has values, and let it write its values
there.

## Three paths in

All three end with the same thing — a `PersistentState` the runtime owns —
and differ only in where the values come from.

| path | values come from | host cost |
|---|---|---|
| **construct into the pool** | the model initialising itself | none proportional to the state |
| **import a live model** | a model already built | a full copy while the import runs |
| **import from a checkpoint** | a file, mapped rather than read | reclaimable page cache |

The first is the one to use when the caller owns the model's definition. The
second exists for state a caller already has and did not build for this
purpose. The third orders itself deliberately: it makes the target
pool-backed *first* and then writes the file's values through, so the values
land in the pool rather than being copied into it, and the mapped file stays
reclaimable rather than becoming anonymous memory.

A model built on `meta` is rebound in place and handed back as the same
object, because it held no values to copy. A model that was already
materialised is copied into a new module whose tensors point at the pool, so
the caller keeps the return value rather than the model it passed.

## The contract

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
3. **`reset_parameters()` initialises in place.** Every module that owns state
   directly implements it, writing through the storage it already has rather
   than assigning a new tensor.

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

### What "no host transient" promises

No transient **proportional to the state's size**. A module may use bounded
scratch to compute a value it then copies in, as above; what is excluded is
any allocation that scales with parameter count, because that is the term
that decides whether the state fits at all.

## What is refused

The contract is enforced rather than assumed, because every way of breaking
it produces a plausible-looking wrong answer rather than an error.

- A model whose modules own state without a `reset_parameters` is refused,
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

The alternative -- one object, converted on transfer -- is deliberately not
taken. It would give a single alias different sizes on each side of the link,
contradicting the residency model's assumption that device and spill hold the
same bytes, the planner's transfer accounting, and the runtime's lane
arithmetic. It would also land the high-precision copy on the device whenever
the low-precision one was wanted, which is the opposite of what a separate
object achieves.

## See also

- [Persistence](program.md) for how long a piece of state must survive, which is a
  different question from where it lives.
- [PressureFit](pressurefit.md) for how residency decides when state is on the
  device.
