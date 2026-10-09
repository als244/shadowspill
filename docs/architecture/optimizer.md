# The optimizer

What ShadowSpill needs from an optimizer, what it promises in return, and why
the boundary falls where it does.

Quantized parameter wrappers expose their payloads and scales through the
[tensor representation contract](state-import.md#tensor-representations).
Their optimizer gradient remains one dense logical tensor. If a backward
returns several parameter gradients as views of one larger allocation,
lowering materializes independent gradient tensors inside the measured backward
task. This preserves their separate optimizer lifetimes; the copies count
toward the reported workspace and runtime. A full dense gradient that already
owns its allocation needs no such copy.

## One principle

**Declaration is separated from value, and each belongs to the side that knows
it.**

Declaration is structure: which tensors exist, in what shape, in what dtype.
The side that owns the definition knows it, and it can be established without
allocating anything.

Value is meaning: what a tensor starts at, and what a hyperparameter is on this
step. The side that owns the values knows that, and no one else can supply it
without guessing.

Fusing the two is what makes state impossible to place and updates impossible
to reuse. A line that allocates and fills together has decided where the bytes
live before anyone could say where they should live; a value baked into a
captured graph has decided what the update is before anyone could say what it
should be this step.

ShadowSpill sits between the halves. It gives declared structure storage, and
it runs declared updates. It declares nothing and it decides no values, which
is what lets it stay agnostic of model, of optimizer, and of dtype.

## State

An optimizer's state is often several times the model, so where it is
allocated matters more than almost anything else about it.

### Declared on meta

The optimizer declares its own state by being run on meta parameters: it names
every entry, fixes every shape, and chooses every dtype, without allocating a
byte. Entries that are not parameter-shaped come through as themselves, and an
optimizer that keeps its moments in a different dtype from the parameter is
read as it is rather than assumed to match. The same run says where each entry
starts ([below](#started-where-the-optimizer-starts-it)).

This costs nothing and asks nothing of the caller. ShadowSpill does it with
the optimizer the caller already passes.

### Created in the pool, then filled

Every declared entry is allocated in the spill pool before its start value is
written. No full CPU moment or master tensor bank is allocated beside the pool.
For addressable pools, initialization writes directly into the pool. For
non-addressable pools it stages the roots touched by one operation and publishes
the result through the pool's write API. The same mechanism initializes models;
see [state import](state-import.md#three-paths-in).

A few bytes of CPU control state can remain for scalar counters used by Python
during discovery. Tensor shapes, strides, dtypes and aliases remain available
without retaining large payloads. Ordinary Python scalar state requires no pool
allocation.

### Started where the optimizer starts it

ShadowSpill does not choose a starting value. The optimizer does, on its first
step, and the meta run watches it: the operation that makes an entry, before
the update first writes to it, says where the entry starts.

- An entry made as a constant -- zeros for moments and step counters, any
  other fill -- starts at that constant.
- An entry made as a copy of its parameter, cast or not -- a higher-precision
  master copy of a bf16 weight, say -- starts at the parameter, at the entry's
  own dtype.
- An entry the optimizer already holds, as after loading a checkpoint into
  it, starts at what it holds.

So nothing is assumed about any optimizer, and nothing is asked of the caller:
the value an entry starts at is the one the optimizer's own first step would
give it. An entry made from anything else has no value before the first step
-- SGD's momentum buffer begins as the first gradient -- and planning refuses
it, naming the entry and what it was made from, rather than choosing a value
the optimizer never starts at. To start such state elsewhere, supply it whole.

### Or supplied whole, by the caller

A caller who would rather materialise state themselves does so and hands the
optimizer over, and planning adopts what it finds rather than declaring
anything:

```python
optimizer = AdamW(model.parameters(), lr=rate)
import_optimizer_state(optimizer, runtime=runtime, pool="spill")
training = plan_step(model, optimizer=lambda params: optimizer)
```

The optimizer handed to planning is the reference. State imported for *that*
optimizer is adopted as it stands and outlives the plan, because the caller
owns it; state planning declared and started belongs to the plan, and closing
the plan releases it.

Importing state outside planning is therefore always valid and never changes
what planning does. It has an effect only when that optimizer is the one
planning is given -- state imported for an optimizer planning never sees is
simply not planning's business.

### Frozen parameters carry no state

A parameter that is not trained has no gradient in the declaration, an
optimizer skips parameters whose gradient is absent, and so nothing is
declared, allocated or filled on its behalf. This needs no handling and no
flag: it falls out of letting the optimizer declare its own state.

Both of the usual spellings work, and cost the same -- pass only the trainable
parameters to the optimizer, or pass all of them and leave the frozen ones
with `requires_grad=False`:

```python
model.embedding.weight.requires_grad_(False)   # keep the embedding fixed

trainable = [item for item in model.parameters() if item.requires_grad]
```

The frozen parameter itself still lives in the pool as model state, because it
is read every step. What does not exist is state for an update that will not
happen.

The flag is not the only way a parameter goes without a gradient. What decides
it is whether any captured backward produces one, which the graph pairs
already say, and a parameter the objective never reaches produces none however
it is flagged. Eager training steps over such a parameter for exactly that
reason; the plan reads the same fact off the captured pairs, so what the
optimizer declares and what the tasks bind agree by construction rather than
by the caller having flagged it correctly. The alternative is a plan that
keeps state nothing steps and reserves a gradient nothing writes.

### Master copies, and the dtype gradients are kept at

A model that computes at bf16 can still be trained at fp32.
`plan_step(..., master_dtype=torch.float32)` gives every weight the step
trains at another dtype a master copy at fp32, and builds the optimizer over
the masters in the weights' place. The optimizer is whatever the caller
passes, and nothing about it has to know: to it the masters are simply its
parameters.

- A master starts at its weight, cast to the master's dtype, and is created in
  the pool with the rest of the optimizer's state -- it is optimizer state,
  the plan's like the rest of it.
- The step computes with the weights as they are. The update, one traced graph
  per stage as always, steps each master and then writes its weight from it,
  cast to the weight's dtype, so every forward reads the weights as of the
  last update. The pool holds both, and the weights are what a plan fetches
  to compute with.
- An ordinary weight already at `master_dtype` is its own master, and nothing
  is added for it. A tensor wrapper still gets a dense master: its logical
  dtype can be fp32 while its physical payload is quantized.

For a [tensor representation](state-import.md#tensor-representations), the
update publishes the master with `compute.copy_(master)`. That operation's
captured graph owns quantization and writes every physical component. The
forward and backward tasks consume those components; the dense master belongs
to the optimizer task. There is one logical dense gradient per parameter,
regardless of the number of physical components.

Distributed master shards use the same publication rule after gathering the
logical compute value. For arbitrary scaled representations, quantizing each
flat shard independently would change the scaling recipe. Checkpoint restore
therefore stages one complete logical parameter on CPU before publication;
that temporary is bounded by the largest parameter, not the whole model.
The current generic sharded update gathers at the wrapper's logical dtype;
it does not infer a compressed collective format from its component dtypes.

`grad_dtype` is the dtype gradients are created and accumulated at, the
weights' own when it is `None`, and it is independent of the masters. With
fp32, each backward gives its parameters' gradients at fp32, and its
accumulating form adds each contribution into a gradient kept at fp32, so a
step's microbatches are summed at fp32. How a gradient comes out at that dtype
is decided by the operation that computes it in the backward graph, never by
the model:

- An operation that returns a gradient at `grad_dtype` already -- a kernel
  library asked for its weight gradients at fp32 -- has it converted to the
  parameter's dtype as it leaves, since autograd gives a parameter its
  gradient at the parameter's dtype. That conversion is dropped and the value
  the operation returned kept.
- A matrix multiply's result (`mm`, `bmm`, `addmm`, `baddbmm`), moved at most
  by views and copies on its way out and read by nothing else, is written at
  `grad_dtype` by the multiply itself where PyTorch has a kernel for that on
  the device: its products are summed at that dtype and never rounded to the
  operands'. Which dtypes it writes -- fp32 from bf16 or fp16 operands, say --
  is the operator's own check.
- Any other gradient is the backward's result, cast as it leaves. Compiled,
  the cast is fused into the kernel that computes the gradient, so one a
  reduction or an embedding lookup computes -- at fp32 from bf16, as Inductor
  does -- is stored without being rounded to bf16 first. A multiply is a
  library call that stores its result before anything reads it, which is why
  it is told the dtype instead.

The update takes each gradient at the dtype it is kept at and casts it only
where the parameter it steps is at another: fp32 masters over bf16 gradients
are cast in the update, over fp32 gradients not at all. The two are normally
given together.

A checkpoint saves one selected weight representation per logical parameter.
`weights="master"` is the default: save masters where configured, and restore
compute weights by casting. `weights="compute"` saves compute weights only and
upcasts them to restore any configured masters, without recovering precision
that was not saved. Distributed checkpoints keep only each owner's saved master
shard. Compute and master copies are never both checkpointed for one parameter.

Optimizer group fields omitted by the optimizer's own `state_dict()` are
reconstructed or retained as derived execution metadata. They are not additional
checkpoint state. Restoration retains their existing tensor objects so compiled
input bindings remain valid; saved fields still follow the checkpoint values.

The update casts and writes in the graph it captures, so an optimizer whose
update cannot be traced is refused masters, and gradients kept at another
dtype than its parameters, rather than run eagerly without them.

## Values that change between steps

An optimizer's update is captured once and replayed every step, so anything
read during capture is fixed for the life of that capture. A learning-rate
schedule therefore has to reach the update as something the capture did not
fix.

It can, and the capture's identity is what makes it work. A group value
contributes to that identity in one of two ways:

| the value is | it contributes | consequence |
|---|---|---|
| a float, int, bool or string | **its value** | each distinct value is a different capture |
| a tensor | **its geometry** -- shape, stride, dtype, device | every value shares one capture |

### The canonical form

Name the values a step may set when the step is planned, and set them on every
call:

```python
training = plan_step(
    model,
    optimizer=torch.optim.AdamW,
    hyperparams=("lr",),
    objective=objective,
    example_inputs=example_inputs,
    runtime=runtime,
    execution="device",
    spill="spill",
)
for step in schedule:
    training(step.microbatches, hyperparams={"lr": step.rate})
```

The optimizer is passed as it is -- no placeholder value, no wrapper. Planning
holds each named value in a scalar tensor before the update is captured, so the
capture takes it as an input; each call writes the values into those tensors.
A name that is a model buffer is a tensor already, and state the plan owns:
the module's tensor is the plan's handle on it after planning, so each call
writes the value into the pool the state lives in, once the previous step has
finished with the old one. Nothing is recaptured, nothing is recompiled, and
the step is otherwise unchanged.

That is the same split the state above follows: planning is handed a
declaration, and each step is handed the values.

### When a plain float is the better choice

Naming a value exists to let it change. If it never will, leave it out and pass
it to the optimizer as an ordinary number:

```python
optimizer = partial(AdamW, lr=3.0e-4)      # fixed for the plan's life
```

The cost is that it is fixed: the value is part of the captured update, so
`hyperparams={"lr": ...}` on a value that was not named is refused rather than
silently ignored, and the message says to name it. Choose the number when the
simplicity is worth giving up the ability to change it.

### What this asks of an optimizer

Naming a value in `plan_step(hyperparams=...)` makes ShadowSpill hold it in a
scalar tensor before the step is captured. Everything after that is the
optimizer's own doing, and an optimizer that means to support schedules has to
hold up four things. `torch.optim` already does.

1. **The value is reachable by name.** It lives in `param_groups` under the
   name a caller would use, or as a registered buffer on the model. Those are
   the two registries of named values that already exist, and they are the only
   places a name is looked up.
2. **A tensor is accepted where a number is.** A setting that can be named this
   way takes a scalar tensor as readily as a float.
3. **The update reads it, and does not branch on it.** Reading is what makes
   the value an input to the capture. Comparing it -- validating a rate is
   positive, say -- is a branch on data the capture cannot resolve, and turns a
   graph that could have been partitioned per stage into one opaque task.
   Settings are checked where they are set, not on every step.
4. **Its geometry never changes.** Shape, dtype and device are what the
   capture's identity records, so they are fixed for the plan's life; only the
   value moves.

A number is held in the widest type of its own kind -- float64 for a float,
int64 for an int -- because that is what a Python number already is. A **bool**
is refused, and the reason is the rule rather than an omission: a bool in an
optimizer selects what the update does (`amsgrad`, `maximize`, `nesterov`)
rather than scaling it, and what the update does is what was captured. Two
behaviours are two captures, and two plans, so a bool changes by planning again.

The scalars stay **on the host**, which is what makes setting one free. Nothing
is copied to the device and nothing is synchronized: the update reads the value
when it launches its kernel and passes it as a launch argument, and a caller
writing the next step's value writes host memory.

Why a tensor rather than passing the number itself: a number is read when the
step is *traced*, and what a trace reads it folds in. That is not a property
ShadowSpill can work around, and it is not uniform -- a float survives as an
input where it is a plain operand, and is folded where it reaches a `Scalar`
argument or a custom operator, which is an implementation detail of whichever
kernels the optimizer happens to call. A tensor is an input in every case,
which is why it is the contract.

What ShadowSpill will **not** do is hold a value nobody asked it to. An
optimizer is entitled to require a number -- to choose a branch, or to derive a
constant at the precision the number carried -- and turning one into a tensor
behind its back either changes what it computes or fails inside it, far from
the line that caused it. Naming a value is the caller saying that this one is
safe to hold, which is why the list is given and not inferred.

An entry holding several values, as `betas` does, has each of them held, and is
written element-wise: one number sets them all, a sequence sets them in order.

```python
training = plan_step(
    model,
    optimizer=torch.optim.AdamW,
    hyperparams=("lr", "betas"),
    objective=objective,
    example_inputs=example_inputs,
    runtime=runtime,
    execution="device",
    spill="spill",
)
training(batches, hyperparams={"lr": 3.0e-4, "betas": (0.9, 0.95)})
```

### Parameter groups

A name is written to **every group that has it**, because one schedule across
all groups is the common case and reads as it means:

```python
training(batches, hyperparams={"lr": rate})   # every group's lr
```

Groups that must differ -- a lower rate for embeddings, say -- are written
directly, which is the mechanism `hyperparams` uses underneath. Nothing here
is a ShadowSpill concept: parameter groups and the values in them belong to
the optimizer the caller defined, and a schedule that treats parts of a model
differently is written exactly as it would be without ShadowSpill.

```python
decay = torch.tensor(3.0e-4)
slow = torch.tensor(3.0e-5)
optimizer = AdamW(
    [
        {"params": list(model.blocks.parameters()), "lr": decay},
        {"params": list(model.embedding.parameters()), "lr": slow},
    ]
)

for step in schedule:
    decay.fill_(step.rate)
    slow.fill_(step.rate * 0.1)
    training(step.microbatches)
```

The only ShadowSpill-visible property is that both are tensors, so one capture
serves every pair of values they take; make them host scalars, as planning
does, because a value the update reads and passes to its kernel costs nothing
to write there and an optimizer may require it there. The two spellings
compose -- `hyperparams` for what is uniform, direct writes for what is not --
because both end at the same tensors.

### Model constants, by the same mechanism

Nothing above is special to optimizers. A model constant obeys the same rule
-- declared as a tensor, valued per step -- and `hyperparams` reaches it too,
because a model already has a registry of named tensors in its buffers:

```python
class Scaled(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("temperature", torch.empty(()), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.temperature.fill_(1.0)
```

```python
training(batches, hyperparams={"temperature": 0.9})
```

A name is looked for in the optimizer's groups and in the model's buffers.
Buffer names are the dotted paths `named_buffers()` reports, so a constant
inside a submodule is named as it is found.

Note that such a buffer is declared empty and filled by `reset_parameters`,
like any other derived constant -- the same contract as
[importing state](state-import.md), for the same reason.

### What is refused

Names are checked twice, at the two moments each mistake becomes knowable.

When the step is **planned**, `hyperparams=(...)` refuses a name that exists
in neither registry, a name that exists in both, and a name whose value cannot
be held in a tensor at all -- a flag, a mode, anything that is not a number or
a sequence of them. That is the earliest point at which any of the three is
knowable, and planning is where a caller can still change what they asked for.

When a step is **called**, `hyperparams={...}` refuses the same first two --
`KeyError` for a name in neither registry, because a value silently going
nowhere would look like a schedule that ran, and `KeyError` for a name in both
rather than picking one, since either choice would be silently wrong half the
time. It also refuses, with `TypeError`, a name still held as a plain number:
that value was fixed when the update was captured, so writing it now could not
take effect, and the message says to name it in `plan_step(hyperparams=...)`.
Refusing there is what makes the rule discoverable at the moment it matters,
rather than through a schedule that quietly does nothing.

## Observing parameters and gradients

`plan_step`, `build_step_programs`, and `plan_step_search` accept an optional
`parameter_metrics(weight, gradient)` callback. For example:

```python
def parameter_metrics(weight, gradient):
    return {
        "param_norm": torch.linalg.vector_norm(weight, dtype=torch.float32),
        "grad_norm": torch.linalg.vector_norm(gradient, dtype=torch.float32),
    }
```

This is a pure tensor computation. Each observation reads the model's compute
weight and the **final accumulated gradient**, before optimizer casts or
writes; when a master exists, the first argument is still the compute weight.
Parameters with no update/gradient have no observation. Tied parameters are
observed under their canonical name.

The PyTorch frontend captures a read-only observation task before each update
component. Its inputs, output tensors, workspace and runtime participate in
ordinary compilation, profiling and planning. This preserves stage-interleaved
updates and does not make the entire model resident at once. Results are
snapshots, so later updates cannot overwrite them.
Observation capture uses the same explicit device ordinal as the surrounding
training plan, including nonzero device ordinals in distributed processes.

`result.parameter_metrics[name]` has the callback's pytree, once per optimizer
step. **Its leaves are detached tensors on the device**, not Python numbers.
There is no `.item()`, CPU copy or synchronization inside these tasks or while
rebuilding the result. The caller copies the small returned summaries and
logs them after the planned step returns. The callback must not mutate either
argument or read tensor values on the host.

Loss/router observations belong in `ObjectiveResult.metrics` instead: those
are returned per microbatch and the caller decides how to aggregate them.
Logging libraries and aggregation policy remain outside ShadowSpill.

## See also

- [Importing state](state-import.md) for the same contract applied to model
  state, and for how dtype is decided.
- [The artifact store](../python/artifact-store.md) for how a captured update
  is keyed, stored and reused across processes.

Previous: [Importing state](state-import.md). Next: [The
ShadowSpillProgram](program.md).

## Distributed ownership

A distributed binding resolves each logical parameter's replicas and remaining
gradient contributors. ShadowSpill composes a local coordinate-wise optimizer
with explicit SUM/reduce-scatter, owned moment/master updates, and compute-weight
all-gather. Those operations live within the optimizer task and finish before its
completion. All tensor inputs, mutations, outputs and temporaries participate in
the ordinary capture/profile/admission contract.

Sharding is enabled by default and can be disabled with `shard_optimizer=False`.
The supplied optimizer performs local math. MLOps AdamW now provides only that
local operation; the training engine owns all communication and shard lifetimes.
See [distributed planning](../python/api/distributed.md) for the capability and
normalization contracts. A checkpoint saves masters or compute weights, selected
by `weights`, and reconstructs the other representation during restore.


## Alternative optimizers and distributed ownership

The capture/state/task path accepts a supplied `torch.optim.Optimizer`; it does
not assume AdamW moments or parameter-shaped state. Tensor-only updates, explicit
state starts, and tensor-valued changing hyperparameters let alternative algorithms
reuse compilation, profiling, admission, scheduling, metrics and checkpoints.

Distributed **flat elementwise sharding** is narrower. The current
`supports_flat_parameter_shards=True` capability asserts that arbitrary flat slices
can be updated independently. Matrix-wide algorithms must use
`shard_optimizer=False` until a complete-matrix ownership policy is implemented.
That mode retains matrix shapes and completes gradients before the local update.
Do not declare flat sharding safe for a matrix orthogonalization algorithm.

A future complete-matrix policy will assign one owner per update unit and reuse
the existing state and captured-task machinery, with owner-only update work and
compute-precision weight publication. It requires ownership/profile/checkpoint
coverage; it is not currently implemented. Models with different update rules can
use parameter groups within one local optimizer rather than separate trainers.

Startup diagnostics require the separate `zero_lr_preserves_state=True` promise.
It is optional for normal training: ordinary optimizers can mutate momentum or
other state even when LR=0, and must not advertise it without preserving that
state while executing the update work.
