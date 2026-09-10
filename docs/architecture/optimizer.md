# The optimizer

What ShadowSpill needs from an optimizer, what it promises in return, and why
the boundary falls where it does.

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
read as it is rather than assumed to match.

This costs nothing and asks nothing of the caller. ShadowSpill does it with
the optimizer the caller already passes.

### Allocated in the pool

Every declared entry is allocated in the spill pool, which is where it will
live for the run. Nothing is allocated outside it, and there is no size below
which an entry is treated differently: a step counter an optimizer keeps as a
scalar tensor is allocated there like any other entry, and reads from the
host as it would anywhere else. An entry an optimizer keeps as a plain Python
number is not a tensor, so it is not declared and nothing is allocated for
it.

### Filled by the caller

ShadowSpill does not fill state. A default of zeros would be an assumption
that fails silently -- an optimizer whose state starts elsewhere would train
subtly wrong rather than fail -- and it would be ShadowSpill deciding a value.

So the caller passes an initialiser as `plan_step(optimizer_state_init=...)`,
beside the optimizer it belongs to:

```python
def zero_state(
    name: str, tensor: torch.Tensor, parameter: torch.nn.Parameter
) -> None:
    """Moment-based optimizers start at zero; this one says so."""

    with torch.no_grad():
        tensor.zero_()
```

It is called once per declared entry, with the entry's name, the pool-backed
tensor to fill, and the parameter the entry belongs to -- so an initialiser
that depends on the parameter can see it. Per entry rather than per state
mapping, so an initialiser needing scratch gets it for one entry at a time and
the transient stays bounded by construction.

An optimizer that declares state with no initialiser to fill it is refused at
planning, naming the entries it declared. Running on whatever the pool memory
held would be the same silent wrong answer that
[a missing `reset_parameters`](state-import.md#what-is-refused) would be.

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
owns it; state declared and filled through the initialiser belongs to the
plan, and closing the plan releases it.

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
    optimizer_state_init=optimizer_state_init,
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
Nothing is recaptured, nothing is recompiled, and the step is otherwise
unchanged.

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
    optimizer_state_init=optimizer_state_init,
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

### What does not belong here

Metrics and logging are not inputs to the update. They are computed from what
a step produced, and the caller computes them around the step rather than
through it -- putting them into the captured graph would make them part of the
program being planned, which they are not.

## See also

- [Importing state](state-import.md) for the same contract applied to model
  state, and for how dtype is decided.
- [The artifact store](../python/artifact-store.md) for how a captured update
  is keyed, stored and reused across processes.
