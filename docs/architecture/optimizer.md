# The optimizer

`build_step` captures a supplied optimizer's local update. Build declares tensor
geometry and initialization recipes; Executor later owns and initializes the
storage. Planning decides update order without constructing an optimizer or
allocating model state. See the [phase API](offline-pipeline.md) for the complete
build, profile, plan and execution sequence.

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=torch.tensor(3e-4), foreach=False,
)
program = build_step(
    model, optimizer=optimizer, objective=objective,
    example_args=(features,), example_kwargs={"targets": targets},
    num_examples=4, grad_dtype=torch.float32, master_dtype=torch.float32,
)
```

The optimizer must refer to the supplied model's parameters. Build copies it to
meta tensors for discovery and capture; it does not mutate the original model or
optimizer. A portable update must be traceable. An opaque optimizer is rejected
instead of silently executed outside the saved Plan.

## Declaring and initializing tensors

Discovery runs the optimizer on meta parameters to identify its tensor shapes,
dtypes and starting values. Moments need not have the parameter's dtype or shape.
The Program records these declarations without retaining a model-sized payload.

### Started where the optimizer starts it

The optimizer's own initialization determines the saved recipe:

| Initialization | Saved behavior |
| --- | --- |
| Constant fill, such as zero moments | Fill the owned storage with that constant |
| Copy of a parameter | Copy the initialized parameter, casting to the entry's dtype |
| Bounded tensor value | Save that small value with the Program |
| State already held by the supplied optimizer | Require the execution initializer or checkpoint to supply it |

The initial implementation rejects state that can only be created from the first
gradient, such as a lazily initialized SGD momentum buffer, with an explicit
first-use-initializer error. It also refuses to embed optimizer initialization
values larger than 1,024 elements. Existing model-sized state remains external.

In a fresh execution worker, `Executor.initialize(fill_tensor)` fills bounded
Executor-provided buffers. Constant and parameter-copy defaults are applied by
Executor; the loader supplies external contents. There is no full model or
optimizer bank allocated beside the spill pool. Checkpoint loading uses the same
bounded tensor interface.

### Frozen parameters carry no state

Only parameters reached by the captured backward receive gradients and optimizer
state. A frozen parameter remains available to forward/backward but has no update,
master, moments or gradient allocation. Passing all parameters to the optimizer
or passing only trainable parameters both preserve this behavior.

### Master copies, and the dtype gradients are kept at

`master_dtype` and `grad_dtype` are independent build controls:

- `master_dtype` selects the optimizer's parameter precision. A separate master
  starts from its compute weight; the update publishes the changed compute weight
  after updating the master. An ordinary weight already at that dtype needs no
  additional master. Trainable tensor wrappers require an explicit `master_dtype`
  and always have a dense logical master, even when their advertised dtype
  matches it. Frozen wrappers need no master.
- `grad_dtype` selects stored and accumulated parameter-gradient precision. The
  backward keeps provider-produced wide gradients and requests wide GEMM outputs
  where supported, instead of rounding a result and then widening it. Other
  conversions remain part of the compiled backward.
- Moment and counter dtypes come from the supplied optimizer. Selecting FP32
  masters does not itself override the optimizer's state policy.

Quantized weights expose physical payloads and scales through the
[tensor representation interface](state-import.md#tensor-representations), while
their gradient remains one dense logical tensor. Publishing a master uses that
representation's copy operation. Sharded tensor-wrapper updates require a
representation-specific shard/copy implementation and are currently rejected by
the generic owned-update builder.

The bounded initializer receives physical names such as `weight.payload` and
`weight.scale`, and `master.weight` for the logical master. For ordinary weights,
Executor initializes the master by casting the matching stored weight chunks.
For tensor wrappers, the caller must also fill `master.weight`: physical leaves
alone do not define a generic chunkwise dequantizer. Build captures the wrapper's
copy operation to publish updated payloads/scales; execution never reconstructs
the Python wrapper. Saved checkpoints retain both the exact physical tensors and
the master, so resume requires no dequantization or wrapper-class import.

## Values that change between calls

A numeric optimizer setting is captured as a constant. To change a setting
without rebuilding, provide it as a tensor when constructing the optimizer. Its
geometry is fixed, but its value becomes a saved runtime binding.

```python
# After initialization, in the fresh execution worker:
pending = executor.run(
    args=[(features,)] * 4,
    kwargs=[{"targets": targets}] * 4,
    hyperparams={"optimizer_group.0.lr": 1e-4},
)
result = pending.result()
```

Use one input call per example declared in the Program. Each group has its own
binding, such as `optimizer_group.1.lr`. Trainer schedules use those same names.
The original Python optimizer is no longer a live execution handle: mutating it
after build does not change a saved Plan.

Executor accepts CPU values of the saved shape, validates every supplied name and
shape before changing any control, and uses the captured default for omitted
values. A tensor binding without a saved default must be supplied. Unknown names
and settings captured as constants cannot be changed through `hyperparams`.
This interface covers saved optimizer bindings, not arbitrary model buffers.
Model inputs and their static arguments keep the ordinary builder/input contract.

Bounded CPU counters and controls can stay on the host when the captured update
supports that layout. Their values are read at execution, rather than becoming
separate GPU transfers merely because they belong to an optimizer.

## Ordering and distributed ownership

Build records the local update and its dependencies. Planning chooses
`optimizer_ordering="stage_interleaved"` or `"tail"`. An update may run only after
all contributing backwards and all tasks that read its old weight have finished.
Interleaving therefore never changes weights still needed by recomputation.

The central `parallelism` specification determines parameter replicas, gradient
contributors and storage ownership. Gradient reduction, local update and weight
publication have separate IR cost records, but execute consecutively on the same
ordered compute resource. Local update build/profile requires no process group;
communication timing and memory evidence come from `CommunicationInfo`.

Parameter fetch and gradient eviction retain the native compound-transfer
completion boundary. Final optimizer reduction writes a separate owned output,
so a concurrent spill of the original accumulator remains valid. This design
does not introduce independently scheduled optimizer communication streams.

Flat sharding requires the optimizer capability
`supports_flat_parameter_shards=True`: each flat slice must be independently
updatable. Matrix-wide algorithms cannot claim that capability and must use
unsharded optimizer ownership until their update-unit placement is supported.
Sharded builds start with an uninitialized optimizer and restore persistent
optimizer tensors in the execution worker.

## Observing parameters and gradients

`Trainer(plan, parameter_norms_every=N)` enables optional eager norm observations.
Weights are observed at an existing forward/backward read; final gradients are
observed before their local optimizer update. The collector uses bounded,
budgeted scratch and observes compute weights even when a master exists.
Distributed observations count owned regions once and combine their summaries.

These diagnostics are not compiled tasks and do not affect simulated task
durations. They contribute to measured training time only on calls where
collection is enabled. Ordinary runs launch no norm kernels. Arbitrary compiled
`parameter_metrics` callbacks are not part of the new builder interface.
Loss and model metrics can still be objective outputs; Trainer's host reducer
aggregates completed observations. Logging policy stays in the workload.

## Checkpoints

`save_checkpoint` streams the Executor's persistent tensors, including configured
compute weights, masters, optimizer state and host controls. It does not select
one weight representation and reconstruct the others. Distributed callers save
one archive per logical rank, containing that rank's owned regions.

An explicit `frozen_state_id` allows unchanged frozen parameters to be omitted.
Trainable tensors, optimizer state, buffers and tensors mutated by an admitted
Plan remain in the archive. Restore validates the frozen-base identity and uses
the bounded loader for omitted contents. Trainer adds update count and
caller-owned progress metadata; the caller supplies any data/RNG progress needed
for exact continuation. See [training composition](training.md).

Previous: [Importing state](state-import.md). Next: [The logical program](program.md).
