# Executor and shared plan state

Execute an admitted saved Plan and own its persistent state. Use
the [training client](../client.md) for batch submission and service-managed execution.

## Fresh worker and invocation

Load the saved Plan before constructing an Executor. The worker must not have
initialized CUDA with another allocator. Build/measurement belong in a different
process; this worker needs neither the original model object nor compiler caches.

```python
from shadowspill import Executor, Plan

executor = Executor(Plan.load("training.ssp"))
executor.initialize(fill_tensor)  # fills provided, bounded CPU spill views
pending = executor.run(args=[(features,)], kwargs=[{"targets": targets}])
result = pending.result()
```

`run` returns a result handle; the next `run` waits for an outstanding invocation.
Forward results preserve the model's output structure. Training returns losses
and metrics per example. Outputs are caller-owned across invocations; release
held GPU outputs before closing the Executor. A refused close can be retried
once those outputs have been released.

## Sharing state across Plans

An Executor can admit named forward, backward, optimizer-only and complete-step
Plans alongside its default Plan:

```python
executor = Executor(training_plan, plans={"evaluation": validation_plan})
executor.initialize(fill_tensor)
update = executor.run(args=training_examples).result()
predictions = executor.run(plan="evaluation", args=validation_examples).result()
```

All Plans must use the same target/build, logical parallelism and memory budgets.
Model tensors, masters, optimizer state and persistent gradients are matched by
logical identity, exact dtype/shape/stride/storage offset, ownership and aliasing.
Internal object IDs may differ. State has one canonical owner even when it first
appears in a later Plan; CPU optimizer counters share that ownership too.
Evaluation reads current compute weights and buffers while preserving pending
gradients in spill. Additional forward Plans are read-only unless explicitly
declared training-mode in their saved metadata. Different input shapes are
supported by separately built Plans; execution never rebuilds one.

Admission checks capacity for all Plans before opening pools. Sequential calls
reuse one fixed GPU arena sized for the largest saved layout. Persistent spill
storage is the union of shared persistent state and each Plan's private inputs and
constants, plus active invocation requirements. Fixed device overhead is the
worker's, which opens its context, provider handles and resources, stream
workspaces and reserved headroom once for every Plan it holds, so the largest
Plan's overhead counts once. Plans that fit separately can still fail joint
admission: the largest fixed slice and the largest dynamic requirement need not
belong to one Plan, and every Plan's persistent spill storage is held at once.
Initialization/export visit shared state once; private constants and controls use
`plans.<name>.<tensor-name>`. A checkpoint binds to that tensor inventory, the
private tensors included, rather than to the Programs.

## Gradient accumulation and controls

A forward/backward Plan contains both fresh and accumulating invocation paths.
The default starts fresh gradients. `accumulate_gradients=True` uses stored
gradients. Both paths share a physical arena and persistent tensor identities;
the archive includes both schedules and memory admission records. There is no
runtime planning or hidden eager gradient addition.

Named backward Plans support the same flag, including accumulation across
separately built input shapes:

```python
executor.run(plan="backward_small", args=first_examples).result()
executor.run(plan="evaluation", args=validation_examples).result()
executor.run(
    plan="backward_large", args=next_examples, accumulate_gradients=True
).result()
```

This low-level interface leaves accumulation sequencing to its caller. It does
not infer whether an earlier contribution participated or apply an optimizer
update when a backward call completes.

`hyperparams={"optimizer_group.0.lr": value}` overrides a captured tensor
hyperparameter for that call. Omitted entries use the captured defaults;
Python numeric optimizer settings remain specialized constants. CPU controls
remain on the host, and device control tensors occupy planned memory.

For a step built with `runtime_gradient_scale=True`, pass the saved
`plan.program.bindings["gradient_scale_control"]` name in `hyperparams` to scale
the final gradient at each optimizer consumer. The default is 1 on every call;
an override is not sticky. Optimizer-only Programs retain the same control.
Scaling follows conversion to optimizer precision and leaves stored gradients
unchanged. A zero scale still executes the optimizer, including state/counter
updates and weight decay; it does not imply an empty or skipped update.

## Initialization and export

Initialization uses `TensorChunk` metadata and fills an Executor-owned buffer.
Normal initialization loads model tensors and applies optimizer/gradient defaults.
`initialize(fill_tensor, restore=True)` instead asks the loader for all persistent
tensors, including optimizer data and gradients. `export_tensors(write_chunk)`
streams the same named storage in bounded views. Callbacks must consume/fill each
view before returning and must not retain it. Checkpoint file formats belong to
helpers above this interface, not the allocator or task dispatcher.

`export_tensors(write_chunk, names={...})` selects exact canonical names from
`tensor_chunks()`. Unknown names fail before tensor I/O. Unselected spill storage
and host controls are not read; an empty set exports nothing. Omitting `names`
keeps the complete state export. Selection does not apply to initialization.

Worker shard descriptions preserve offsets into the original tensor storage.
Load/export callbacks see only valid owned elements; padded tail elements are
zeroed internally. Master initialization converts directly from the corresponding
local compute-weight shard into each bounded destination chunk. Optimizer bindings
record their associated parameter explicitly, including names containing dots.

## Checkpoint storage

`save_checkpoint(executor, path, metadata={...})` streams persistent tensor storage
into one atomic archive. `load_checkpoint(executor, path)` fills the same bounded
views and returns the caller metadata. A checkpoint binds to the training state,
not the Program: every persistent tensor's name, geometry and owned region. A
rebuilt Program, other kernels, another microbatch geometry, memory budgets or
scheduling order restore it; a different inventory is refused with the differing
tensors named. It preserves storage geometry, exact dtypes and CPU optimizer
controls. Chunk sizes can differ between writing and reading. Each tensor is a
packed archive member, so there are no per-tensor filesystem writes or
complete-model host snapshots. Format, geometry and checksum checks precede
successful initialization. Application scheduler/RNG state belongs in the
caller's JSON metadata; it is not discovered implicitly.

`checkpoint.write_weights(executor, stream, weights="master")` uses the same
bounded archive writer for inference model state. `checkpoint.weights_size(...)`
returns its exact size without reading tensors. The archive records logical
parameter/buffer views, tied storage, owned shard coordinates and authoritative
weight selection. Optimizer state, gradients, RNG and execution temporaries are
excluded before storage reads. Keep the Executor quiescent while describing and
writing it; the caller owns stream publication and the complete rank set.

A verified `frozen_state_id` permits omission only of parameters proven unchanged
across all admitted Plans. Persistent buffers and modified frozen weights remain
in the export. Without that identity, all model state is retained. The distinct
`shadowspill.weights` format cannot be used for optimizer resume. Model encodings
without declared logical geometry require an installed format adapter; ownership
is never guessed from tensor names or element counts.

## Output and storage views

Returned views of input or parameter storage become owned snapshots during build.
Those copies are compiled and profiled as part of the task. Views sharing that
storage still share their returned snapshot; changing it cannot change stored
parameters. Mixed-dtype views of one returned input storage currently require
explicit model-side copies. TensorChunk includes stride and storage offset for
loaders handling noncontiguous registrations; chunk offsets address physical
storage rather than a flattened logical tensor.
