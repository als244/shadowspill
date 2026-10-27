# Model tensors and initialization

Build records model structure; execution owns its values. A saved Program or Plan
contains names, shapes, dtypes, storage relationships and initialization
descriptions. It does not contain a checkpoint or a live Python model.

See [the offline pipeline](offline-pipeline.md) for the phase boundaries. This
page explains how tensors cross the execution boundary without creating another
model-sized host allocation.

## Declare first, fill later

Construct large models on `meta`, with parameters and buffers already at their
intended dtypes. The builders inspect that structure and capture its computation.
They do not install runtime pools or initialize persistent model values. Kernel
finalization can require a representative GPU; this is separate from state loading.

Create Executor in a fresh process from a saved Plan. It admits the physical
layout, allocates persistent storage and accepts a bounded initializer.
`Trainer.initialize` delegates to the same interface.

```python
from shadowspill import Executor, Plan

plan = Plan.load("training.ssp")

def fill_tensor(request, destination):
    # Application-owned reader fills exactly this physical storage region.
    checkpoint_reader.copy_chunk(request, destination)

with Executor(plan) as executor:
    executor.initialize(fill_tensor, chunk_bytes=64 << 20)
    pending = executor.run(args=[(features, targets)])
    result = pending.result()
```

`checkpoint_reader`, `features` and `targets` are application inputs here. The
loader writes into Executor-provided CPU buffers. It must not replace them,
retain them beyond the call or load the whole model just to serve a chunk.
Random initialization and derived constants use the same callback.

An ordinary materialized model can also be built, but its payload is not embedded
in the artifact. Supplying its values at execution remains explicit.

## What a chunk describes

`TensorChunk` names a contiguous region of physical storage. Its offsets are in
**dtype elements**, not bytes or flattened logical tensor indices.

| Field | Meaning |
| --- | --- |
| `name` | Canonical tensor/component name supplied to the loader |
| `dtype` | Declared physical dtype of the destination |
| `shape`, `stride`, `storage_offset` | Original tensor view over storage |
| `storage_elements` | Complete physical storage extent |
| `offset` | First storage element in this callback |
| `region_offset`, `region_elements` | This worker's valid storage region |

The destination's length gives the number of elements to fill. A transposed view
cannot generally be initialized by flattening its logical values: respect the
storage ordering. Tied tensors and views sharing storage are initialized once.
Their relationships are saved in Program and reconstructed by Executor. Empty
tensors retain their geometry without requiring a payload callback.

With supported distributed storage, workers receive only their declared shards.
Offsets remain relative to the original storage; the runtime initializes internal
padding. Loading a shard does not require allocating the complete tensor. See
[distributed execution](../python/api/distributed.md) for supported layouts.

`chunk_bytes` bounds the supplied slice, except that one element is indivisible.
The application must also bound its scratch. A file-backed loader can read only
intersecting ranges. Reproducible sharded random initialization should use tensor
identity and global offset rather than depend on request order.

## Initial values and precision

The model declares compute-weight and buffer dtypes. The optimizer declares moment
and counter dtypes. Build's `master_dtype` and `grad_dtype` select logical master and
stored-gradient precision independently. See [optimizer capture](optimizer.md).

Executor fills captured constants and declared optimizer defaults itself. Ordinary
masters can be initialized by casting matching compute-weight chunks directly
into their destinations. The callback supplies externally initialized tensors,
including model parameters and buffers. Restore uses saved values instead of
initialization defaults.

Models can mix BF16, FP32 and compressed storage. ShadowSpill does not widen frozen
weights or choose quantization schemes. Frozen parameters still consume storage
and participate in forward/backward but have no masters, parameter gradients or
optimizer moments.

Derived buffers, such as positional tables, have explicit geometry and dtype. The
loader computes their requested ranges. Executor does not invoke a module's
`reset_parameters()`: it does not reconstruct the model.

## Tensor representations

Registered tensors can implement PyTorch's `__tensor_flatten__` /
`__tensor_unflatten__` protocol to expose ordinary physical tensors: for example,
integer payloads, scales and transposed payloads. Nested wrappers use the same
traversal. ShadowSpill accounts for the components and their aliases; a wrapper's
advertised dtype does not determine its storage size.

Build uses wrapper operations to capture the computation. Saved tasks take
ordinary component tensors, so execution and restore need neither the Python
wrapper instance nor its class. Custom runtime operations still need their
recorded, importable provider registrations.

All physical storage must be exposed by the protocol. Wrapper metadata must be
serializable, data-free configuration. Hidden library allocations and live
communication handles are not managed tensor storage. This protocol covers
registered model state, not arbitrary wrapper-valued public inputs and outputs.

Logical backward defines one dense gradient for a trainable wrapper. Integer
payloads and scales are not independent trainable parameters. Trainable wrappers
require an explicit `master_dtype` and a separate dense master, even when that
dtype matches the wrapper's advertised dtype. For example:

| Initializer name | Value |
| --- | --- |
| `weight.payload` | Quantized physical payload |
| `weight.scale` | Quantization scale |
| `master.weight` | Dense logical optimizer parameter |

The caller fills all three through bounded requests. Arbitrary physical components
do not define a universal chunkwise dequantizer, so Executor cannot infer the
master's initial contents. After updates, the wrapper's traceable
`wrapped.copy_(dense)` operation publishes every changed component. The tensor
library owns that math; ShadowSpill does not choose or reinterpret it.

Owned optimizer updates of sharded wrappers still require a representation-specific
shard/copy contract and are rejected by the generic builder. Frozen wrappers, as
used by LoRA, require no logical master.

## Export and checkpoints

`Executor.export_tensors(write_chunk, chunk_bytes=...)` provides bounded CPU views
using the same names and storage offsets. Consume each view before the callback
returns; stream it to files without retaining a full-model snapshot.

`save_checkpoint` and `load_checkpoint` use this interface for atomic archives.
Each checkpoint contains a JSON manifest and packed tensor payloads, including
layout metadata and checksums. This is a ShadowSpill archive, not a `torch.load`
state dictionary. Physical components and masters are retained separately and
restored without conversions.

```python
from shadowspill import load_checkpoint, save_checkpoint

save_checkpoint(executor, "step.ssp", chunk_bytes=64 << 20)
load_checkpoint(executor, "step.ssp", chunk_bytes=64 << 20)
```

Restore validates the Program, logical rank, additional Plans and tensor geometry.
Trainer checkpoints also preserve loop progress, schedules, source position and
RNG. An evaluation Plan shares current model storage; only its additional,
nonshared tensors need separate checkpoint payloads.

For partial checkpoints, `frozen_state_id` identifies the external frozen base.
Only unchanged frozen parameters can be omitted. Trainable tensors, buffers,
optimizer state and mutated parameters remain included. Restore requires the
matching identity and a bounded loader for omitted tensors; it does not assume
that an already initialized base is correct.

## Library memory and pool tools

Profiles distinguish retained provider setup allocations from per-task workspace.
The former contributes to fixed memory; the latter belongs to the task allocation
contract. Memory outside the framework allocator is covered by external headroom.
A process-global cache must not retain a task-scoped allocation after its Plan
releases it.

Executor uses pinned-host spill by default and also accepts `SSDPool` or
`RemotePool`. Every worker owns its spill pool. The same bounded tensor loader
and checkpoint interface serves each backend; model-based pool import is not
part of the public workflow.

Previous: [The step artifacts](step.md). Next: [The optimizer](optimizer.md).
