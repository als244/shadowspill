# Model tensors and initialization

Build records model structure; execution owns its values. A saved Program or Plan
contains names, shapes, dtypes, storage relationships and initialization
descriptions. It does not contain a checkpoint or a live Python model.

See [the offline pipeline](../python/phases.md) for the phase boundaries. This
page explains how tensors cross the execution boundary without creating another
model-sized host allocation.

## Declare first, fill later

Construct large models on `meta`, with parameters and buffers already at their
intended dtypes. The builders inspect that structure and capture its computation.
They do not install runtime pools or initialize persistent model values. Kernel
finalization can require a representative GPU; this is separate from state loading.

Create Executor in a fresh process from a saved Plan. It admits the physical
layout, allocates persistent storage and accepts a bounded initializer.

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

The shared workload checkpoint readers use bounded positional reads when file
and destination dtypes match. The file reader fills the supplied CPU destination
directly; several disjoint reads can run concurrently within that destination.
Every read completes before the callback returns, including when another read
fails. This avoids a full-tensor staging allocation and does not change the
Executor's buffer lifetime. Layout transformations and dtype conversions retain
their bounded copy paths. The application/provider owns file I/O; the compiler
and runtime do not inspect model-specific checkpoint keys.

Keep random-number block boundaries separate from copy granularity. A loader can
generate fixed seeded blocks, assemble several in bounded host scratch, and copy
them together without changing the values assigned to any shard. This avoids
repeated CPU thread-pool launches for small conversions. Fill constant tensors
directly in the supplied destination; generating and copying zero blocks is
unnecessary. These are loader optimizations, independent of execution chunking
and checkpoint-region policies.

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

Owned optimizer updates support a single contiguous physical payload when the
wrapper preserves its logical shape through reshape, slicing and copy. Its
physical dtype still controls storage and publication. Packed or scaled wrappers
that do not preserve that shard/copy relationship remain unsupported for sharded
updates. Frozen wrappers, as used by LoRA, require no logical master.

## Export and checkpoints

`Executor.export_tensors(write_chunk, chunk_bytes=...)` provides bounded CPU views
using the same names and storage offsets. Consume each view before the callback
returns; stream it to files without retaining a full-model snapshot.

`save_checkpoint` and `load_checkpoint` use this interface for atomic archives.
Each checkpoint is an uncompressed TAR archive containing a JSON manifest and
packed tensor payloads, including layout metadata and BLAKE3 content hashes.
The manifest has its own small content hash. TAR checks only its headers, so
there is no second full-payload CRC pass. This is a ShadowSpill archive, not a
`torch.load` state dictionary. By default, `weights="master"` stores each ordinary
master weight once and reconstructs its compute copy on load. Parameters without a
separate master are stored normally. `weights="compute"` instead saves the
compute copy and casts it into the master on load, intentionally losing the
master's additional precision. Optimizer state and buffers remain independent
of this choice.

The selected representation is authoritative. Reconstruction uses captured
optimizer ownership, registered storage aliases and the receiving rank's tensor
geometry. A local source must cover the destination's complete owned region.
Replicated compute weights can instead be reconstructed collectively from the
restored master shards in their declared optimizer group. Each rank saves only
its master shard; after all local reads and checksums succeed, bounded CPU
broadcasts cast and fill the already allocated replicas. This adds no full
GPU weight allocation and runs no optimizer update. Normal sharded DP still
derives its local compute shard and gathers it during normal execution.
Packed
or scaled tensor components, storage with holes or unused backing elements,
incompatible storage layouts, and unsupported ownership combinations retain
the required physical copies; the manifest's
`retained_weight_copies` records the reason. A plain payload is never mistaken
for a logical encoded weight. Frozen encoded bases can still be omitted using
`frozen_state_id` independently of this policy.

The archive records the selection, local derived tensors and collectively
gathered tensors, validates them against the receiving Program, and checksums
each stored source completely.
Restoration uses at most one ordinary CPU scratch chunk for casts, in addition
to the destination supplied by Executor. It rereads the selected source when
filling the derived copy, avoiding a whole-model snapshot or dependency on
Executor's initialization order. This trades extra read/checksum work during
restore for a smaller checkpoint. A subregion restore still validates all bytes
of its selected source. Collective reconstruction reads the restored host shards
instead of rereading the archive and uses bounded host transfer buffers.
`Executor.initialize(replicated_copies={name: (source_name, group_ref)})` is the
underlying initialization primitive; it is separate from archive parsing and
requires every worker to participate. Version 2 archives encode this contract;
older archives without it are not accepted.

For large export chunks, BLAKE3 hashing overlaps the archive write in one CPU
worker and uses four native hashing threads. The same native pool is reused
across tensors; small chunks use a serial hasher. Both hashing and writing finish
before the export callback returns; no borrowed buffer is retained afterward.
Payload verification, file synchronization, atomic replacement and destination
directory synchronization remain part of the synchronous save operation.
Reads fill the destination directly, without an additional archive-sized or
chunk-sized byte-string copy. The immutable Program identity is cached, so
repeated saves do not serialize the entire graph again merely for provenance.

This version change concerns training checkpoints only. Saved Programs, Profiles
and Plans keep their existing artifact format.

```python
from shadowspill import load_checkpoint, save_checkpoint

save_checkpoint(executor, "step.ssp", chunk_bytes=64 << 20)
load_checkpoint(executor, "step.ssp", chunk_bytes=64 << 20)

# Optional smaller checkpoint: restoring masters loses compute-rounding bits.
save_checkpoint(executor, "compute-step.ssp", weights="compute")
```

Restore validates the logical rank and the training state, not the Program: the
name, geometry and owned region of every persistent tensor, the additional
Plans' private tensors included. A rebuilt Program, other kernels or another
microbatch geometry restore the same state, and a different inventory is refused
with the differing tensors named. Low-level callers store their progress,
schedules, source position and RNG in checkpoint metadata. The training service
preserves its session and worker RNG state; client-owned dataset progress remains
with the client. An evaluation Plan shares current model storage; only its
additional, nonshared tensors need separate checkpoint payloads.

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

Previous: [Step structure and ordering](step.md). Next: [The
optimizer](optimizer.md).
