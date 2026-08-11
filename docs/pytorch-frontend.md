# PyTorch Frontend

ShadowSpill's PyTorch frontend owns capture and ordinary task dispatch. It does
not move model arithmetic into the neutral runtime.

## Public forward callable

`forward_pass(model, example_inputs=..., device_budget=...,
host_budget=..., partition="auto")` is implemented. Planning must occur before
an incompatible CUDA allocator has initialized the process. Parameters and
buffers start on CPU; the returned callable gives the original registered
tensors CUDA identity while it owns the model and restores them to CPU on
`close()`.

Every invocation validates the complete tensor geometry, storage alias
relationships, and static metadata before writing persistent input slots.
PressureFit's initial placement and exact ordered actions are submitted to the
neutral runtime without trigger changes. Each compiled stage remains an
ordinary PyTorch call wrapped by `before_task` and `after_task`. Public output
pytrees are reconstructed from Export's user-output positions, and their live
slab allocations transfer to ordinary caller ownership without a copy.

`state_dict()` and `load_state_dict()` are explicitly synchronizing and use
ordinary CPU tensors with the original model names. `close()` is synchronizing,
idempotent, preserves Parameter objects and ties, restores host-authoritative
bytes, and unregisters plan objects. Caller-retained outputs remain valid after
close because they are no longer plan-owned.

Registered buffers marked `persistent=False` remain runtime-owned and are
restored on close, but they are omitted from checkpoints and are not required
by `load_state_dict()`, matching ordinary PyTorch. Loading a checkpoint first
preserves the current bytes of every non-persistent registration and then
overwrites only checkpoint-persistent entries.

The public `plan()` callable composes every fixed microbatch position into
forward, objective, backward, and gradient-accumulation tasks followed by one
optimizer update. It preserves the original model and optimizer checkpoint
schema and restores CPU storage on deterministic close.

## Fixed input contract

`TensorSpec` describes storage-free shape, stride, dtype, layout, and gradient
requirements. Real example tensors must be CPU or meta tensors. Every nested
non-tensor value is copied into the position-specific signature and compared
by exact type and equality before a call begins. Runtime tensors may arrive on
CPU or CUDA, but fixed-shape v1 requires their complete geometry to match.

Training examples have one signature per outer microbatch position. Different
positions may have different geometry or metadata. The outer length is the
gradient-accumulation count, not a catalog of interchangeable configurations.

## Capture boundary

Registered CPU parameters and buffers are copied into a storage-free FakeTensor
CUDA replica. Tied registrations, views, storage offsets, shapes, and strides
are retained without copying model contents. Strict Export lifts registered
state into explicit graph inputs. An empty decomposition table preserves
differentiable operator semantics until AOTAutograd constructs the explicit
forward/backward pair.

Training capture produces both save-oriented and min-cut recomputation graph
pairs. Metrics are detached tensor outputs or preserved static pytree leaves;
only the scalar objective is differentiated. Structural graph identities use
operator targets, tensor geometry, calling convention, and the pinned
Torch/CUDA implementation identity, not task ordinal or microbatch position.

`partition="auto"` discovers outer containers with repeated sibling module
types and splits the functional Export graph at those child boundaries. Nested
repeated groups, such as experts inside one repeated transformer block, stay in
their owning outer block. Prologue operations join the first stage and epilogue
operations join the last. If there is no repeated structure, the complete
graph is one legal stage. Opaque operations and data dependencies remain
ordinary FX edges; partitioning does not rewrite their semantics.

Each split stage receives its own save/recompute VJP. Profile keys canonicalize
FX dataflow without placeholder, node, layer, or task names. Tensor geometry,
gradient requirements, static arguments, operators, compiler/provider identity,
Torch/CUDA version, and device capability remain semantic. Thus identical
interior layers share one profile, while a first layer that does not return an
input gradient or a last layer containing the objective correctly remains a
different ABI.

PyTorch 2.13 AOTAutograd may initialize CUDA provider state even when its model
and examples are FakeTensors. Public planning therefore installs ShadowSpill's
allocator before entering Export/AOT capture. Calling the private capture
helpers directly is unsupported; the public planning session enforces this
ordering.

Registered custom operators are accepted when their normal PyTorch contracts
are complete: schema, fake/meta implementation, alias and mutation declaration,
and—when differentiated—an autograd implementation. ShadowSpill does not
special-case operation libraries.

Compiled task entrypoints execute with dispatcher autograd disabled. Training
already contains the explicit AOTAutograd forward/backward programs, and
forward-only mode intentionally captures no backward. This prevents registered
custom operations from creating a second hidden autograd context and
saved-tensor lifetime outside the canonical Program.

Structural profiling also audits allocations retained by provider code or
custom operations. Retention is accepted only when repeated isolated calls
reach a stable logical-live-byte baseline; its observed high-water is reserved
from the slab and reported as fixed slab use. Continued growth is rejected as
an unbounded operation contract rather than hidden inside workspace leeway.

## Optimizers

Optimizer capture has no class allowlist. The optimizer created by the caller's
factory is inventoried by parameter identity, then copied into a capture
sandbox. A discovery update identifies lazy state without mutating the caller's
parameters or optimizer. The first update remains a separately profiled opaque
task when it creates Python or tensor state. Once the state structure is stable,
parameters, gradients, tensor state, and tensor-valued group options are lifted
into an explicit recurrent graph when PyTorch can represent the update.

An optimizer whose valid Python update cannot be represented as a graph remains
a bounded, profiled opaque task. ShadowSpill materializes an isolated CUDA copy
of its standard parameter/state inventory, measures the eager update through
the same allocator telemetry, and feeds that structural profile to PressureFit.
Steady-state execution still calls the user's ordinary `Optimizer.step()` under
the annotated task boundary. An optimizer that cannot be copied or whose state
cannot be discovered is rejected before a training step; unknown workspace is
never treated as free.

CUDA-only registered optimizer operations use the same generic path. When CPU
discovery constructs lazy state and then rejects its kernel, the copied standard
optimizer inventory is converted to FakeTensor CUDA values and the recurrent
update is captured through each operation's fake/meta contract. ShadowSpill
contains no `mlops` import; an externally supplied optimized optimizer uses this
boundary without a provider-specific branch.

## Compilation and profiling

The private PyTorch artifact retains representative arguments only for task
compilation; those framework values never enter canonical ShadowSpill IR or its
serialized cache identity. Real CUDA representatives preserve shared storages,
offsets, shapes, strides, dtypes, and gradient requirements. Inductor compiles
the explicit task graph once, and warmup runs occur before measurement.

Calibrated runtime samples use CUDA events. One additional isolated invocation
runs inside the production `before_task`/`after_task` boundary while the slab
allocator records requested and charged allocation lifetimes. Workspace is the
maximum simultaneously-live anonymous extent set, not allocation volume.
Returned task storages are identified through a read-only exact pointer lookup
and excluded from workspace without promoting them into permanent runtime
objects. Content-addressed profiling invokes this machinery once per structural
ABI; a warm cache launches no compilation or profiling kernels.

Structural task profiles and complete PressureFit selections are cached
independently. `PlanReport.profile_cache_hits` and `profile_cache_misses`
describe executable ABI measurements; `recomputation_cache_hits` and
`recomputation_cache_misses` describe complete simulator-validated schedule
selections. A warm selection cache still runs one standalone simulator replay
before physical slab admission.

## Canonical lowering

Lowering assigns dense identities in encounter order and never serializes
framework objects. One physical storage becomes one alias group; tied tensors
and views become logical objects within that group with explicit byte offsets
and sizes. Registered parameters and buffers are checkpoint-persistent and
retain host backing. Caller inputs, intermediate activations, and returned
outputs have step-scoped identities and declared initial/final residency.

Each automatic stage becomes one canonical task with its structural profile,
dependencies, inputs, outputs, and compute resource. A separate private
entrypoint binding retains only the stage module target, compiled artifact, and
tensor leaf positions needed by the PyTorch executor. The resulting `Program`
is accepted directly by the standalone simulator and PressureFit; neither sees
PyTorch tensors, graph modules, pointers, or operation names.
