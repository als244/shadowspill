# PyTorch Frontend

ShadowSpill's PyTorch frontend owns capture and ordinary task dispatch. It does
not move model arithmetic into the neutral runtime.

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

## Optimizers

Optimizer capture has no class allowlist. The optimizer created by the caller's
factory is inventoried by parameter identity, then copied into a capture
sandbox. A discovery update identifies lazy state without mutating the caller's
parameters or optimizer. The first update remains a separately profiled opaque
task when it creates Python or tensor state. Once the state structure is stable,
parameters, gradients, tensor state, and tensor-valued group options are lifted
into an explicit recurrent graph when PyTorch can represent the update.

An optimizer whose Python behavior cannot be copied or represented as a graph
remains a bounded, profiled opaque task. This preserves generality without
claiming that unknown workspace or external device allocations are free. The
runtime executes lifted optimizer graphs under `torch.no_grad()`, matching the
ordinary `Optimizer.step()` mutation contract. ShadowSpill contains no `mlops`
import; an externally supplied optimized optimizer uses this same boundary.

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
