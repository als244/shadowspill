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
