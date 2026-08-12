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

## Planning and execution diagnostics

Every successful `plan()` and `forward_pass()` returns a callable whose
`plan_report` is already complete. `plan_report.diagnostics` is mandatory and
does not require a later synchronization:

```python
train_step = plan(model, ...)

planning = train_step.plan_report.diagnostics
selected = planning.task("task_000123")
print(selected.chosen_graph_pair_variant)

for stage in planning.unique_stages:
    for pair in stage.graph_pairs:
        print(pair.variant, pair.forward.runtime_ns)
```

The planning diagnostic reports mutually exclusive phase intervals and an
explicit unattributed remainder; their sum equals its recorded total wall
time. It also reports structural profile and graph-pair cache work, a direct
task-to-unique-stage map, the candidate and chosen graph-pair variant for every
task, and every legal graph-pair alternative for each deduplicated stage.

Each forward and backward graph profile contains its calibrated runtime and
samples; input, mutation, and output object/alias identities and byte sizes;
logical and allocation-byte totals; anonymous workspace requested/charged
peaks and live extent multiset; persistent provider extents; and the complete
task-local allocation/free timeline. These records are deterministic logical
evidence and contain no framework pointer or CUDA handle.

Detailed real-step tracing is separate and opt-in. An individual call requests
a trace without changing the normal behavior of other calls:

```python
result = train_step(microbatches, trace=True)  # trace=False by default
# Explicitly synchronizing: waits for callbacks/events and copies trace data.
step_diagnostics = result.diagnostics.result()
```

`StepResult` construction remains asynchronous. Starting another call before
resolving the preceding traced result is rejected because the bounded trace
buffers cannot be reused safely. The first `trace=True` call prepares bounded
CPU trace buffers and timing events before `trace_begin`; diagnostics report
that one-time setup separately, and later traces reuse the buffers. Untraced
calls allocate no trace resources. Each task record includes its expected
profiled wall time, a CUDA-event duration cross-check, allocator/transfer
evidence, and exactly seven `CLOCK_MONOTONIC` boundary timestamps:

1. host `before_task.enter`;
2. host `before_task.exit`;
3. host `after_task.enter`;
4. host `after_task.exit`;
5. compute-stream `before_readiness_waits`;
6. compute-stream `before_task_compute`;
7. compute-stream `after_task_compute`.

Every plan and task trace retains the stable canonical `task_id` used by the
schedule and caches. Selected tasks additionally expose a dense
`execution_ordinal`, an `execution_XXXXXX` identifier, and a semantic name such
as `microbatch_0000.stage_0007.backward.recompute`. These execution identities
follow actual chronological order; unselected save/recompute alternatives
have no execution ordinal.

The three stream-ordered timestamps use preallocated reusable CUDA events.
They are resolved only by the explicitly synchronizing diagnostics handle;
ordinary execution launches no timing events. An earlier host-callback
implementation was rejected because callbacks serialized fine-grained stream
work and materially perturbed the measured schedule.

`StepDiagnostics` is organized into `timing`, `tasks`, `allocator`,
`transfers`, `runtime`, and `simulator_comparison`. The runtime section carries
the complete ordered neutral-C event stream and overflow/capacity summary. The
transfer section provides the schedule actions, completed byte/count deltas,
and a filtered queue/reservation/dispatch/completion view. Allocation records
remain a separate lifetime ledger. Allocator requests are split into total,
zero-byte, and materialized requests; zero-byte requests return `nullptr` and
are not expected to receive a free callback. The allocator section reports
live-allocation and allocated-byte state before and after the call, which are
the authoritative leak indicators. Resolving diagnostics drains the progress
service so terminal transfers are included; this synchronization happens only
because the caller explicitly requested and resolved a trace.

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

Training first exports the complete objective for semantic partitioning. It
does not differentiate that whole graph and then discard it. After partitioning,
each executable stage produces save-oriented and min-cut recomputation graph
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

Repeated stage occurrences also reuse AOT graph code before compilation. The
reuse key includes the canonical pre-AOT graph, guarded tensor geometry,
storage offsets, argument alias groups, static inputs, and differentiable output
roots. A reused graph pair is rebound to the occurrence's own FakeTensor
storages, and both forward and backward artifact digests must match the
representative. `PlanReport.captured_stage_count`, `aot_unique_stage_abis`, and
the AOT graph-pair cache hit/miss counts expose this reduction directly.

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
