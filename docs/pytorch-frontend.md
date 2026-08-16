# PyTorch Frontend

ShadowSpill's PyTorch frontend owns capture and ordinary task dispatch. It does
not move model arithmetic into the neutral runtime.

## Runtime and public callables

Initialize one process-lifetime runtime before planning:

```python
from shadowspill.memory import device, pinned_host
from shadowspill.pytorch import Runtime, relocate_model_state

runtime = Runtime(
    pools={
        "device": device(physical_capacity=24 << 30),
        "spill": pinned_host(capacity=64 << 30),
    }
)

model = relocate_model_state(
    model,
    runtime=runtime,
    pool="spill",
    release_source=True,
)
```

`Runtime` loads the adapter and the planning path loads the simulator and
planner from the installed ShadowSpill package. Editable development builds use
the configured project build directory. Binary selection is not controlled by
environment variables and does not patch Python or PyTorch objects.

`relocate_model_state()` is an explicit prerequisite to planning. It returns a
distinct model hierarchy whose registered tensors point directly into the
selected runtime pool. It preserves module topology, parameter ties, buffer
views, values, layout, and model mode without copying the registered payload
through a second anonymous model allocation. `release_source=True` is the
default and canonical behavior: ShadowSpill retains no source reference, and
assigning the result back to `model` lets Python release the old model when no
other caller reference exists. Set `release_source=False` only when the caller
intentionally needs the original model and its anonymous CPU storage to remain
available independently.

`plan_forward(model, example_inputs=..., runtime=runtime, execution="device",
spill="spill", partition="auto")` constructs forward-only execution.
`plan_step(...)` constructs accumulated training. Planning validates existing
runtime object bindings and never copies or releases model state. Planning must
occur before an incompatible accelerator allocator initializes the process.
The callable gives the relocated registrations accelerator identity while it
owns execution and restores their spill-backed CPU identity on `close()`.

`execution_budget=None` and an explicit value equal to the runtime's complete
execution-device physical cap both select the full derived execution pool.
This lets the conventional `Runtime(...16 GiB...)` plus
`plan_step(...execution_budget=16 GiB...)` spelling charge context/provider
headroom exactly once. A value no larger than the derived pool capacity is a
logical per-plan limit. `spill_budget` defaults to and may reduce its selected
pool capacity. `execution_device=None` uses PyTorch's current accelerator
device; an explicit ordinal or `torch.device` selects it. The resolved device
must equal the execution pool's device.

`planning_cachedir` selects one shared, inspectable artifact store. Export is
rerun on every planning call, while graph pairs, compiler artifacts, task
profiles, PressureFit selections, and resolved plans are content-addressed
within that store. See [Planning Cache](planning-cache.md) for its layout,
precise identities, freshness controls, and custom-kernel invalidation rules.

`profiling_metadata` is an optional JSON-compatible description of
value-dependent performance characteristics not represented by tensor
geometry. Training accepts one entry per example microbatch; forward planning
accepts one entry. It only enters profiling/cache identity and PlanReport
diagnostics. It is never passed to the model or runtime. For example, it keeps
packed `[4096, D]` workloads representing one 4,096-token sequence distinct
from eight 512-token sequences while still reusing their common compiled ABI.

Every invocation validates the complete tensor geometry, storage alias
relationships, and static metadata before writing persistent input slots.
PressureFit's initial placement and exact ordered actions are submitted to the
neutral runtime without trigger changes. Each compiled stage remains an
ordinary PyTorch call wrapped by `before_task` and `after_task`. Public output
pytrees are reconstructed from Export's user-output positions, and their live
slab allocations transfer to ordinary caller ownership without a copy.

Planning is transactional. `plan_step()` and `plan_forward()` either return a
fully admitted callable or raise after rolling back provisional task records,
object IDs, materialization, and allocator state. Failures retain their
specific source:

- `CaptureError` identifies graphs that strict Export/AOT cannot represent and
  chains the exact PyTorch exception as its cause;
- `CompilationError` identifies Inductor construction or executable-storage
  contract failures and retains the structural ABI, task kind, operators, and
  exact compiler exception as its cause;
- `ProfilingError` identifies isolated task warmup, measurement, provider-kernel,
  or allocation-telemetry failures and retains the same structural context and
  original cause;
- `AdmissionError` reports invalid or insufficient execution/spill budgets,
  workspace/provider headroom, physical replay, and pool sealing failures;
- `PlanInfeasibleError`, a specialized `AdmissionError`, reports the exact
  capacity/residency constraint that prevents any valid schedule and retains
  its machine-readable fields. Irreducible task-capacity failures are detected
  by feasibility preflight before PressureFit candidate search begins;
- `PlanSearchExhaustedError` reports that a bounded planner search stopped
  before proving feasibility or infeasibility; it is deliberately not an
  `AdmissionError`;
- `PlanningError` remains the common planning base and directly reports
  non-resource signature and optimizer-contract errors; and
- `RuntimeExecutionError` carries structured allocator diagnostics for
  out-of-memory/no-progress during profiling or execution.

The private PyTorch adapter fails a rejected nonzero allocation before a tensor
can escape from `CUDAPluggableAllocator`.  The neutral C runtime first latches
complete structured diagnostics; its C++ callback wrapper then raises inside
PyTorch's allocator call because PyTorch does not itself validate a null
pointer.  The enclosing task boundary translates that exception into the
latched OOM, no-progress, envelope, or allocation-ABI error before an opaque
operator can launch a kernel against invalid storage.
The planning wrappers preserve exact lower-level exceptions through
`__cause__`. During execution of a returned callable, illegal memory accesses
and other genuine provider failures retain their original exception rather
than being mislabeled as profiling failures. A recoverable planning-time
no-progress OOM is quiesced,
its latch is cleared, and provisional state is rolled back; the failed
operation is never resumed or silently retried. `Runtime.last_failure` retains
the latest structured allocator failure for inspection.

`state_dict()` and `load_state_dict()` are explicitly synchronizing and use
ordinary CPU tensors with the original model names. A training state mapping
contains model, optimizer, and logical-step state; a forward-only mapping is
directly a model state dict.
`close()` is synchronizing, idempotent, preserves the relocated model's
Parameter objects and ties, restores spill-authoritative bytes, and unregisters
plan objects. It does not release persistent model state. Use
`externalize_model_state(model, runtime=runtime, release_runtime=True)` to copy
state into ordinary CPU allocations and release the runtime objects before
closing the runtime. Caller-retained outputs remain valid after callable close
because they are no longer plan-owned.

If execution raises, callable cleanup is attempted before the original error
is re-raised. Cleanup failures become exception notes and cannot mask the first
cause. The process-global PyTorch allocator cannot be uninstalled safely, so a
C `atexit` handler performs final native teardown: it rejects new callbacks,
stops and joins the worker, closes every memory pool, unregisters pinned spill
memory, and frees the device slab. Explicit callable and runtime close remain
the normal lifecycle.

Registered buffers marked `persistent=False` remain runtime-owned and are
restored on close, but they are omitted from checkpoints and are not required
by `load_state_dict()`, matching ordinary PyTorch. Loading a checkpoint first
preserves the current bytes of every non-persistent registration and then
overwrites only checkpoint-persistent entries.

The public `plan_step()` callable composes every fixed microbatch position into
forward, objective, backward, and gradient-accumulation tasks followed by one
optimizer update. It preserves the relocated model and optimizer checkpoint
schema and restores spill-backed CPU storage on deterministic close.

## Checkpoints and ordinary PyTorch restoration

Checkpoint methods belong to the planned callable. They are explicitly
synchronizing and return ordinary CPU tensors that do not reference CUDA or
runtime spill storage. Snapshot construction and its memory copy finish before
`state_dict()` returns. A training checkpoint has exactly this structure:

```python
checkpoint = train_step.state_dict()

assert set(checkpoint) == {"model", "optimizer", "step"}
torch.save(checkpoint, checkpoint_path)
```

### Overlapping filesystem serialization with training

After the synchronous `state_dict()` call has finished creating its isolated
anonymous-memory copy, serialization from that copy to the filesystem may run
in a background thread while later training steps use and mutate the original
runtime-owned state:

```python
from concurrent.futures import ThreadPoolExecutor

# This call synchronizes and finishes the state copy before returning.
checkpoint = train_step.state_dict()

with ThreadPoolExecutor(max_workers=1) as checkpoint_io:
    save_future = checkpoint_io.submit(
        torch.save,
        checkpoint,
        checkpoint_path,
    )

    # The future retains checkpoint while torch.save() reads it. These steps
    # mutate the original runtime state, not the isolated checkpoint copy.
    for microbatches in subsequent_steps:
        train_step(microbatches)

    save_future.result()  # Propagate any filesystem/serialization failure.

del checkpoint  # Permit the ordinary CPU snapshot allocations to be reclaimed.
```

The overlap begins only after `state_dict()` returns. The runtime-to-anonymous
memory synchronization and copy are entirely synchronous; only the subsequent
anonymous-memory-to-filesystem serialization is asynchronous with respect to
continued training. Snapshot tensors occupy ordinary anonymous, pageable CPU
memory. That memory is not allocated from a ShadowSpill pool, is not included
in `spill_budget`, and is not reported by ShadowSpill memory telemetry. The
process can therefore use approximately one additional checkpoint's worth of
system RAM until serialization completes and all references to the checkpoint
are released.

Restore the complete state into an active ShadowSpill callable through the
callable, not through its temporarily runtime-owned model:

```python
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
train_step.load_state_dict(checkpoint)
```

The `model` member is itself a conventional PyTorch model state dict with the
original parameter and persistent-buffer names. A separately instantiated
matching model and optimizer can therefore consume it normally:

```python
restored_model = CustomModel(...)
restored_model.load_state_dict(checkpoint["model"], strict=True)

restored_optimizer = optimizer_factory(restored_model.parameters())
restored_optimizer.load_state_dict(checkpoint["optimizer"])
logical_step = checkpoint["step"]
```

Do not pass the complete three-key training checkpoint to
`model.load_state_dict()`; pass `checkpoint["model"]`. A forward-only
callable's `state_dict()` is already the model mapping and can be passed
directly to a matching ordinary model.

For direct loading into the relocated model object, first close its planned
callable and call
`externalize_model_state(model, runtime=runtime, release_runtime=True)`.
During active execution use `train_step.load_state_dict()` so ShadowSpill can
update authoritative versions and optimizer-plan state correctly.

The training checkpoint covers model state, optimizer state, and ShadowSpill's
logical step number. Application-owned RNG state, schedulers, gradient scalers,
and data-loader position must be saved separately when exact stochastic replay
requires them.

## Planning and execution diagnostics

Every successful `plan_step()` and `plan_forward()` returns a callable whose
`plan_report` is already complete. `plan_report.diagnostics` is mandatory and
does not require a later synchronization:

```python
train_step = plan_step(model, ..., runtime=runtime, execution="device", spill="spill")

planning = train_step.plan_report.diagnostics
selected = planning.task("execution_000123")
print(selected.chosen_graph_pair_variant)

for stage in planning.unique_stages:
    for pair in stage.graph_pairs:
        print(pair.variant, pair.forward.runtime_ns)
```

The planning diagnostic reports mutually exclusive phase intervals and an
explicit unattributed remainder; their sum equals its recorded total wall
time. It also reports structural profile and graph-pair cache work, a direct
task-to-unique-stage map, the candidate and chosen graph-pair variant for every
task, the task's profile and `profiling_metadata` identities, and every legal
graph-pair alternative for each deduplicated stage. It also lists every cache
artifact read, written, matched, or managed by the planning call.

`PlanReport` also records the selected pool names, effective capacities,
execution-device ordinal, and the complete immutable transfer-capability
matrix snapshot. `fetch_profile` and `evict_profile` expose the exact measured
routes consumed by PressureFit, including matrix generation, digest,
provenance, timestamp, latency, solo bandwidth, simultaneous bidirectional
bandwidth, sustained measurement windows, and the effective bandwidth used by
planning.

Each forward and backward graph profile contains its calibrated runtime and
samples; input, mutation, and output object/alias identities and byte sizes;
logical and allocation-byte totals; anonymous workspace requested/charged
peaks and live extent multiset; persistent provider extents; and the complete
task-local allocation/free timeline. These records are deterministic logical
evidence and contain no framework pointer or CUDA handle.

Detailed real-step tracing is separate and opt-in. An individual call requests
a trace without changing the normal behavior of other calls:

```python
result = train_step(microbatches, runtime_trace=True)
# Explicitly synchronizing: waits for callbacks/events and copies trace data.
step_diagnostics = result.diagnostics.result()
```

Provider annotations are an independent switch and are also off by default:

```python
result = train_step(
    microbatches,
    runtime_trace=True,
    profiler_annotations=True,
)
```

`runtime_trace` controls `StepResult.diagnostics`, the seven task timestamps,
and the bounded allocator/runtime event logs. `profiler_annotations` controls
NVTX (or the active backend profiler) only. Either can be enabled without the
other. Provider annotations remain active long enough to include asynchronous
terminal transfers; the next unannotated call or `close()` drains that prior
work and disables them. Forward callables expose the same
`profiler_annotations=False` switch while continuing to return their ordinary
output pytree.

`StepResult` construction remains asynchronous. Starting another call before
resolving the preceding traced result is rejected because the bounded trace
buffers cannot be reused safely. The first `runtime_trace=True` call prepares
bounded CPU trace buffers and timing events before `trace_begin`; diagnostics report
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

`StepDiagnostics` is organized into `summary`, `timing`, `tasks`, `allocator`,
`transfers`, `runtime`, and `simulator_comparison`. `summary` directly
reconciles the profiled and real task-event sums, simulated and real inter-task
gaps, selected-task spans, terminal simulator tail, per-phase deltas, and trace
completeness. `simulator_comparison` is keyed by `execution_XXXXXX` and records
aligned simulator/real start, end, and duration values plus each delta, so
schedule drift can be localized to the first divergent execution boundary.
The transfer section provides the equivalent per-`fetch_XXXXXX` and
`evict_XXXXXX` comparison: simulator ready/start/end, real queue/reservation/
worker-dispatch/completion, duration, byte identity, and deltas. Transfer
times use the host-observed FIFO frontier and are labeled accordingly; initial
placement copies are counted separately because the recurrent simulator does
not model that deferred cyclic-startup work. The runtime section carries
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
does not differentiate that whole graph and then discard it. Metrics are
detached tensor outputs or preserved static pytree leaves; only the scalar
objective is differentiated.

Partitioning and differentiation are deliberately separate:

```text
captured Export graph
    -> partition_export() -> ordered Stage occurrences
    -> graph-pair capture -> one shared GraphPairPortfolio per structural ABI
    -> Program lowering   -> legal recomputation options per occurrence
```

A `Stage` is one contiguous partition occurrence in topological order. It owns
the FX subgraph, boundary sources, mutations, and public-output projection; it
does not own AOT graphs, recomputation choices, profiles, or cache behavior.
A `StageExample` attaches the representative values needed to determine one
concrete fixed-shape ABI. Static specialization may therefore change a
captured stage graph, but tensor contents do not define partition boundaries.

Graph-pair construction lives separately under `pytorch/graph_pairs/`. It
derives a structural ABI from normalized graph semantics, tensor geometry,
layouts, aliases, mutations, static arguments, and differentiated roots.
Equivalent repeated `Stage` occurrences reuse the same
`GraphPairPortfolio`, while occurrence-local initialized values are rebound
after lookup. Task ordinal, layer name, microbatch position, values, timing,
budgets, and physical allocation sizes do not enter that identity.

A portfolio is an ordered collection of labeled `GraphPairVariant` records.
Each record carries an option ID, AOT forward/backward pair, and—when it uses
the min-cut partitioner—the activation-memory budget that generated it. The
default-partitioner `save` choice therefore has no min-cut budget. The
historical `recompute` choice is PyTorch's runtime-optimized min-cut at an
explicitly fixed budget of `1.0`; this preserves established plans and avoids
silently inheriting ambient Functorch configuration. Lowering, persistence,
diagnostics, and `Program` construction already accept arbitrary future
min-cut portfolios such as `1.0, 0.75, 0.5, 0.25, 0.0` without changing
partitioning.

`partition="auto"` discovers outer containers with repeated sibling module
types and splits the functional Export graph at those child boundaries. Nested
repeated groups, such as experts inside one repeated transformer block, stay in
their owning outer block. Prologue operations join the first stage and epilogue
operations join the last. If there is no repeated structure, the complete
graph is one legal stage. Opaque operations and data dependencies remain
ordinary FX edges; partitioning does not rewrite their semantics.

Each distinct structural ABI receives its graph-pair portfolio and executable
profiles once. Profile keys canonicalize FX dataflow without placeholder,
node, layer, or task names. Tensor geometry, gradient requirements, static
arguments, operators, compiler/provider identity, Torch/CUDA version, and
device capability remain semantic. Thus identical interior layers share one
portfolio and profile, while a first layer that does not return an input
gradient or a last layer containing the objective correctly remains a
different ABI.

`partition="auto"` and `partition="whole"` are built in. Advanced callers may
pass a `PartitionPolicy` whose `assign_stages(graph_module, module)` method
returns one nonnegative integer label for every executable FX node. ShadowSpill
requires complete coverage and topologically contiguous labels, normalizes the
labels to dense stage IDs, and rejects policies that mutate the graph. Forward
and training planning use the same policy contract.

PyTorch 2.13 AOTAutograd may initialize CUDA provider state even when its model
and examples are FakeTensors. Public planning therefore installs ShadowSpill's
allocator before entering Export/AOT capture. The composable planning API
therefore requires resolved runtime memory before capture and enforces this
ordering without a hidden stateful coordinator.

Registered custom operators are accepted when their normal PyTorch contracts
are complete: schema, fake/meta implementation, alias and mutation declaration,
and—when differentiated—an autograd implementation. ShadowSpill does not
special-case operation libraries.

## Composable planning boundaries

`plan_forward()` and `plan_step()` are convenience compositions. Advanced
tools can call the same typed boundaries from `shadowspill.pytorch.planning`
when they already own the corresponding resolved `PlanMemory` and artifact
cache:

```python
from shadowspill.pytorch.planning import (
    PlanningCache,
    PlanningTimer,
    build_forward_program,
    capture_forward_graph,
    open_artifact_repositories,
    pressurefit_forward_program,
    profile_forward_tasks,
)

timer = PlanningTimer(verbose=True)
artifacts = open_artifact_repositories(PlanningCache.resolve(cache_directory))

captured = capture_forward_graph(
    model,
    example_inputs=example_inputs,
    memory=memory,
    partition="auto",
    profiling_metadata=profiling_metadata,
    artifact_cache=artifacts,
    timer=timer,
)
profiled = profile_forward_tasks(
    captured,
    artifact_cache=artifacts,
    timer=timer,
)
program = build_forward_program(
    captured,
    profiled,
    memory=memory,
    timer=timer,
)
selection = pressurefit_forward_program(
    program,
    artifact_cache=artifacts,
    timer=timer,
)
```

Training exposes the corresponding `capture_training_graphs()`,
`materialize_training_state()`, `profile_training_tasks()`,
`build_training_programs()`, `pressurefit_training_programs()`,
`compile_selected_training_tasks()`, and `admit_training_plan()` functions.
Every boundary returns an immutable artifact sufficient for the next one.
`rollback_training_materialization()` explicitly restores spill-backed CPU
bindings if an advanced caller stops after optimizer materialization.

PressureFit itself remains framework neutral. A caller that only wants to
sweep planner capacity starts from `planned.plan_report.program` and
`planned.plan_report.pressurefit_result`; it does not need a runtime allocation,
PyTorch capture, compilation, or profiling call.

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

Compilation and profiling are separate packages and artifact boundaries.
`pytorch.compilation` constructs process-local callables and reconciles
Inductor's physical output layout with the offline semantic contract.
`pytorch.profiling` owns representative values, value-sensitive metadata,
measurement policy, workspace telemetry, structural profile keys, and profile
cache records. Profiling consumes compiled artifacts; compilation does not own
profiling policy or profile persistence.

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
ABI. A warm profile cache launches no profiling kernels; executable
construction remains a separate compiler-cache concern because the returned
callable is process-local.

`PlanReport.diagnostics.phases` attributes this work without overlapping
clocks. `compiled_entrypoint_construction` covers compiler construction of
unique graph ABIs, `unique_stage_warmup_profiling` covers provider first-use
warmup, calibrated samples, and workspace auditing for profile-cache misses,
`cached_entrypoint_warmup` covers the warmup needed when a profile was loaded
but its process-local callable was newly built, and
`profile_cache_and_entrypoint_orchestration` is the measured remainder for
cache I/O and callable adoption. In particular, slow provider JIT or
autotuning encountered by the first warmup is reported as warmup/profiling,
not misidentified as repeated stage profiling.

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
retain spill copies. Caller inputs, intermediate activations, and returned
outputs have step-scoped identities and declared initial/final residency.

Each automatic stage becomes one canonical task with its structural profile,
dependencies, inputs, outputs, and compute resource. A separate private
entrypoint binding retains only the stage module target, compiled artifact, and
tensor leaf positions needed by the PyTorch executor. The resulting `Program`
is accepted directly by the standalone simulator and PressureFit; neither sees
PyTorch tensors, graph modules, pointers, or operation names.

The implementation mirrors those contracts directly.
`lowering/catalog.py`, `task_binding.py`, `profiles.py`, and `program.py`
are shared by both modes. The `lowering/forward/` package owns only forward
output ownership and stage emission. The `lowering/training/` package owns
only stage boundaries, graph-pair variants, gradients, accumulation, and
optimizer task composition. Each package exposes one short
partitioned-lowering orchestrator whose calls correspond directly to those
phases. The former unpartitioned training lowerer has been removed; production
and tests use the same partitioned contract.

Planning is likewise split by artifact boundary under `pytorch/planning`.
Cache resolution, capture, profiling, Program construction, PressureFit,
physical admission, and reporting are separate modules and functions. The
public convenience call is a short composition of those functions rather than
a monolithic builder with hidden mutable state.
