# Executor interface: review proposal

Status: fourth of the six grouped reviews. Most lifecycle behavior is already
selected. This document assembles it into a concrete call sequence and identifies
the remaining return/binding details. No implementation changes are authorized
by this proposal alone.

## Common lifecycle

The following example uses the selected per-call result handle.
It assumes the Plan was built for two rank-local examples. Every worker receives
the same complete `rank_map`; exact constructor argument spelling remains
provisional.

```python
# Inside a fresh worker process, before CUDA initialization.
plan = Plan.load("training.plan.zip")

with Executor(plan, rank_map=rank_map, device=device) as executor:
    executor.initialize(fill_tensor)

    pending = executor.run(
        args=[(features0, targets0), (features1, targets1)],
        kwargs=[{"mask": mask0}, {"mask": mask1}],
        hyperparams={"lr": 2e-4},
    )
    # CPU work can proceed while this invocation finishes.
    outputs = pending.result()

    # Optional checkpoint helper supplies a bounded chunk consumer.
    executor.export_tensors(write_chunk)
```

`fill_tensor(request, destination)` fills the Executor-provided bounded region.
`write_chunk` consumes bounded exported tensor regions; the precise export
callback arguments still need to be specified. Neither helper assembles a full
model in anonymous host memory. The built optimizer must declare `lr` as a
runtime-changeable value for this example; the complete key/group schema is a
remaining detail.

Executor construction owns normal allocator, pool, logical-group, and provider
resource setup. Trainer uses this same interface. Loading the Plan is CPU work;
setup rejects simulation-only memory and incompatible execution requirements
before dispatch. Startup does not build, profile, or replan. Context-manager
exit drains outstanding work before releasing owned resources; it cannot undo
the process-wide allocator installation.

## Selected return interface

The user selected reuse of the existing `InvocationResult` concept, with `run`
returning a small handle and `result()` waiting for that invocation, including
required writebacks, and exposing its results. It propagates execution errors
and makes readiness explicit for CPU-visible results as well as GPU work. It
does not require callers to construct another configuration object.

```python
pending = executor.run(args=examples)
# Do CPU work.
outputs = pending.result()  # Wait for this invocation, then return its payload.
```

`executor.synchronize()` remains available to wait for outstanding executor work.
After that wait, resolving an already-completed handle should be immediate. The
handle keeps the agreed caller-owned output lifetime; it does not turn outputs
into borrowed storage. Retained results still consume accounted memory.

Returning output tensors/structures directly was not selected. This is a choice
for the whole-plan execution API; ordinary PyTorch GPU tensor operations can
consume results on a correctly ordered stream without a CPU wait. The selected
`result()` has a host-visible invocation-completion meaning, rather than merely
establishing a dependency on a consumer CUDA stream.

The selected payload structure below is distinct from readiness. Existing
objective support includes scalar loss and auxiliary metrics; the refactor must
not accidentally remove it. The handle does not authorize concurrent invocations, an unbounded pending
queue, or a device-wide wait.

## Selected result payload

The user selected preserving the native model output for a `build_forward`
Plan, and using a small named training-result record for
`build_forward_backward` and `build_step`. The handle has the same completion
interface in all three cases; the Plan's computation boundary determines its
payload.

```python
# Forward Plan: preserve the model's tensor/dict/tuple/nested output structure.
prediction = forward_executor.run(args=[(features,)]).result()

# Training Plan built for two examples.
result = training_executor.run(
    args=[(features0, targets0), (features1, targets1)],
).result()
result.losses[0]   # Scalar objective tensor from example slot 0.
result.losses[1]   # Scalar objective tensor from example slot 1.
result.metrics[0]  # Slot 0's auxiliary metrics, preserving their nested structure.
```

Losses/metrics are in stable example-slot order, regardless of execution order.
The proposal keeps raw detached tensor values and preserves dtype/device; it
does not automatically convert them to Python scalars or add a CPU transfer.
Metrics without an objective-supplied payload have a clear empty value, such as
`None`. Explicitly captured parameter observations remain supported; the
migration inventory must preserve that existing capability.

The low-level record does not silently average across examples or
ranks. Objective scaling defines differentiation semantics, and Trainer/callers
may perform deliberate metric aggregation. Output handoff remains part of memory
accounting. Gradients are persistent Executor-owned tensors, as already selected;
they are not automatically copied into the training result.

The forward example is the existing one-call `build_forward` scope. It does not
add multi-example forward execution or an automatic singleton output list.
Returning the objective's auxiliary outputs does not automatically return every
intermediate model output. The output schema is captured during build.

A common result wrapper for every Plan kind was not selected. The final
training record type name and execution-diagnostics access belong in the final
public API/migration review; avoid multiple overlapping result classes.

## Selected subsequent invocation admission

The user selected one active invocation per Executor initially. If a caller invokes
`run` while the previous invocation is still finishing, wait for that invocation
before staging the new invocation into reusable storage or dispatching its work.
The caller need not explicitly retrieve the previous outputs to make progress.

```python
first = executor.run(args=first_inputs)
prepare_next_batch()  # CPU work can overlap the first invocation.
second = executor.run(args=next_inputs)  # Waits for first if still unfinished.
first_outputs = first.result()          # Already completed; still valid.
second_outputs = second.result()
```

This is sequential execution with CPU work between dispatches; it does not
enqueue an unbounded sequence or assume that two copies of the workspace fit.
The next invocation must use the completed prior parameter/gradient/optimizer
values, with its own runtime hyperparameters. Waiting covers the prior
invocation's tracked work and writebacks, not all work on the device. If that
invocation failed, propagate its error before dispatching the next one.

An earlier completed handle may remain alive while the next invocation runs.
Its outputs remain caller-owned and require accounted storage. Completion of
compute does not free caller-held results. If retained outputs leave insufficient
capacity for the next call, report the admission failure; do not wait indefinitely
for a caller-held reference or silently overwrite its storage. Exact handoff and
capacity implementation still need validation.

Rejecting a new `run` solely because the caller has not explicitly waited with
`result()` or `synchronize()` was not selected. Overlapping invocations would
need a separately designed schedule and memory contract.

Current code has both patterns: callable `submit` rejects an unresolved prior
handle, while underlying forward/training invocation setup drains prior runtime
work before reuse. The refactor will expose the selected automatic-wait rule
through Executor.

## Already selected invocation controls

- Sequence entries identify fixed example slots; the saved plan chooses task
  order. `run(input_loader=get_example)` is an alternative to supplying those
  sequences. The loader returns `(args, kwargs)` for an example index and replay
  must see the same values.
- Hyperparameters are a plain per-call mapping rather than a persistent setter.
- One forward/backward Plan supports fresh gradients and accumulation into
  existing persistent gradients. The user selected fresh gradients by default
  (`accumulate_gradients=False`), with explicit `accumulate_gradients=True` to
  add to stored gradients from an earlier invocation. Accumulation across
  `num_examples` within one invocation is unchanged. Old reduced gradients must
  not be reduced a second time; the Plan covers both supported behaviors.
- Returned outputs survive later calls. Persistent parameters/gradients remain
  Executor-managed and are exported through a separate bounded interface.
- CPU data, output retention, provider resources, communication buffers, and
  in-flight lifetimes must be accounted for; ownership is not free capacity.

## Logical rank binding and remaining details

`logical_rank` identifies ownership in the saved Plan; it is not necessarily the
launcher rank or local CUDA device index. All workers need a consistent mapping
from launch participants to logical ranks, validated against the fixed plan
size and planning topology/capacity assumptions. Group membership and member
order must follow that mapping rather than accidentally following physical rank
order.

### Selected choice: complete mapping supplied to every worker

The user selected a complete launcher-worker-to-logical-rank mapping supplied
identically on every worker. Each Executor derives its logical assignment from
its launch identity and this map. `rank_map` is proposed spelling, illustrated
as an ordinary dictionary without a new mandatory binding class:

```python
# Same values in every worker process: launch rank -> logical rank.
rank_map = {0: 2, 1: 0, 2: 3, 3: 1}

with Executor(plan, rank_map=rank_map, device=f"cuda:{local_rank}") as executor:
    ...
```

Every worker still loads the same archive; launch worker 0 receives logical rank 2's
tensor ownership and group memberships. The launch supplies rendezvous
information and identity independently of this model assignment. There is no
automatic load-balancing or arbitrary change in topology hidden in binding.
An ordinary identity assignment is `rank_map = {r: r for r in range(world_size)}`.
Whether a convenience launcher supplies that common map automatically is a
separate launcher detail; this choice does not silently select an omitted-map
default for the direct Executor constructor.

Before GPU collectives run, coordinated validation must detect different maps
supplied by different workers, duplicated or missing logical ranks, differing
plan identities, and incompatible
topology/capacity assumptions. Missing workers need a bounded startup timeout,
not an indefinite wait. Implementation must preserve logical collective member
order even if a backend orders groups by physical process rank; binding cannot
consist only of permuting a Python list of members.

Independent per-worker logical-rank declarations were superseded by the user's
complete-map choice. The full map is authoritative; do not introduce another
independently configurable local logical rank that can contradict it.

Complete this review with the result payload, launch-context details, callback lifetime,
and invocation admission details. Reuse existing simple records where they fit;
do not add a separate Trainer-only execution path or a mandatory worker class.
Artifact compatibility metadata belongs in the next review; actual mismatches
must fail explicitly rather than silently recompiling into a different plan.

## Current implementation reference

`src/shadowspill/pytorch/invocation.py` already implements `InvocationResult`
with `result()` / `wait()` and remembered completion/failure. The public
callables' `submit` paths use it; their ordinary call and `synchronize` paths are
separate. The proposal would give the new Executor one common asynchronous call
shape. It is a design for the refactor, not a claim that `Executor.run` with this
signature exists today.
