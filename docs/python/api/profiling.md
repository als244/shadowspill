# Profiling controls and measurement evidence

Profiling measures the exact finalized task implementations in a saved `Program`.
It does not rebuild the model or create execution/spill pools. Repeated layers
and microbatches share measurements when their task implementations and input
contracts are structurally identical. Save/recompute variants, gradient-creation
and accumulation forms, different shapes and different chunk implementations
can require distinct measurements.

```python
from shadowspill import (
    ProfilingOptions, TaskProfileCache, apply_profiles, measure_program,
)

timing = ProfilingOptions(
    conditioning_seconds=0.25,       # GPU-time floor before measuring each task
    conditioning_max_samples=32,    # Stop tiny-task conditioning sooner
    measurement_seconds=0.18,       # GPU-time floor across timing samples
    minimum_samples=3,
    maximum_samples=128,
)
cache = TaskProfileCache("/local-ssd/shadowspill-cache")
profiles = measure_program(program, options=timing, cache=cache)
profiled = apply_profiles(program, profiles)
profiles.save("profiles.ssp")
```

`ProfilingOptions` is immutable and does not import Torch or initialize CUDA.
`to_dict()` / `from_dict()` round-trip its complete JSON-compatible policy.
For only the two duration floors, `measure_program(warmup_seconds=...,
measure_seconds=...)` is a shortcut to the same policy. Do not combine these
shortcuts with `options`; conflicting specifications are rejected.

## Contents

- [Bounds and defaults](#bounds-and-defaults)
- [What actually runs](#what-actually-runs)
- [Cache reuse and artifacts](#cache-reuse-and-artifacts)
- [Training runner and quickstart](#training-runner-and-quickstart)
- [Task warmup versus sustained GPU operation](#task-warmup-versus-sustained-gpu-operation)
- [Task measurement](#task-measurement)
- [Profile overrides](#profile-overrides)

## Bounds and defaults

The following controls bound each unique task's conditioning and timing.
Time floors count CUDA-event device time, not CPU preparation time.

| `ProfilingOptions` field | Default | Meaning |
|---|---:|---|
| `conditioning_seconds` | `0.25` | Accumulated device-time target for the exact task before sampling. Zero disables this floor. |
| `conditioning_max_samples` | `32` | Maximum conditioning invocations, even if the time floor is unmet. |
| `conditioning_wall_seconds` | `3.0` | Wall-time limit on conditioning repetitions. |
| `minimum_samples` | `3` | Required timing samples, including for long-running tasks. |
| `maximum_samples` | `128` | Maximum timing samples, including when timings remain variable. Must be at least `minimum_samples`. |
| `measurement_seconds` | `0.18` | Accumulated device-time target across timing samples. Zero disables this floor. |
| `measurement_wall_seconds` | `2.0` | Wall-time limit on further timing samples after the minimum count. |
| `relative_mad_threshold` | `0.03` | Maximum median absolute deviation divided by the median for stable samples. |
| `half_drift_threshold` | `0.03` | Maximum difference between the first- and second-half medians, divided by the overall median. |

All count limits are positive integers. Durations and thresholds are finite,
nonnegative numbers. A sample already in progress is never interrupted.
The minimum timing count takes precedence over the wall limit: three four-second
tasks still take twelve seconds. Tiny tasks stop at their count limits instead
of running thousands of times to fill a time window.

The time floor is a target subject to these limits. For example,
`measurement_seconds=1.0, maximum_samples=256` reaches one second only if an
invocation averages at least 3.90625 ms. A 1 ms task stops at roughly 256 ms of
device samples, with `target_met=false` and `limit="samples"`. To retain that
cutoff with a two-second target, set `maximum_samples=512`. The time target and
count cap are independent explicit controls; changing one does not alter the other.

Timing normally finishes after the minimum count, device-time floor and both
stability checks pass. A wall or count limit can stop it sooner. The resulting
profile retains that fact; a time floor is not silently reported as satisfied.
Reaching a count limit with stable samples is distinct from unstable timings.
Profiling does not generally fail on timing noise; the evidence lets callers
decide whether to remeasure.

## What actually runs

For each uncached task, profiling performs one cold allocation observation,
three warmed allocation observations, allocation and alias validation, then
bounded task conditioning immediately followed by the timing window. No
allocation decoding, validation, input preparation or telemetry polling occurs
between conditioning and timing. Ordinary host dispatch and distributed stop
coordination still take time; this is not a promise of zero inter-call latency.
The cold/warmed allocation observations are separate from the configurable
timing counts. Their
four executions remain necessary memory-contract checks. Constructing authentic
control inputs or backward saved state can require other forward executions.

Timing uses timing-enabled CUDA events around one exact task invocation and
waits for its end event. The median device interval is the task duration.
Allocation decoding, validation and layout reconciliation occur outside these
timed intervals. The interval can include host-submission gaps inside a task;
it is not a sum of individual kernel durations. Normal executor dependency
events remain untimed; detailed runtime tracing is independently opt-in.

All distributed participants must specify identical policies. They agree on
repetition decisions, including count and wall limits, so a worker cannot stop
while another enters an unmatched collective. The minimum sample count still
applies. Communication calibration uses the same policy type:

```python
from shadowspill import measure_communication

communication = measure_communication(program=program, options=timing)
```

Pinned-host copy and collective probes use the bounded timing windows above.
External spill backends use native calibration with three timing windows:
sample limits apply per native window, and time-based counts are estimated from
a pilot, including the requested wall budgets. A native window cannot be
interrupted. Its metadata records the chosen counts and observations.
Compute/collective-overlap calibration remains a separate `measure_overlap`
phase with its documented adaptive `trials` control; this policy does not
silently change overlap trials or kernel selection.

## Cache reuse and artifacts

Compatible `TaskProfileCache` hits skip task conditioning, allocation observation
and timing. Resource restoration and candidate-wide fixed-memory accounting
still run. Reuse is keyed by the finalized implementation, input/control
lineage, environment and effective profiling policy, including count caps.
More recorded timing samples can satisfy a smaller minimum. Explicit task
inputs disable cross-program task-cache reuse.

`Profiles.metadata["profiling_options"]` records the effective session policy.
Each measured `TaskProfile.metadata` also contains:

- `profiling_options`, `samples_seconds`, `timing_relative_mad`,
  `timing_half_drift`, and `timing_unstable`;
- `conditioning` and `sampling` windows, with `gpu_ns`, `wall_ns`, `iterations`,
  `target_met`, and `limit` (`null`, `"samples"`, or `"wall"`);
- phase wall times for cold observation, warmed observations, validation,
  conditioning, timing and layout, plus allocation-observation timing details.

A stable count-limited window can have `target_met=false` and
`timing_unstable=false`. A noisy count-limited window keeps
`timing_unstable=true`. A wall-limited window with an unmet time floor also
remains flagged. Read the counts and stop reason together; no single flag
certifies that profiling matches sustained training conditions.

See [measurement artifacts and cache modes](artifacts.md#profiles-and-taskprofile)
for refresh/require behavior and physical memory admission.

## Training runner and quickstart

The shared Python training API accepts the same policy as `config["profiling"]`:

```python
from shadowspill import ProfilingOptions
from workloads.recipes import train

config = {
    "run_dir": "runs/experiment",
    # Model, data and search choices belong in the rest of this configuration.
    "profiling": ProfilingOptions(
        conditioning_max_samples=32,
        maximum_samples=64,
    ),
}
train(model, config)
```

The runner also accepts a JSON object of policy fields, filling omitted fields
from the same defaults. It serializes Python policies into the run configuration.
Changing profiling controls is allowed in `phase="profile"`; completed profiles
with a different effective policy are remeasured, and new planning must use the
new profiled artifacts. `refresh_profiles=True` also requests fresh measurements.
The generic `benchmarking.quickstart.run(..., profiling=timing)` exposes the
same policy. The two duration shortcuts remain mutually exclusive with an
explicit policy throughout these paths.

## Task warmup versus sustained GPU operation

The default warms each exact task and bounds measurement cost. It does **not**
promise thermal saturation or fixed clocks. Clock boost can settle quickly
while temperature-dependent task duration continues drifting for tens of
seconds. Input construction, allocation analysis and cache hits can also leave
gaps in GPU work. Increasing every task's time floor cannot guarantee that the
whole profiling session matches a long training run.

For high-fidelity performance studies, record clocks, temperature and power
alongside profiling and compare representative sustained execution. Condition
the device once with representative work when needed, amortizing that cost
over a profiling campaign; do not add a long arbitrary heater before every
tiny task. The appropriate duration depends on hardware, initial temperature,
power limits and workload. There is no universally sufficient minimum number
of seconds, and these APIs do not silently lock clocks or change power limits.
An optional pilot training trace can validate predictions under the actual
compute/transfer mixture. Judge refinements on subsequent steps or other
budgets, rather than fitting and evaluating the same trace.

## Task measurement

`measure_program(program)` restores the finalized tasks on a representative GPU
and returns a reusable Profiles collection. It uses PyTorch's ordinary native
allocator with bounded allocation observation, not ShadowSpill pools. By default
it measures all unique task forms; `stages=[...]` selects stage indices and omits
standalone optimizer work. Backward inputs include saved values from the exact
forward form. An `inputs` mapping can supply flat task arguments by profile key
when automatic values are unsuitable or control semantics are unknown.
Build and measurement preserve the caller's default CPU and target-device random
generator states. Execution consumes random values normally.

Structural capture does not read model payloads to compute inter-stage controls.
When an integer/boolean input is produced by another stage, build-time provider
finalization and profiling obtain it by replaying the compiled producer with the
ordinary profiling inputs. This preparation runs outside task timing and also
works when profiling a stage subset. Caller overrides apply to producer inputs
as well. Only the needed dependency slice is materialized, with CPU snapshots
between producers; generated index arrays are not embedded in Program archives.
Unknown root controls still require explicit values. CPU task outputs retain
their dependencies but have zero GPU extent, and Executor holds their host values
until the last ordered consumer.

Measurement records durations, allocation lifetimes, output aliases and observed
fixed memory. Logical free requests establish nonoverlapping temporary lifetimes
on each allocation stream, independently of native addresses. Profiling compares
exact-size reuse with a largest-first interval assignment and retains the smaller
slot inventory. An interval slot reserves its largest request and can serve
different sizes at the same address; returned roots and persistent state remain
separate in that assignment. Different allocation streams retain separate slots.
The saved allocation contract describes exact requests and ownership; admission
reserves slot capacities, and execution binds each request within that capacity.

`external_headroom_bytes` (512 MiB by default) additionally reserves
worker headroom in that fixed-memory total; Executor inherits this saved setting.
This is conservative: profiling observes loaded library/context memory, while
the worker reserves headroom above its startup baseline. Planning therefore
includes the reservation before fitting the task arena into the total GPU budget.
The build/target setup must match. It does not compile or change
task definitions. Provider resources that cannot be observed completely
remain an execution-qualification concern; they are not silently treated as zero.

Workspace totals permit logical simulation. They do **not** establish allocation
order, storage handoffs, charged alignment, persistent provider allocations or
the task code required for execution. Built Programs contain packaged code
for fresh workers; memory evidence must describe those exact implementations.
With complete observed allocation evidence, planning also constructs a physical
`memory_layout` entirely on CPU. It validates the saved traces against compiler
storage contracts, allocates ranges, and includes dependencies needed for safe
range reuse in the simulated timeline. Layout and admission facts travel in the
Plan archive, with identities checked against its program and schedule.

Estimated memory, missing code, or incomplete allocation evidence keep a Plan
simulation-only, as reported by `simulation_only_reasons`. Inconsistent evidence
is an error. Physical placement is necessary for execution but does not replace
the worker's dependency, provider-resource and available-memory checks. Executor
validates those requirements before using its admitted layout.


## Profile overrides

```python
from shadowspill import TaskProfile

overrides = Profiles(
    setup=profiles.setup,
    tasks={profile_key: TaskProfile(duration_seconds=0.0072)},
)
profiles = profiles.merge(overrides)
```

Later supplied values overwrite earlier values. Omitted fields are preserved,
so this timing override preserves existing memory information. Supplied values
without classification metadata are estimates. Task compatibility and setup
mismatches are errors. Changes to editable Profiles never alter an already
created ProfiledProgram or Plan.
