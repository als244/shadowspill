# Interpreting StepResult diagnostics

A normal `PlannedTrainStep` call returns a `StepResult` without detailed
runtime evidence. Pass `runtime_trace=True` for one debugging or qualification
step:

```python
result = train_step(inputs, runtime_trace=True)
assert result.diagnostics is not None

diagnostics = result.diagnostics.result()
```

`DiagnosticsHandle.result()` (also available as `wait()`) resolves once and
may synchronize recorded events. Resolve a traced step before launching
another traced step. `runtime_trace=False` is the default and performs no
native trace-buffer appends.

`profiler_annotations=True` is independent. It emits backend profiler ranges
for tools such as NSYS but does not create `StepDiagnostics`.

## What StepResult contains

| Field | Meaning |
|---|---|
| `objectives` | Detached scalar objective tensor for each accumulation round. |
| `metrics` | Reconstructed nondifferentiated objective metrics for each round. |
| `step_number` | Completed optimizer-step count. |
| `diagnostics` | `None`, or a deferred `DiagnosticsHandle` for a traced call. |

The resolved `StepDiagnostics` has seven main views:

| View | Purpose |
|---|---|
| `summary` | Compact simulator-versus-runtime timing reconciliation. |
| `tasks` | Per-execution boundary timestamps and host subphase costs. |
| `simulator_comparison` | Simulated versus real start/end/duration per task. |
| `transfers` | Selected actions, real transfer events, totals, and per-transfer comparison. |
| `allocator` | Ordered allocation/free events and terminal pool geometry. |
| `runtime` | Runtime counters, queue state, failure counts, and trace integrity. |
| `timing` | Call-level host/device phase totals and the same task mapping. |

Task dictionaries are keyed by chronological `execution_XXXXXX`, matching
`PlanReport.diagnostics.tasks`.

## Start with the summary

```python
summary = diagnostics.summary

print("profiled tasks", summary.profiled_task_seconds)
print("real task events", summary.real_task_event_seconds)
print("simulated idle", summary.simulated_inter_task_idle_seconds)
print("real idle", summary.real_inter_task_idle_seconds)
print("simulated span", summary.simulated_selected_span_seconds)
print("real span", summary.real_selected_span_seconds)
print("simulator makespan", summary.simulator_makespan_seconds)
print("terminal tail", summary.simulator_terminal_tail_seconds)
print("complete", summary.trace_complete)
```

The summary decomposes the selected-task span approximately as:

```text
selected-task span = sum of selected task-event durations
                   + gaps between selected task events
```

The fields mean:

| Field | Interpretation |
|---|---|
| `profiled_task_seconds` | Sum of isolated task profiles selected by the plan. |
| `real_task_event_seconds` | Sum of compute-stream intervals measured in the real step. |
| `task_event_delta_seconds` | Real task-event sum minus profiled sum. |
| `simulated_inter_task_idle_seconds` | Idle between simulated selected task intervals. |
| `real_inter_task_idle_seconds` | Idle between real selected task events. |
| `inter_task_idle_delta_seconds` | Real idle sum minus simulated idle sum. |
| `real_inter_task_readiness_wait_seconds` | Idle spent waiting for a task's inputs to be resident. |
| `real_inter_task_exposed_overhead_seconds` | The rest of that idle: the frontend had not reached the next task. |
| `simulated_inter_task_readiness_wait_seconds` | The modeled counterpart of the readiness wait. |
| `real_initial_readiness_wait_seconds` | The first task's wait, which precedes the span. |
| `real_minimum_frontend_lead_seconds` | The smallest lead the frontend held over the stream. |
| `simulated_selected_span_seconds` | First selected task start through last selected task end in simulation. |
| `real_selected_span_seconds` | Same boundary using real compute-stream events. |
| `selected_span_delta_seconds` | Real selected span minus simulated selected span. |
| `simulator_makespan_seconds` | Complete simulated schedule, including modeled terminal work. |
| `simulator_terminal_tail_seconds` | Simulated work after the last selected compute task. |
| `phase_comparisons` | Profiled versus real task-event sum by semantic phase. |

`real_inter_task_idle_seconds` is exactly `real_inter_task_readiness_wait_seconds` plus
`real_inter_task_exposed_overhead_seconds`: the stream is either waiting on a task's
inputs, or it has nothing to run at all. Between one task ending and the next
computing, the compute stream travels to the next task's readiness marker and
then waits there until that task's inputs are resident; the first is what
dispatch costs and the second is what residency costs. Reading the total alone
invites the wrong conclusion, because the two move independently -- a schedule
that fetches later raises the wait while leaving dispatch untouched.

It is called exposed because the frontend does this work at every boundary,
and running ahead hides whatever the lead covers. The field is the shortfall,
not the cost of the work. Do not reach for `dispatch_total_seconds` as the
other half of that comparison: the host blocks inside the launch call once the
device queue is full, so most of it is time spent waiting for the device
rather than working -- `dispatch_invoke_seconds` tracks its own task's
`gpu_duration_seconds` at a median ratio of 0.94.

`real_inter_task_exposed_overhead_seconds` is zero wherever the frontend runs ahead of
the device, because a task it has already reached has its readiness marker on
the stream before the previous task ends. It is never *exactly* zero, because two
consecutive event records on a stream sit about half a microsecond apart no
matter what lies between them. Measured over one olmoe step that floor ran
from 0.24 microseconds at the lowest boundary to 1.22 at the upper quartile,
so read a microsecond or two as nothing and anything above it as real.

The distribution is what to read, not the total. Over one olmoe step, 71 of 89
boundaries came in near that floor and together contributed 67 microseconds,
while 18 boundaries contributed the remaining 28.9 milliseconds -- every one of them a backward-to-optimizer transition, where
the frontend had spent its lead. A rising median means the frontend is losing
its lead everywhere; a heavy tail means it is losing it somewhere specific,
and the boundary names say where.

`frontend_lead_seconds` is the other side of the same coin, and the two are
worth reading together: exposed overhead is what the lead failed to cover, so
it stays at the floor while the lead holds and appears the moment it reaches
zero. Both ends come from the same step origin -- the stream's marker is
measured from the origin event and the handoff from the trace beginning, which
the runtime records at the same point -- so no clock fitting is involved.
`real_minimum_frontend_lead_seconds` is the margin that was left over the
whole step.

The simulator places a task as soon as its dependencies are met, so it has no
notion of a stream travelling to a marker, and so cannot starve one. That
absence is the point of reporting the two separately: whatever
`real_inter_task_exposed_overhead_seconds` holds is cost the model does not represent.

`real_initial_readiness_wait_seconds` is the first task's wait. It ends where
the span begins, so no span-relative number here contains it, and on a step
that opens by fetching its parameters it can exceed everything that follows.
| `trace_complete` | False when any required trace source overflowed or was incomplete. |

Use the selected span for the warmed first-task-to-final-task comparison. Do
not compare an end-to-end host call that includes startup/cooldown with a
selected-task simulator interval without accounting for those boundaries.

## The seven task-boundary timestamps

Every `TaskExecutionTiming` contains exactly seven primary timestamps:

| Clock | Timestamp |
|---|---|
| Host `CLOCK_MONOTONIC` | `before_task_enter_timestamp_ns` |
| Host `CLOCK_MONOTONIC` | `before_task_exit_timestamp_ns` |
| Host `CLOCK_MONOTONIC` | `after_task_enter_timestamp_ns` |
| Host `CLOCK_MONOTONIC` | `after_task_exit_timestamp_ns` |
| Compute stream | `before_readiness_waits_timestamp_ns` |
| Compute stream | `before_task_compute_timestamp_ns` |
| Compute stream | `after_task_compute_timestamp_ns` |

The host and compute-stream timestamps are different clock domains. Compare
durations within a domain; use the precomputed aligned seconds fields for
cross-task and simulator comparisons.

The timeline is:

```text
host:    before_task enter ---- frontend/runtime preparation ---- exit
stream:                    marker before readiness waits
stream:                    wait on required fetch events
stream:                    marker before task compute
host:                                                dispatch callable
stream:                    compiled task kernels ---------------- end marker
host:                                                after_task enter
host:                                                output/runtime work -- exit
```

Useful derived values are:

- `frontend_lead_seconds`: how long after the frontend handed this task off
  the compute stream arrived at it, so how far ahead the frontend was;
- `dispatch_before_task_seconds`: complete frontend `before_task` wall time;
- `readiness_wait_seconds`: compute-stream delay introduced by unfinished
  readiness dependencies;
- `dispatch_after_task_seconds`: complete frontend `after_task` wall time;
- `dispatch_total_seconds`: before + dispatch + after host work;
- `gpu_start_seconds`, `gpu_end_seconds`, `gpu_duration_seconds`: the compute-
  stream task interval, aligned for cross-execution analysis.

A compute-stream readiness gap does not imply that the Python thread blocked.
Ordinary fetch readiness is expressed as a stream event wait. A large host
`before_task` interval instead points to lookup/rebinding/runtime work or a
capacity wait.

## Host boundary breakdown

`before_task` includes:

| Field | Work |
|---|---|
| `dispatch_stream_resolution_seconds` | Resolve the current compute stream. |
| `dispatch_readiness_marker_seconds` | Record the pre-readiness timing marker. |
| `dispatch_runtime_before_task_seconds` | Neutral runtime acquisition/readiness publication. |
| `dispatch_input_lookup_seconds` | Resolve frontend tensor/object bindings. |
| `dispatch_storage_rebind_seconds` | Rebind changed PyTorch storages. |
| `dispatch_argument_assembly_seconds` | Assemble predecoded callable arguments. |
| `dispatch_rebind_seconds` | Aggregate rebinding path retained for qualification. |

The callable portion is `dispatch_invoke_seconds`.

`after_task` includes:

| Field | Work |
|---|---|
| `dispatch_output_flatten_seconds` | Flatten the callable output pytree. |
| `dispatch_output_classification_seconds` | Match leaves with output/mutation contracts. |
| `dispatch_output_adoption_seconds` | Adopt returned allocations into logical objects. |
| `dispatch_output_state_publish_seconds` | Publish output state and bindings. |
| `dispatch_gradient_accumulation_seconds` | Accumulate explicit gradient contributions. |
| `dispatch_output_publish_seconds` | Publish public outputs. |
| `dispatch_dematerialize_seconds` | Drop frontend bindings selected for release. |
| `dispatch_postprocess_seconds` | Aggregate mode-specific postprocessing. |
| `dispatch_runtime_after_task_seconds` | Record completion and submit planned actions. |
| `dispatch_cleanup_seconds` | Remove released bindings and terminal state. |

The enclosing `dispatch_before_task_seconds` and `dispatch_after_task_seconds` are the
authoritative complete boundaries. Nested fields are explanatory and may
include aggregate views; do not assume every named subfield is additive.

To find expensive graph-task boundaries without bias from many short optimizer
tasks:

```python
rows = sorted(
    diagnostics.tasks.values(),
    key=lambda task: task.dispatch_before_task_seconds + task.dispatch_after_task_seconds,
    reverse=True,
)

for task in rows[:10]:
    print(
        task.execution_task_id,
        task.semantic_name,
        task.phase,
        task.dispatch_before_task_seconds,
        task.dispatch_after_task_seconds,
    )
```

## Task-by-task simulator comparison

`diagnostics.simulator_comparison[execution_task_id]` contains:

- simulated and real start;
- simulated and real end;
- start and end deltas;
- expected isolated profile duration;
- observed real task-event duration;
- duration delta.

```python
comparison = diagnostics.simulator_comparison["execution_000017"]
print(comparison.start_delta_seconds)
print(comparison.duration_delta_seconds)
print(comparison.end_delta_seconds)
```

Interpret the deltas together:

| Pattern | Likely area |
|---|---|
| Duration delta is large, start delta is small | Task profile, allocator callbacks inside the callable, provider behavior, or kernel variation. |
| Start delta grows while duration deltas stay small | Readiness, dispatch gaps, or earlier schedule/transfer drift. |
| Start delta grows and later shrinks | Simulator conservatism or recovered overlap. |
| All tasks shift by a constant | Startup alignment difference rather than per-task error. |

## Transfer diagnostics

`diagnostics.transfers` separates scheduled actions from completed real copy
counters. The aggregate fields are fetch/evict counts and bytes plus initial
fetch traffic, which is reported separately from in-step actions.

Each `SimulatorTransferComparison` is keyed by a stable transfer ID and
contains:

| Field group | Meaning |
|---|---|
| Identity | direction, sequence, trigger task, execution task, alias group, bytes |
| Simulation | ready, start, end, and duration |
| Runtime queueing | real queued and destination-reserved time when available |
| Runtime worker | dispatch and completion timestamp/frontier duration |
| Deltas | simulated-versus-real start, end, and duration |

The real interval is the host worker's observed transfer frontier, aligned to
the first scheduled transfer. It is not a backend-event bandwidth
microbenchmark.
Use an NSYS trace when exact provider API nesting or wire overlap is required.

Check action identity and byte totals before interpreting timing. A mismatch
there is a schedule/runtime fidelity failure, not merely a calibration error.

## Allocator diagnostics

`AllocatorTrace.events` is the ordered allocation/free ledger. Every event
includes sequence, task ID, allocation ID, generation, requested and charged
bytes, slab offset, kind, and category.

The terminal summary reports:

- live allocation and allocated-byte counts before/after the step;
- peak allocated bytes;
- free bytes and free-prefix bytes;
- largest free range and external fragmentation;
- blocked allocator count;
- trace overflow.

`requested_bytes` is the caller request. `charged_bytes` is the aligned range
owned by the pool. Compare requested totals when studying framework behavior
and charged totals when studying physical capacity.

Growth in live allocations or allocated bytes over repeated equivalent steps
requires explanation. A small `largest_free_range_bytes_after` despite large
`free_bytes_after` indicates fragmentation. A blocked allocator remaining at
step end is not a healthy steady state.

## Runtime counters and trace integrity

`RuntimeTrace` reports:

| Field | Meaning |
|---|---|
| `wait_events_inserted` | Stream dependencies inserted for unfinished inputs/reuse. |
| `allocation_requests` | All allocator callbacks, including zero-byte requests. |
| `zero_byte_allocation_requests` | Zero-byte callbacks, reported separately because they have no material lease/free lifecycle. |
| `materialized_allocation_requests` | Nonzero requests that require memory. |
| `free_requests` | Logical free callbacks. |
| `record_stream_callbacks` | Additional stream-use declarations. |
| `event_queries` | Worker completion-head queries. |
| `queued_actions_after` | Actions still queued when the trace closes. |
| `pending_retirements_after` | Allocations still causally protected. |
| `callback_failures_after` | Allocator callback failures observed by the runtime. |
| `events` | Raw bounded runtime-event records. |

Trace storage is preallocated and bounded. The report records both configured
capacities and `event_overflow`/`allocation_event_overflow`. Treat any overflow
as incomplete evidence. The training frontend currently prepares 1,000,000
runtime-event slots and 1,000,000 allocation-event slots for a traced call;
the recorded capacities, rather than that default, are authoritative for a
specific result. `summary.trace_complete` provides the combined verdict.

## Exporting diagnostics

`StepDiagnostics.as_dict()` returns a JSON-friendly value with schema
`shadowspill.step_diagnostics/v2`:

```python
import json

with open("step-diagnostics.json", "w", encoding="utf-8") as stream:
    json.dump(diagnostics.as_dict(), stream, indent=2, sort_keys=True)
```

The dictionary is an observation of one real call. It is not an execution
plan and has no `from_json()` planning constructor.

## Common investigations

| Symptom | Look here first |
|---|---|
| Real span is slower than simulated | Summary task-event delta versus inter-task-gap delta. |
| One task is slower than profiled | Per-task duration comparison, then allocator events within that task. |
| Device execution has gaps between graph tasks | Real task start/end sequence, readiness gaps, and complete host boundary costs. |
| Fetch appears late | Per-transfer simulated/real start, queue/reservation time, worker dispatch, and prior FIFO transfers. |
| Memory unexpectedly fills | Charged allocator events, peak bytes, largest free range, pending retirements, and action destinations. |
| A step appears to leak | Compare before/after live bytes and repeat across steps; include terminal caller-owned outputs in the interpretation. |
| Trace looks truncated | Both overflow flags, event capacities, and `summary.trace_complete`. |
| Runtime and plan identities are unclear | Join `execution_task_id` with the [PlanReport guide](plan-report.md). |
