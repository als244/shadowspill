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
another traced step. `runtime_trace=False` is the default, performs no
native trace-buffer appends, and records no timing events: none of the
measurement below runs in an untraced step.

`profiler_annotations=True` is independent. It emits backend profiler ranges
for tools such as NSYS but does not create `StepDiagnostics`.

## Structure

| Field | Meaning |
|---|---|
| `objectives` | Detached scalar objective tensor for each accumulation round. |
| `metrics` | Reconstructed nondifferentiated objective metrics for each round. |
| `step_number` | Completed optimizer-step count. |
| `diagnostics` | `None`, or a deferred `DiagnosticsHandle` for a traced call. |

The resolved `StepDiagnostics` has six views, in two tiers. The first tier
is the reference for reading the step against its plan:

| View | Purpose |
|---|---|
| `summary` | The reconciliation: profiled versus real task time, simulated versus real waiting, the selected span, the makespan, and the call-level host totals. |
| `tasks` | Every selected task by execution task id: one `TaskRecord` placing simulated beside measured, with the host boundary costs. |
| `transfers` | Every scheduled transfer by transfer id, grouped as `transfers.fetch` and `transfers.evict`: one `TransferRecord` each, simulated beside measured, naming the tasks it sits between. |
| `timelines` | The order of the step on three lanes -- `compute`, `fetch`, `evict` -- as references into `tasks` and `transfers`, with each transfer lane's summary and the alignment between simulated and device time. |

The second tier is the evidence behind it, named by the component that
produced it:

| View | Purpose |
|---|---|
| `allocator` | The ordered allocation and free ledger, and the pool's geometry before and after the step. |
| `runtime` | Runtime counter deltas, terminal queue state, trace capacity and overflow, and the raw runtime event records. |

`StepDiagnostics.as_dict()` renders exactly these six under
`shadowspill.step_diagnostics/v1`; see [Exporting diagnostics](#exporting-diagnostics).

The sections below follow that shape, one per view, and each names every
field the view carries:

```text
StepDiagnostics
├── summary                          StepTimingSummary
│   └── phase_comparisons[]          PhaseTimingComparison
├── tasks[execution_task_id]         TaskRecord
├── transfers.fetch / .evict[id]     TransferRecord
├── timelines                        Timelines
│   ├── compute[]                    execution task ids, in compute-stream order
│   └── fetch / evict                TransferLane
│       ├── order[]                  transfer ids, in lane order
│       └── summary                  LaneSummary
├── allocator                        AllocatorTrace
└── runtime                          RuntimeTrace
```

The planning-side equivalent is the
[PlanReport field reference](plan-report-fields.md), a separate page because
it describes a different object measured on different clocks.

### Field names

Two conventions hold everywhere in these records, and reading a field name
tells you what kind of number it is.

| Suffix | Kind | Example |
|---|---|---|
| `_at_seconds` | An instant, measured from the trace's origin | `compute_started_at_seconds` |
| `_seconds` | A duration | `input_readiness_wait_seconds` |

No record stores a value that is the difference of two of its own fields, so
there is one place to read each measurement and no second copy to disagree
with it. The quantities that used to have fields are the obvious
subtractions: a task's kernel time is `compute_finished_at_seconds` minus
`compute_started_at_seconds`, the opening boundary's cost is
`before_task_exited_at_seconds` minus `before_task_entered_at_seconds`, and
the frontend's lead is `compute_reached_at_seconds` minus
`before_task_exited_at_seconds`.

### Clocks

Every time is in seconds, on one of the two clocks
[timelines](../architecture/timelines.md) defines. Fields on the device
timeline are named `compute_*` for the compute stream and `lane_*` for a
transfer lane; fields on the host clock sit under `host` in the records and
carry `before_task`, `after_task`, `dispatch`, `queued`, `reserved`,
`dispatched`, or `completion_observed` in their names. Simulated times are
the simulator's own clock shifted so that the
first selected task starts at zero, and every delta is taken after that shift,
so a delta reads as drift within the step. The step's prologue,
`timelines.first_task_started_at_seconds`, is therefore read here once rather than
folded into every delta; the part of it spent waiting for the first task's
inputs is `summary.real_initial_readiness_wait_seconds`, and the rest is input
staging and the first dispatch.

## Summary

```python
summary = diagnostics.summary

print("profiled tasks", summary.profiled_task_seconds)
print("real task events", summary.real_task_event_seconds)
print("simulated idle", summary.simulated_inter_task_idle_seconds)
print("real idle", summary.real_inter_task_idle_seconds)
print("simulated span", summary.simulated_selected_span_seconds)
print("real span", summary.real_selected_span_seconds)
print("simulator makespan", summary.simulator_makespan_seconds)
print("complete", summary.trace_complete)
```

The summary decomposes the selected-task span as:

```text
selected-task span = sum of selected task-event durations
                   + gaps between selected task events
```

| Field | Interpretation |
|---|---|
| `profiled_task_seconds` | Sum of isolated task profiles selected by the plan. |
| `real_task_event_seconds` | Sum of compute-stream intervals measured in the real step. |
| `task_event_delta_seconds` | Real task-event sum minus profiled sum. |
| `simulated_inter_task_idle_seconds` | Idle between simulated selected task intervals. |
| `real_inter_task_idle_seconds` | Idle between real selected task events. |
| `inter_task_idle_delta_seconds` | Real idle sum minus simulated idle sum. |
| `real_inter_task_readiness_wait_seconds` | Idle spent waiting at a task's boundary, for its inputs and for the ranges it allocates into. |
| `real_inter_task_exposed_overhead_seconds` | The rest of that idle: the frontend had not reached the next task. |
| `simulated_inter_task_readiness_wait_seconds` | The modeled counterpart of the readiness wait. |
| `real_initial_readiness_wait_seconds` | The first task's wait, which precedes the span. |
| `real_minimum_frontend_lead_seconds` | The smallest lead the frontend held over the stream. |
| `simulated_selected_span_seconds` | First selected task start through last selected task end in simulation. |
| `real_selected_span_seconds` | Same boundary using real compute-stream events. |
| `selected_span_delta_seconds` | Real selected span minus simulated selected span. |
| `simulator_makespan_seconds` | Complete simulated schedule, including modeled terminal work. |
| `simulator_terminal_tail_seconds` | Simulated work after the last selected compute task. |
| `call_seconds` | The whole planned call on the host clock. |
| `prior_invocation_drain_seconds` | Host time this call spent at its start waiting for the previous invocation's plan to go idle. A plan assumes its initial objects are resident when it begins, and the invocation before it ends by writing them back, so the next call cannot overlap that writeback. The first invocation has nothing to wait for and reports zero, which is why a trace taken on a warm first step shows zero here even though every later step pays it. |
| `initial_actions_seconds` | Host time submitting the opening placement batch. |
| `trace_setup_seconds` | One-time trace setup, which only the first traced call pays. |
| `optimizer_span_seconds` | First optimizer task's start through the last one's end, on the device timeline. |
| `phase_comparisons` | Profiled versus real task-event sum by semantic phase. One `PhaseTimingComparison` per phase, carrying `phase`, `profiled_task_seconds`, `real_task_event_seconds`, and `delta_seconds`, which is the second minus the first. |
| `trace_complete` | False when any required trace source overflowed or was incomplete. |

`real_inter_task_idle_seconds` is exactly `real_inter_task_readiness_wait_seconds` plus
`real_inter_task_exposed_overhead_seconds`: the stream is either waiting at a task's
boundary, or it has nothing to run at all. Between one task ending and the next
computing, the compute stream travels to the next task's readiness marker and
then waits there -- first for the task's inputs to be resident, then for the
ranges its allocations reuse to be released. The travel is what dispatch costs
and the waiting is what residency costs. Reading the total alone invites the
wrong conclusion, because the parts move independently -- a schedule that
fetches later raises the wait while leaving dispatch untouched.

The waiting is reported by cause on each task record, as
`input_readiness_wait_seconds` and `allocation_reuse_wait_seconds`. Both leave
the stream equally idle, so the aggregate counts them together; they are
separate fields because one is data that has not arrived and the other is an
address that is not free.

The exposed overhead is called exposed because the frontend does this work
at every boundary, and running ahead hides whatever the lead covers. The
field is the shortfall, not the cost of the work. It is zero wherever the
frontend runs ahead of the device, because a task it has already reached has
its readiness marker on the stream before the previous task ends. It is never
*exactly* zero, because two consecutive event records on a stream sit about
half a microsecond apart no matter what lies between them -- measured over
one olmoe step that floor ran from 0.24 microseconds at the lowest boundary
to 1.22 at the upper quartile -- so read a microsecond or two as nothing and
anything above it as real. The distribution is what to read, not the total:
a rising median means the frontend is losing its lead everywhere, and a
heavy tail means it is losing it somewhere specific, which the compute
records say where: `compute_reached_at_seconds` minus
`before_task_exited_at_seconds` is the lead the frontend held for that task.

`real_initial_readiness_wait_seconds` is the first task's wait. It ends where
the span begins, so no span-relative number here contains it, and on a step
that opens by fetching its parameters it can exceed everything that follows.
Use the selected span for the warmed first-task-to-final-task comparison. Do
not compare an end-to-end host call that includes startup and cooldown with
a selected-task simulator interval without accounting for those boundaries;
`call_seconds` is that end-to-end call.

## Tasks

`diagnostics.tasks` maps chronological `execution_XXXXXX` ids, matching
`PlanReport.diagnostics.tasks`, to one `TaskRecord` each; `task_id` names
the Program task. Every record carries the same groups -- identity,
simulated, stream, delta, host -- so a task reads like a transfer.

| Group | Fields | Meaning |
|---|---|---|
| Identity | `execution_task_id`, `task_id`, `execution_ordinal`, `semantic_name`, `phase`, `microbatch` | Which task, and where it sits in the plan. |
| Simulated | `simulated_ready_at_seconds`, `simulated_started_at_seconds`, `simulated_finished_at_seconds`, `expected_profile_seconds` | When the simulator had the inputs ready, when it started and ended the task, and the isolated profile it ran for, which is what the plan was built from. |
| Compute stream | `compute_reached_at_seconds`, `compute_started_at_seconds`, `compute_finished_at_seconds` | When the compute stream reached the task's readiness marker, when its kernels started after the waits, and when they finished. |
| Compute-stream waits | `input_readiness_wait_seconds`, `allocation_reuse_wait_seconds` | Between reaching and starting: inputs still being fetched, then ranges the task's allocations reuse still owned by a transfer. These two split the interval, so together they are `compute_started_at_seconds` minus `compute_reached_at_seconds`. Both are the *stream* waiting; the host's own wait for the same ranges is `dispatch_allocation_reuse_seconds`. |
| Delta | `start_delta_seconds`, `end_delta_seconds` | Device minus simulated, after aligning the two clocks. Both need that alignment, which is why they are stored rather than derived. |
| Host boundaries | `before_task_entered_at_seconds`, `before_task_exited_at_seconds`, `after_task_entered_at_seconds`, `after_task_exited_at_seconds` | Four instants partitioning the frontend's cycle for this task with no gap: entering the opening boundary, leaving it for the compiled call, the call returning, and leaving the closing boundary. The opening boundary, the call, and the closing boundary are the three differences between them. |
| Inside the opening boundary | `dispatch_input_lookup_seconds`, `dispatch_storage_rebind_seconds`, `dispatch_input_acquire_seconds`, `dispatch_allocation_reuse_seconds`, `dispatch_argument_assembly_seconds` | Resolve frontend bindings; rebind changed PyTorch storages; acquire the task's inputs from the runtime; wait on the host until every transfer that still owns a range this task allocates into has published its completion event; assemble predecoded arguments. Disjoint parts of the opening boundary. |
| Inside the closing boundary | `dispatch_output_flatten_seconds`, `dispatch_output_classification_seconds`, `dispatch_output_adoption_seconds`, `dispatch_output_state_publish_seconds`, `dispatch_output_publish_seconds`, `dispatch_dematerialize_seconds`, `dispatch_cleanup_seconds` | Flatten the output pytree; match leaves with contracts; adopt returned allocations; publish output state and bindings; publish public outputs; drop released bindings; remove terminal state. Disjoint parts of the closing boundary. |

The device timeline of one task, with the host's cycle beneath it:

```text
compute:  reached ---- input readiness wait ---- allocation reuse wait ---- started ==== kernels ==== finished
host:     before_task entered ---- lookup / rebind / acquire / allocation reuse / assemble ---- exited
                                                        compiled call
                                   after_task entered ---- outputs ---- exited
```

`dispatch_allocation_reuse_seconds` and `allocation_reuse_wait_seconds` are
the same dependency on two clocks and are usually very different sizes. The
host spins until the worker has *published* each event; the stream then waits
for the copy itself. A large host value with a small stream value means the
frontend absorbed the wait and could not run ahead. See
[task boundaries](../architecture/task-boundaries.md).

Interpret the deltas together:

| Pattern | Likely area |
|---|---|
| Kernel time exceeds `expected_profile_seconds` while the start delta stays small | Task profile, allocator callbacks inside the callable, provider behavior, or kernel variation. |
| Start delta grows along the lane while kernel times match their profiles | Readiness, dispatch gaps, or drift on a transfer lane -- read the lanes next. |
| Start delta grows and later shrinks | Simulator conservatism or recovered overlap. |
| A compute-stream readiness gap with a small `before_task` cost | Ordinary fetch readiness, expressed as a stream event wait, not a blocked Python thread. |
| A large opening boundary | Lookup, rebinding, runtime work, or the host's own wait on range reuse -- read the `dispatch_*` parts to tell them apart. |

To find expensive graph-task boundaries without bias from many short optimizer
tasks:

```python
rows = sorted(
    diagnostics.tasks.values(),
    key=lambda task: task.after_task_exited_at_seconds - task.before_task_entered_at_seconds,
    reverse=True,
)
for task in rows[:10]:
    print(task.execution_task_id, task.semantic_name, task.phase)
```

## Transfers

`diagnostics.transfers.fetch` and `diagnostics.transfers.evict` map transfer
ids to one `TransferRecord` each. `transfer_id` is `<direction>_<sequence>`,
and `sequence` is the transfer's FIFO position on its lane. The three
relation fields are execution task ids, so `diagnostics.tasks[record.next_access]`
is the task a fetch was made for.

| Group | Fields | Meaning |
|---|---|---|
| Identity | `transfer_id`, `direction`, `sequence`, `triggered_by`, `alias_group_id`, `bytes` | Which transfer and what it moved. `triggered_by` is what released it: the execution task id of the task whose completion did, a key into `tasks`, or `init` for the opening placement batch the runtime issues before the first task. A scheduled transfer's id is `<direction>_<sequence>`; an opening one's is `<direction>_opening_<index>`. |
| Relations | `previous_access`, `next_access`, `modified_by` | The object's place in the step, by execution task id: the last selected task up to and including the trigger that referenced the object, the first later one that does, and the last one up to the trigger that created or mutated it. `init` means no such task before the transfer, so the bytes are what the step was given; `persistent` means none after it within this call, so the object outlives the step. A fetch exists for its next access; an evict saves what its modifier produced. |
| Simulated | `simulated_ready_at_seconds`, `simulated_started_at_seconds`, `simulated_finished_at_seconds` | When the transfer could start, when the lane started it, and when it ended, at the bandwidth the plan assumed, which the plan summary states. `None` for an opening transfer, which the simulator does not model. |
| Lane | `lane_started_at_seconds`, `lane_finished_at_seconds` | The copy's interval on its transfer lane, bracketed by timing events the worker recorded immediately before and after the copy. `None` when the trace could not measure this transfer. |
| Delta | `start_delta_seconds`, `end_delta_seconds` | Device minus simulated after alignment; `None` without a lane interval or without a simulation. |
| Host | `queued_at_seconds`, `reserved_at_seconds`, `dispatched_at_seconds`, `completion_observed_at_seconds` | When the action was queued, when its destination was reserved, when the worker handed the copy to the lane, and when the worker's nonblocking poll saw it complete. |

Read `dispatched_at_seconds` against `lane_started_at_seconds` to see how long a
copy sat behind its predecessors on the lane, and `completion_observed_at_seconds`
against `lane_finished_at_seconds` to see the worker's polling lag. Neither gap is
transfer time; the lane interval is.

Check action identity and byte totals before interpreting timing: the
records are joined to the trace by lane sequence and validated against task,
object, and bytes, so a mismatch raises rather than producing a misleading
lane.

## Timelines

`diagnostics.timelines` is the order of the step, as references. `compute`
is the tuple of execution task ids in compute-stream order; `fetch` and
`evict` are each a `TransferLane` whose `order` is the tuple of transfer ids
in the lane's FIFO order -- the opening batch first, then the plan's
transfers by `sequence` -- and whose `summary` is a `LaneSummary`. `first_task_started_at_seconds` is where the
first selected task's simulated start fell on the device timeline.

```python
timelines = diagnostics.timelines
print("first task on the device at", timelines.first_task_started_at_seconds)
for task_id in timelines.compute:
    record = diagnostics.tasks[task_id]
    kernels = record.compute_finished_at_seconds - record.compute_started_at_seconds
    print(task_id, record.start_delta_seconds, kernels - record.expected_profile_seconds)
for lane, records in (
    (timelines.fetch, diagnostics.transfers.fetch),
    (timelines.evict, diagnostics.transfers.evict),
):
    print(lane.summary.direction, lane.summary.effective_bandwidth_bytes_per_second)
    for transfer_id in lane.order:
        print(transfer_id, records[transfer_id].start_delta_seconds)
```

`LaneSummary` states what the lane did over the step:

| Field | Meaning |
|---|---|
| `direction`, `transfers`, `bytes` | The lane and its scheduled traffic. |
| `simulated_busy_seconds` | Lane time the simulator priced for these transfers. |
| `measured_transfers`, `lane_busy_seconds` | How many transfers carry a stream interval, and the lane time those intervals add up to. |
| `effective_bandwidth_bytes_per_second` | Measured bytes over `lane_busy_seconds`; compare with the assumed bandwidth in the plan summary. `None` when nothing was measured. |
| `largest_start_delta_seconds`, `largest_start_delta_transfer_id` | The signed start delta furthest from zero on the lane, and the transfer that reached it: where the lane had drifted furthest from the simulation. |
| `opening_transfers`, `opening_bytes` | The opening placement batch on this lane: transfers the runtime issued before the first task to restore the step's initial objects. They precede the span, carry no simulation, and lead the lane's order with `triggered_by` `init`. |

## Allocator

`AllocatorTrace.events` is the ordered allocation/free ledger. Every event
includes sequence, task ID, allocation ID, generation, requested and charged
bytes, slab offset, kind, and category.

The terminal summary is the pool's geometry around the step:

| Field | Meaning |
|---|---|
| `live_allocations_before`, `live_allocations_after` | Allocations the pool held when the step began and ended. |
| `allocated_bytes_before`, `allocated_bytes_after` | Bytes those allocations charged. |
| `peak_allocated_bytes` | The high-water mark reached during the step. |
| `free_bytes_after` | Bytes the pool holds free at the end. |
| `free_prefix_bytes_after` | Of those, the contiguous run at the front of the slab. |
| `largest_free_range_bytes_after` | The largest single contiguous free range. |
| `external_fragmentation_bytes_after` | Free bytes that no single allocation can use because they are split across ranges. |
| `blocked_allocators_after` | Allocators still waiting for a range when the step ended. |
| `overflow` | Whether the event ledger filled and dropped records, which makes `events` partial. |

`requested_bytes` is the caller request. `charged_bytes` is the aligned range
owned by the pool. Compare requested totals when studying framework behavior
and charged totals when studying physical capacity.

Growth in live allocations or allocated bytes over repeated equivalent steps
requires explanation. A small `largest_free_range_bytes_after` despite large
`free_bytes_after` indicates fragmentation. A blocked allocator remaining at
step end is not a healthy steady state.

## Runtime

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
| `step_id`, `began_at_ns`, `ended_at_ns` | The trace's identity and its host-clock bounds; the beginning is the host origin the timelines count from. |
| `event_capacity`, `allocation_event_capacity`, `event_overflow`, `allocation_event_overflow` | Configured capacities and whether either buffer overflowed. |
| `events` | Raw bounded runtime-event records, including the transfer events the lanes were built from. |

Trace storage is preallocated and bounded. Treat any overflow as incomplete
evidence. The training frontend currently prepares 1,000,000 runtime-event
slots and 1,000,000 allocation-event slots for a traced call; the recorded
capacities, rather than that default, are authoritative for a specific
result. `summary.trace_complete` provides the combined verdict.

## Exporting diagnostics

`StepDiagnostics.as_dict()` returns a JSON-friendly value with schema
`shadowspill.step_diagnostics/v1` and exactly the six views above:

```python
import json
with open("step-diagnostics.json", "w", encoding="utf-8") as stream:
    json.dump(diagnostics.as_dict(), stream, indent=2, sort_keys=True)
```

`tasks` is keyed by execution task id and `transfers` holds `fetch` and
`evict`, each keyed by transfer id. Under `timelines`, `clocks` restates the
three clocks and the first task's start, `compute` is the list of execution
task ids in stream order, and `fetch` and `evict` each hold `summary` and
`order`, so a lane plots by walking its order through the records. The dictionary is an observation of one real
call. It is not an execution plan and has no `from_json()` planning
constructor.

## Common investigations

| Symptom | Look here first |
|---|---|
| Real span is slower than simulated | `summary`: task-event delta versus inter-task-idle delta. |
| One task is slower than profiled | Its task record's duration delta, then allocator events within that task. |
| Device execution has gaps between graph tasks | Task records' waits and `frontend_lead_seconds`, then their host costs. |
| Fetch appears late | Its transfer record: simulated versus stream start, host queued/reserved/dispatched, and the records before it on the lane. |
| Simulated waiting exceeds real waiting | Lane summaries: effective bandwidth against the assumed bandwidth in the plan summary. |
| Memory unexpectedly fills | Charged allocator events, peak bytes, largest free range, pending retirements, and each transfer's `next_access`. |
| A step appears to leak | Compare before/after live bytes and repeat across steps; include terminal caller-owned outputs in the interpretation. |
| Trace looks truncated | Both overflow flags, event capacities, and `summary.trace_complete`. |
| Runtime and plan identities are unclear | Join `execution_task_id` with the [PlanReport guide](plan-report.md). |

How the two clocks are measured, and why every lane can share one zero, is
described in [timelines](../architecture/timelines.md).
