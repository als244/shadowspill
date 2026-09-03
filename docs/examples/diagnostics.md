# Diagnosing a plan and real step

The stable join key between planning and execution is
`execution_XXXXXX`. This recipe reads the traced step's summary, finds the
tasks contributing the most real selected-span time beside their selected
graph-pair variant, and reads the transfer lanes to separate task-duration
error from transfer drift.

```python
result = train_step(inputs, runtime_trace=True)
step = result.diagnostics.result()
report = train_step.plan_report

summary = step.summary
print("profiled task sum", summary.profiled_task_seconds)
print("real task-event sum", summary.real_task_event_seconds)
print("simulated inter-task idle", summary.simulated_inter_task_idle_seconds)
print("real inter-task idle", summary.real_inter_task_idle_seconds)
print("  waiting for inputs", summary.real_inter_task_readiness_wait_seconds)
print("  stream had nothing to run", summary.real_inter_task_exposed_overhead_seconds)
print("simulated selected span", summary.simulated_selected_span_seconds)
print("real selected span", summary.real_selected_span_seconds)

largest = sorted(
    step.tasks.values(),
    key=lambda record: record.compute_duration_seconds,
    reverse=True,
)
for record in largest[:10]:
    planned = report.diagnostics.task(record.execution_task_id)
    print(
        record.execution_task_id,
        planned.semantic_name,
        planned.chosen_graph_pair_variant,
        record.expected_profile_seconds,
        record.compute_duration_seconds,
        record.duration_delta_seconds,
        record.dispatch_before_task_seconds,
        record.dispatch_after_task_seconds,
    )

timelines = step.timelines
assumed = report.summary.fetch_bandwidth_bytes_per_second
for lane in (timelines.fetch, timelines.evict):
    print(
        lane.summary.direction,
        "effective", lane.summary.effective_bandwidth_bytes_per_second,
        "assumed", assumed if lane.summary.direction == "fetch"
        else report.summary.evict_bandwidth_bytes_per_second,
        "largest drift", lane.summary.largest_start_delta_seconds,
        "at", lane.summary.largest_start_delta_transfer_id,
    )
```

Interpret the first-level result before inspecting raw events:

| Observation | Next evidence |
|---|---|
| Real task-event sum is high | Sort `step.tasks` by `duration_delta_seconds`, then inspect that task's allocator events. |
| Real inter-task gaps are high | Sort `step.tasks` by `dispatch_before_task_seconds` plus `dispatch_after_task_seconds`, and read each record's waits and `frontend_lead_seconds`. |
| Simulated waiting exceeds real waiting | Compare each lane's effective bandwidth with the bandwidth the plan summary assumed. |
| Transfer timing diverges | Walk the lane's `order` through `step.transfers.fetch` or `.evict`: read the record's simulated start against its stream start, then its host queued, reserved, and dispatched times and the records before it. |
| Physical memory differs | Check charged allocator events, peak bytes, pending retirements, largest free range, fragmentation, and each transfer's `next_access`. |
| Evidence appears incomplete | Require `summary.trace_complete` and both overflow flags to be clean. |

The [PlanReport](../python/plan-report.md) and [StepResult
diagnostics](../python/step-diagnostics.md) guides define every field and the
corresponding serialization APIs.
