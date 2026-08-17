# Diagnosing a plan and real step

The stable join key between planning and execution is
`execution_XXXXXX`. This recipe finds the tasks contributing the most real
selected-span time, shows their selected graph-pair variant, and separates
task-duration error from inter-task-gap error.

```python
result = train_step(inputs, runtime_trace=True)
step = result.diagnostics.result()
report = train_step.plan_report

summary = step.summary
print("profiled task sum", summary.profiled_task_seconds)
print("real task-event sum", summary.real_task_event_seconds)
print("simulated inter-task gaps", summary.simulated_inter_task_gap_seconds)
print("real inter-task gaps", summary.real_inter_task_gap_seconds)
print("simulated selected span", summary.simulated_selected_span_seconds)
print("real selected span", summary.real_selected_span_seconds)

largest = sorted(
    step.tasks.values(),
    key=lambda item: item.task_compute_seconds or 0.0,
    reverse=True,
)

for timing in largest[:10]:
    planned = report.diagnostics.task(timing.execution_task_id)
    comparison = step.simulator_comparison[timing.execution_task_id]
    print(
        timing.execution_task_id,
        planned.semantic_name,
        planned.chosen_graph_pair_variant,
        comparison.expected_profile_seconds,
        comparison.observed_gpu_seconds,
        comparison.duration_delta_seconds,
        timing.host_before_task_seconds,
        timing.host_after_task_seconds,
    )
```

Interpret the first-level result before inspecting raw events:

| Observation | Next evidence |
|---|---|
| Real task-event sum is high | Sort `SimulatorTaskComparison.duration_delta_seconds`, then inspect task allocator events. |
| Real inter-task gaps are high | Sort complete host `before_task` + `after_task` costs and inspect readiness waits. |
| Transfer timing diverges | Compare transfer identity/bytes first, then queue, reservation, dispatch, and completion times. |
| Physical memory differs | Check charged allocator events, peak bytes, pending retirements, largest free range, and fragmentation. |
| Evidence appears incomplete | Require `summary.trace_complete` and both overflow flags to be clean. |

Planning and runtime dictionaries can be saved for offline inspection:

```python
import json

with open("plan-diagnostics.json", "w", encoding="utf-8") as stream:
    json.dump(report.diagnostics.as_dict(), stream, indent=2, sort_keys=True)

with open("step-diagnostics.json", "w", encoding="utf-8") as stream:
    json.dump(step.as_dict(), stream, indent=2, sort_keys=True)
```

These are diagnostic observations. Use `StepProgram.to_json()` and
`AnnotatedProgramPlan.to_json()` for artifacts that can be loaded back into a
planning workflow. The [PlanReport](../python/plan-report.md) and [StepResult
diagnostics](../python/step-diagnostics.md) guides define every layer in more
detail.
