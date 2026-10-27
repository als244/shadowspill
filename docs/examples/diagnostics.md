# Diagnosing a plan and real step

A saved Plan supplies the prediction. The result handle supplies optional measured
execution evidence. Use the same interface for forward and training Plans:

```python
from shadowspill import write_report

print(plan.summary)
write_report(plan, "reports/simulated")
pending = executor.run(args=examples, runtime_trace=True)
outputs = pending.result()
step = pending.diagnostics
assert step is not None
assert step.summary.trace_complete
```

Sort measured tasks by duration. Each record also contains its expected profile
cost and task identity, so no separate callable report is needed:

```python
largest = sorted(
    step.tasks.values(),
    key=lambda record: record.compute_finished_at_seconds
    - record.compute_started_at_seconds,
    reverse=True,
)
for record in largest[:10]:
    print(record.execution_task_id, record.expected_profile_seconds,
          record.compute_finished_at_seconds - record.compute_started_at_seconds)

for transfer in step.transfers.fetch.values():
    print(transfer)
```

Compare `step.summary.profiled_task_seconds` with `real_task_event_seconds`, then
inspect inter-task waits and the `fetch`/`evict` lanes. A slow task calls for a
kernel trace; a transfer delay calls for the lane's readiness, queue and completion
evidence. Check trace completeness before drawing conclusions from missing events.

[Plan inspection](../python/plan-report.md) and
[execution trace fields](../python/step-diagnostics.md) define the records.
