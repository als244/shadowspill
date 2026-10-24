# Reusable planning and budget sweeps

A saved ProfiledProgram contains its Program and applied profiles. It can be
loaded independently, without a model instance, compiler cache, GPU, or runtime
pool. Planning accepts new memory budgets and communication costs each time.

## Replan one program

```python
from shadowspill import (
    ProfiledProgram, CommunicationInfo, MemoryBudgets, plan_program, write_report,
)

program = ProfiledProgram.load("profiled.ssp")
communication = CommunicationInfo(
    fetch_bytes_per_second=24e9, evict_bytes_per_second=20e9,
)
for gib in (8, 16, 28):
    plan = plan_program(
        program,
        memory_budgets=MemoryBudgets(execution_bytes=gib << 30, spill_bytes=144 << 30),
        communication_info=communication,
    )
    plan.save(f"plan-{gib}.ssp")
    write_report(plan, f"report-{gib}")
    print(gib, plan.summary)
```

`execution_bytes` is the total per-rank device allowance, including recorded
runtime/provider overhead. `spill_bytes` is a separate per-rank storage capacity.
Preparation creates no pool whose size limits later searches. Measured task
implementations, shapes, and compatibility requirements remain fixed.

## Compare candidates

`sweep_programs` is an optional convenience loop. It records infeasible cases and
keeps each successful Plan plus a winner for each budget. Callers can instead
write their own loop over `plan_program`.

```python
from shadowspill import sweep_programs

result = sweep_programs(
    {"batch4": ProfiledProgram.load("batch4.ssp"),
     "batch8": ProfiledProgram.load("batch8.ssp")},
    memory_budgets=[MemoryBudgets(gib << 30, 144 << 30) for gib in (8, 16, 28)],
    communication_info=communication,
)
result.save("sweep.ssp")
write_report(result, "sweep-report")
for budget, plan in result.best_per_budget.items():
    print(budget, plan.summary)
```

The candidates should represent equivalent useful work when comparing throughput.
Planning does not build new candidates or measure missing tasks. Invalid inputs
and implementation errors propagate; an unsuccessful search is retained as a
case with diagnostics.

## Execute the selected artifact

A separate worker loads a selected Plan and supplies its model tensors and
rank-local inputs. It does not repeat build, measurement, or search. See the
[complete training example](generic-training.md) for initialization and execution.
The Plan archive includes selected task code, schedule, memory layout, logical
groups and input contracts. Weights and compatible provider libraries remain
external. Compiler caches are optional and are not part of the artifact store.

Manual task durations and memory values can produce simulation reports through
`Profiles` and `apply_profiles`. Values without measurement metadata are estimates.
Estimated memory does not qualify a Plan for execution. Unknown workspace or
provider overhead must be supplied explicitly; it never defaults to zero.

For reproducible comparisons, keep the Program, Profiles, CommunicationInfo,
selected Plans and reports alongside the configuration and source revision. Each
phase artifact is one archive file. The [phase API guide](../architecture/offline-pipeline.md)
describes their contents and compatibility checks.
