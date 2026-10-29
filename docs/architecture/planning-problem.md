# The planning problem

The public `plan_program(profiled_program, memory_budgets=...,
communication_info=...)` binds an ordering, costs communication and searches for
a schedule entirely on CPU. Its result is a saved, inspectable `Plan`.
The same `ProfiledProgram` can be reused across budgets and communication
assumptions without model construction, measurement or execution pools.
`reserved_bytes` keeps part of the execution budget out of the search: the own
overhead of Programs that will share the worker, so the Plan leaves room for
them while declaring the whole budget.

## Inputs and accounting

- `ProfiledProgram` supplies task durations, memory evidence, dependencies and
  residency boundaries.
- `MemoryBudgets` supplies the total per-rank device allowance and separate spill
  capacity. Fixed device overhead and communication overhead reduce the capacity
  available for scheduled objects and workspace.
- `CommunicationInfo` supplies topology, rates, latencies and optional measured
  transfers. Compatible measurements take priority over estimates.
- `SearchOptions` controls the search independently of these inputs.

Changing an ordering changes the task walk and gradient accumulation dependencies.
Changing budgets or communication assumptions changes the simulated machine.
Neither change requires recapturing the model. The [step structure](step.md) and
[search](search.md) pages describe these choices.

Measured memory contracts permit physical layout certification. Explicitly
estimated memory supports simulation only; it cannot certify executable storage.
Executor validates an executable Plan against its actual resources separately.

## Diagnostic planning problems

`ShadowSpillPlanningProblem` is the framework-independent representation used by
the lower-level planning evaluator and retained JSON corpora. It combines a
costed IR program with residency boundaries, simulation configuration, admission
facts and budget limits. It is separate from both the selected schedule and
search policy.

New [program collection](../../benchmarking/program_collection/README.md) saves
`ProfiledProgram` archives. The evaluator derives a problem from the archive and
the requested sweep. Collection does not bind budgets or communication costs.

| Field | Meaning |
| --- | --- |
| `program`, `role` | Costed work and invocation kind |
| `initial_residency`, `final_residency` | Required storage at invocation boundaries |
| `simulation_config` | Capacities and transfer costs for this question |
| `admission_facts` | Allocation contracts needed to certify a physical layout |
| `source_execution_budget_bytes` | Initial budget of this diagnostic problem |
| `maximum_execution_budget_bytes` | Upper limit accepted by this problem |
| `maximum_spill_budget_bytes` | Recorded spill capacity, not a ceiling |
| `fixed_execution_bytes` | Device overhead outside scheduled storage |
| `object_reserve_bytes`, `dynamic_scratch_reserve_bytes` | Additional explicit reservations |

For a new archive, the evaluator derives the diagnostic execution limit from its
sweep; collection imposes no pool-size ceiling. JSON serialization is documented
in the [planning format reference](../python/planning-json.md).

Previous: [The ShadowSpillProgram](program.md). Next:
[Simulation](simulation.md).
