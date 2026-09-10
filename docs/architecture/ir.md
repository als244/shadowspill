# Intermediate representation

The ShadowSpill IR is framework-neutral. Its public Python values live in
`shadowspill.ir`; the same concepts are flattened into C inputs at planner,
simulator, and runtime boundaries.

Four types carry the whole story, and they are worth keeping straight
because three of them are easy to confuse:

| | |
|---|---|
| [`ShadowSpillProgram`](program.md) | The work: tasks over objects, with measurements. Says nothing about a machine. |
| [`ShadowSpillPlanningProblem`](planning-problem.md) | The question: a program, where it starts and ends, and what it runs on. |
| `MemorySchedule` | The answer: what moves, and at which boundary. Below. |
| `ExecutionPlan` | The answer made runnable: the schedule with its alternatives fixed, the layout it was admitted under, and what it is predicted to cost. Below. |

A [search](search.md) is what turns the second into the third. This page
covers the two that come back; the first two have pages of their own.

## Memory schedule

A `MemorySchedule` is a `ResidencySpec` per alias group at the opening
boundary, ordered `MemoryAction` values, and the same at the closing one.
Canonical serialized action kinds are `fetch`, `release`, `evict` and
`write_back`. A **fetch** copies spill to execution; a
**write-back** copies execution to spill and keeps the execution copy, so the
spill copy is current again and a later release costs nothing; a **release**
drops the execution copy, which requires the spill copy to be current; an
**evict** is the two in one, a write-back where the spill copy is stale
followed by the release. Explanatory documentation and runtime names use
**fetch** for spill-to-execution movement and **evict** for
execution-to-spill movement that frees the execution copy.

Every action names a trigger task, so an action's place in time is a task
boundary rather than a position in a lane. Triggering a fetch reserves
destination capacity immediately; reaching the fetch lane head later submits
the copy. This distinction is part of both physical admission and simulation.

A memory action is a planning decision, not an allocator call. Executing one
implies several *pool operations* - reserve, acquire, retire - which are a
separate vocabulary belonging to physical admission
([physical admission](physical-admission.md#two-vocabularies-actions-and-operations)).

## Execution plan

An `ExecutionPlan` is a schedule resolved into something the runtime can
admit:

- `program`, and the `schedule` over it;
- `selections`, one `TaskAlternativeChoice` per group, fixing the active task
  set;
- `entrypoints`, binding every active task that needs one to an executor and
  the contract digest it was compiled against;
- `admission`, the budgets, reserves and slab the layout was built against;
- `prediction`, the device and spill peaks and the makespan expected of it.

Construction validates all of it together -- the schedule against the program
and its selections, one entrypoint per active task that requires one, both
predicted peaks inside the admitted budgets -- so a plan cannot exist
half-resolved or over budget.

## Indexed projections

`IndexedProgram`, `IndexedMemorySchedule`, and `IndexedExecutionPlan` are compact
index-based projections used across the C ABI. `index_program()`,
`index_memory_schedule()`, and `index_execution_plan()` validate and
translate the public IR without changing its semantics.

## Validation

IR constructors validate references, unique identities, action ordering, and
selection consistency. Invalid input raises `ValidationError` before a C
component is invoked. Planner and simulator infeasibility is distinct from IR
validation failure.

Previous: [Architecture overview](overview.md). Next: [PyTorch capture and
lowering](lowering.md).
