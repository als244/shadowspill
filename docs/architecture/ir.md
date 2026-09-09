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
| `ExecutionPlan` | The answer bound to a frontend and to physical memory. Below. |

A [search](search.md) is what turns the second into the third. This page
covers the two that come back; the first two have pages of their own.

## Memory schedule

A `MemorySchedule` combines initial `ResidencySpec` values with ordered
`MemoryAction` values. Canonical serialized action kinds are `fetch`,
`release`, `evict` and `write_back`. A **fetch** copies spill to
execution; a **write-back** copies execution to spill and keeps the
execution copy, so the spill copy is current again and a later release costs
nothing; a **release** drops the execution copy, which must not be the only current
copy of a value still needed; an **evict** is the two in one, a write-back where the
spill copy is stale followed by the release. Explanatory documentation and
runtime names use **fetch** for spill-to-execution movement and **evict**
for execution-to-spill movement that frees the execution copy.

An action trigger is a task boundary. Triggering a fetch reserves destination
capacity immediately; reaching the fetch lane head later submits the copy.
This distinction is part of both physical admission and simulation.

A memory action is a planning decision, not an allocator call. Executing one
implies several *pool operations* - reserve, acquire, retire - which are a
separate vocabulary belonging to physical admission
([physical admission](physical-admission.md#two-vocabularies-actions-and-operations)).

## Execution plan

An `ExecutionPlan` resolves a selected schedule into immutable execution
records:

- chronological `execution_XXXXXX` identity;
- semantic task name and secondary canonical IR ID;
- direct object references and predecoded input slots;
- mutations, outputs, and ordered actions;
- the option this task's group settled on;
- physical admission and predicted timing.

The chronological execution ID is the primary diagnostics key. Semantic names
are human-readable labels; canonical IR IDs remain secondary provenance.

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

Previous: [Architecture overview](overview.md). Next: [The
ShadowSpillProgram](program.md).
