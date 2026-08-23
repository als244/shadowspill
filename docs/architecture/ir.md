# Intermediate representation

The ShadowSpill IR is framework-neutral. Its public Python values live in
`shadowspill.ir`; the same concepts are flattened into C inputs at planner,
simulator, and runtime boundaries. It is produced by [PyTorch lowering](lowering.md)
and consumed by [planning](planning.md), [simulation](simulation.md), and
[runtime materialization](memory-runtime.md).

## Program

A `Program` contains immutable logical facts:

- `ObjectSpec` values and alias groups;
- `TaskSpec` values in execution order;
- per-task `TaskProfile` timing and memory measurements;
- mutations and persistence;
- compute and transfer resources;
- `RecomputationGroup` alternatives.

An object denotes one logical alias bundle, not one observed pointer. Views of
the same storage root share an object and retain their own geometry. Program
identity is deterministic and independent of diagnostic wall times or cache
paths.

An alias group may declare one runtime-global shared-residency policy:

| Policy | Contract |
|---|---|
| `SHARED_READ_ONLY` | The execution-pool lease is shared by callables and may never be mutated or replaced. |
| `SHARED_WRITABLE_CAUSAL` | The lease is shared; every reader/writer acquires the current generation and its readiness dependency. |
| `SHARED_WRITABLE_UNORDERED` | The lease is shared and may be mutated in place; cross-callable read/write visibility is intentionally unordered. |

Shared aliases are not plan-owned residency. They cannot appear in memory
actions, initial/final schedule residency, or recomputation-retained sets.
Only the causal policy may publish task outputs, because each consumer
re-acquires and rebinds the resulting generation. The read-only policy accepts
inputs only. The unordered policy requires stable-address in-place writes.

## Recomputation

A `RecomputationGroup` owns mutually exclusive `RecomputationOption` values.
The common case is a save graph pair and a full-recompute graph pair, but the
IR does not restrict the number of alternatives. A
`RecomputationSelection` chooses exactly one option per group.

Stage partitioning and recomputation are separate concepts. A stage is an
ordered model partition. Structurally equivalent stage occurrences may share
one graph-pair contract after shapes and input roles are known. See [graph-pair
construction](graph-pair-construction.md) for how those alternatives are
built and [recomputation selection](recomputation-selection.md) for how complete
occurrence-level assignments are selected.

## Memory schedule

A `MemorySchedule` combines initial `ResidencySpec` values with ordered
`MemoryAction` values. Canonical serialized action kinds are `prefetch` and
`offload`. Explanatory documentation and runtime names use **fetch** for
spill-to-execution movement and **evict** for execution-to-spill movement.

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
- selected graph-pair identity;
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

Previous: [Architecture overview](overview.md). Next: [PyTorch capture and
lowering](lowering.md).
