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
- task alternatives, as `TaskAlternativeGroup` values.

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
actions, initial/final schedule residency, or a task-alternative option's
retained-alias sets.
Only the causal policy may publish task outputs, because each consumer
re-acquires and rebinds the resulting generation. The read-only policy accepts
inputs only. The unordered policy requires stable-address in-place writes.

## Task alternatives

A `TaskAlternativeGroup` owns mutually exclusive `TaskAlternativeOption`
values. Each option names the tasks it activates and the aliases it keeps
resident. A `TaskAlternativeChoice` fixes exactly one option for one group:

```text
TaskAlternativeGroup("stage_0007")
├── TaskAlternativeOption("save")
│     active_task_ids          = (forward_0007, backward_0007)
│     retained_alias_group_ids = (activation_0007,)
└── TaskAlternativeOption("recompute")
      active_task_ids          = (forward_0007r, backward_0007r)
      retained_alias_group_ids = ()

TaskAlternativeChoice(group_id="stage_0007", option_id="save")
```

Choosing `save` puts `forward_0007` and `backward_0007` into the executing task
set and holds `activation_0007` across the group's boundary. Choosing
`recompute` puts the other two in and retains nothing, spending compute instead
of memory. Fixing one option for *every* group resolves the Program to a single
concrete task set; a Program with no groups is already resolved.

The IR neither restricts the number of alternatives nor knows what they mean.
The two above come from training, where they are a save graph pair and a
full-recompute graph pair, but nothing here says so.

Stage partitioning is a separate concept: a stage is an ordered model
partition, and structurally equivalent stage occurrences may share one
structural contract once shapes and input roles are known. See [graph-pair
construction](graph-pair-construction.md) for how the training frontend builds
these alternatives and [graph-pair selection](graph-pair-selection.md) for how
it fixes a complete set of them -- both being that frontend's names for the
instance it builds of what this page describes generally.

## Phases and sinks

Every `TaskSpec` carries a `phase`: a plain identifier string that defaults to
`compute`. The IR never interprets it. It is validated as an identifier and
carried through, and nothing in `shadowspill.ir`, the simulator, or the runtime
compares it against a particular value. A PyTorch training Program labels its
tasks `forward`, `backward`, and `recomputation`; a Program from anywhere else
may use whatever names describe its own structure.

What the phase is *for* is separating a task's dependency graph from the graph
of the phase it belongs to, which is what the planner needs in order to decide
whether a value can be recomputed at all.

### What a sink is

A task is a **sink of a phase** when no other task *in that same phase*
depends on it:

```text
forward:   t1 --> t2 --> t3      t3 is a sink of `forward`: no forward
                          |      task consumes it ...
                          v
backward:                t4      ... its only consumer is in another phase
```

Arrows here run from producer to consumer, the direction the value travels.
Under that convention a node with no outgoing edge is a sink, which is the
standard graph reading and the one these pages use everywhere.

`TaskSpec.dependencies` stores the opposite orientation: it lists what a task
consumes, so its arrows point backwards, at producers. Read that field
literally and `t3` looks like a *source*, because nothing points at it. Both
describe the same graph. The documentation fixes the data-flow reading, so
"sink" always means nothing downstream in this phase consumes it.

### Why a sink is pinned

A sink's value leaves its phase. Nothing later in the same phase reproduces
the inputs it would need, so at the moment the next phase reads it there is
nothing to recompute it from.

The planner uses this to pin every group whose `forward` tasks are sinks of
the `forward` phase to that group's `save` option, removing the choice from
the search instead of offering it and always rejecting it.

That rule names one phase on purpose, and the IR is the reason it can. A
Program that declares no `forward` phase -- the default phase is `compute` --
matches no forward tasks, so nothing is pinned and every alternative stays
open. Naming the phase is what confines a piece of training knowledge to
Programs that say they are training. Rephrasing it as "sinks of whatever phase
a group enters first" sounds more general and is strictly worse: it would pin
the terminal groups of *any* Program and delete the recomputation search for
everything that is not training.

So the generality here is in the IR, not in the rule. `phase` carries no
meaning, which lets one consumer attach meaning to one value without every
other Program inheriting it.

The rule is graph-derived either way: it uses task phase and dependency edges,
never model family, module name, stage number, or operator identity.

## Memory schedule

A `MemorySchedule` combines initial `ResidencySpec` values with ordered
`MemoryAction` values. Canonical serialized action kinds are `fetch`,
`release`, and
`evict`. Explanatory documentation and runtime names use **fetch** for
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

Previous: [Architecture overview](overview.md). Next: [PyTorch capture and
lowering](lowering.md).
