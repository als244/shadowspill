# The ShadowSpillProgram

What the system plans *for*: one step of work, stated as tasks over objects,
with every measurement a planner needs and no policy at all.

A program is produced by [PyTorch lowering](lowering.md) and consumed by
[planning](planning.md), [simulation](simulation.md) and [runtime
materialization](memory-runtime.md). It says what the work *is*. It does not
say where the work starts, what machine it runs on, or how much memory there
is -- those turn a program into [a planning problem](planning-problem.md),
which is the question a [search](search.md) answers.

The type is `ShadowSpillProgram` in `shadowspill.ir`; the same facts are
flattened into C inputs at the planner, simulator and runtime boundaries.

## What it holds

A `ShadowSpillProgram` contains immutable logical facts:

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
of memory. Fixing one option for *every* group resolves the program to a single
concrete task set; a program with no groups is already resolved.

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
compares it against a particular value. A PyTorch training program labels its
tasks `forward`, `backward`, and `recomputation`; a program from anywhere else
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

Dependencies are data, and only data: a task lists the producers of what it
consumes and the earlier writers of what it accumulates into, nothing else.
The order of `ShadowSpillProgram.tasks` is the schedule, and the schedule is the
lowering's choice; it is not written into the dependencies. So any
topological order of a program's tasks is a legal schedule, and one lowering
can emit the same tasks in more than one order for the planner to compare.
The training lowering's orders are the microbatch walks of
[planning](planning.md#the-walk-through-the-microbatches).

### Why a sink is pinned

A sink's value leaves its phase. Nothing later in the same phase reproduces
the inputs it would need, so at the moment the next phase reads it there is
nothing to recompute it from.

Costing a program's alternatives uses this to pin every group whose `forward`
tasks are sinks of the `forward` phase to that group's `save` option, removing
the choice from the search instead of offering it and always rejecting it.

That rule names one phase on purpose, and the IR is the reason it can. A
program that declares no `forward` phase -- the default phase is `compute` --
matches no forward tasks, so nothing is pinned and every alternative stays
open. Naming the phase is what confines a piece of training knowledge to
programs that say they are training, and leaves the recomputation search
intact for every program that does not.

So the generality is in the IR, not in the rule. `phase` carries no meaning,
which lets one consumer attach meaning to one value without every other
program inheriting it. The rule is graph-derived: it uses task phase and
dependency edges, never model family, module name, stage number, or operator
identity.

## What it deliberately does not hold

No budget, no capacity, no bandwidth, no candidate policy, no plan. A
program is a statement of work, not a question about it, which is why one
program serves an entire budget frontier without being rebuilt: the
question changes, the program does not. Adding a planning option later
therefore cannot change what a saved program means.

Previous: [Intermediate representation](ir.md). Next: [The planning
problem](planning-problem.md).
