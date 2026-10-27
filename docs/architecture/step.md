# Step structure and ordering

`build_step` produces a `Program` containing fixed task implementations, tensor
structure and save/recompute alternatives. `measure_program` supplies reusable
`Profiles`; `apply_profiles` produces a `ProfiledProgram`. None of these phases
requires execution pools or a planning memory budget.

```text
build_step → Program → measure_program → Profiles
                 └── apply_profiles ──→ ProfiledProgram
                                           │
                  ordering, budgets, communication_info
                                           ↓
                                      plan_program → Plan
```

The public [phase API](offline-pipeline.md) defines the artifacts and their
save/load boundaries. [Program collection](../../benchmarking/program_collection/README.md)
stops at a saved `ProfiledProgram`; [planning evaluation](../../benchmarking/planning_eval/README.md)
consumes it without importing a model or framework.

## Planning-time ordering

`StepDataOrdering` describes `depth` passes of `breadth` examples, whose product
must equal the Program's example count. Within a pass, forward visits each stage
for all examples; backward visits stages in reverse. Breadth one gives the
usual one-example-at-a-time order.

`pair_loss` pairs each example's terminal forward and backward.
`reverse_breadth` visits the youngest example first during backward. Both default
to true. Labels such as `2x4rp` record these settings in reports.

The planner binds this ordering to the saved stage templates. It chooses which
backward creates a gradient and which later backwards add into it, without
rebuilding or remeasuring the tasks. `optimizer_ordering` similarly selects
stage-interleaved or tail updates. Mutation and random-state dependencies still
constrain valid schedules; an incompatible ordering is rejected.

Previous: [Graph-pair construction](graph-pair-construction.md). Next:
[Importing state](state-import.md).
