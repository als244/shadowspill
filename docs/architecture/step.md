# The step artifacts

A captured step, the question asked about it and the answer are three different
things. This page is the first: what a step *is*, in the two values that survive
capture and travel onwards. Both live in `shadowspill.step`, and neither imports
a framework.

## Why they are their own package

`StepDataOrdering` and `StepProgram` speak the vocabulary of a training step --
microbatches, passes, a recurrent role and an optional initial one. That is not
planning vocabulary, so they do not belong to the [planner](planning.md), which
is handed a [problem](planning-problem.md) and has no opinion about what shape
of step produced it.

They do not belong to the [PyTorch frontend](lowering.md) either, and this is
the load-bearing half. A `StepProgram` is read, validated and planned with no
PyTorch installed. That is what lets a corpus be collected in one process, on a
machine with a model and an accelerator, and planned in another that has
neither -- which is how [program
collection](../../benchmarking/program_collection/README.md) and [planning
evaluation](../../benchmarking/planning_eval/README.md) run separately. Under
`shadowspill.pytorch` these two values would make the import of a saved corpus
pull in a framework it does not need. So they sit beside both, as a peer
package holding the step's own description.

## `StepDataOrdering`

The walk a step takes through its microbatches: `depth` passes of `breadth`
microbatches, with the product pinned to the microbatch count. Within a pass the
forward is stage-major and the backward stage-major in reverse.

Two flags change the walk rather than its extent, and both are on by default.
`pair_loss` runs a microbatch's last stage forward and backward together, so
that stage's saved state is consumed as it is produced instead of being held for
the whole pass. `reverse_breadth` walks a pass's microbatches in reverse during
backward, so the microbatch whose activations are youngest is consumed first.
Neither means anything at `breadth = 1`, which is the microbatch-major walk: one
microbatch start to finish before the next.

Why the ordering is part of the step and not a planning option: it changes which
objects exist at the same time, so it changes the program that is lowered, not
the policy applied to one. Two orderings are two programs, and the search
compares plans for each rather than choosing between orderings itself. What each
walk does to peak residency is in
[graph-pair construction](graph-pair-construction.md).

`resolve()` fills in whichever of the two counts a caller left out, and refuses a
pair whose product is not the microbatch count rather than quietly dropping or
repeating data. `creates()` says whether one microbatch's backward for a stage
creates that stage's gradient -- the first the walk reaches does, and every later
one adds into it -- which is what makes gradient accumulation expressible as
ordinary object dependencies rather than a special case. `label` and
`from_label()` carry an ordering as `2x4rp`, which is how a figure or a report
names one.

## `StepProgram`

The recurrent [planning problem](planning-problem.md) a captured step lowered
to, the optional initial one for a step whose first invocation differs, the
orderings it was lowered under, and the provenance that says what produced them.

The two roles exist because a training step is not always the same step. An
optimizer whose state is created on the first step has to materialize it once,
so that step is a different program with a different peak, planned separately
and executed once. A step with nothing to materialize carries no initial problem
at all.

`digest` identifies planning content -- the programs, the orderings, and the
measurements they were taken under -- and not the run that produced them, so two
collections of the same step are one artifact rather than two that have to be
planned twice. `to_json()` and `from_json()` round a step through a file, and
that file is the unit a corpus is made of.

## Where they sit

```text
capture → lowering → StepProgram → ShadowSpillPlanningProblem → search → plan
                         │
                    StepDataOrdering travels with it, and with the plan report
```

`build_step_program()` stops at the `StepProgram`; `plan_program()` takes one of
its problems. The [planning orchestration](planning.md) page covers what happens
after, and [reusable artifacts](../python/api/artifacts.md) documents the fields.

Previous: [Graph-pair construction](graph-pair-construction.md). Next:
[Importing state](state-import.md).
