# Plan and search

`plan_program` takes one fully profiled Program and chooses save/recompute
variants, residency, transfers and a physical memory layout. It returns a `Plan`
or a diagnostic failure. This phase is CPU-only and does not build, measure or
allocate execution pools.

```python
from shadowspill import (
    CommunicationInfo, MemoryBudgets, ProfiledProgram, plan_program, write_report,
)

plan = plan_program(
    ProfiledProgram.load("profiled.ssp"),
    memory_budgets=MemoryBudgets(execution_bytes=24 << 30, spill_bytes=128 << 30),
    communication_info=CommunicationInfo.load("communication.ssp"),
)
plan.save("training.ssp")
write_report(plan, "report")
```

## Options

<!-- api-options: packages/engine/src/shadowspill/planning/plan.py:plan_program -->

| Argument | Default | Meaning |
| --- | --- | --- |
| `program` | Required | `ProfiledProgram`, produced by `apply_profiles` or loaded from disk. |
| `memory_budgets` | Required | Total per-rank device allowance and separate spill capacity, in bytes. |
| `communication_info` | Required | Transfer/collective costs and available measurement evidence. |
| `pool_memory` | `None` | Optional calibrated process/pool reservation evidence. |
| `search_options` | `None` | Default PressureFit search, or explicit `SearchOptions`. |
| `ordering` | `None` | Default depth-first traversal, or `StepDataOrdering` for the captured calls. |
| `optimizer_ordering` | `"stage_interleaved"` | Update once required gradients and old-weight reads finish; `"tail"` delays updates until after backward. |

The device allowance includes measured fixed overhead and external headroom;
it is not just a task-workspace limit. Spill capacity is separate. An executable
Plan needs observed allocation evidence for its exact implementations. Numeric
estimates can support simulation, but leave `plan.simulation_only_reasons`
nonempty. Worker startup still checks compatibility and physical availability.
See [pool memory](pool-memory.md), [artifacts](artifacts.md#plan) and
[planning failures](../advanced/failures.md).

## Traversal and save choices

`optimizer_program(step_program)` derives an optimizer-only `Program` from the
saved update components of a complete-step Program. It needs no model, framework
import, capture or measurement. Apply the original profiles to the derived
Program, then plan it with the usual budgets and communication evidence:

```python
from shadowspill import apply_profiles, optimizer_program, plan_program

optimizer_plan = plan_program(
    apply_profiles(optimizer_program(step_program), step_profiles),
    memory_budgets=budgets,
    communication_info=communication,
)
```

It retains the captured optimizer tasks, reduction dependencies and weight
publications, with the same gradient-storage ownership as the source Program.
Admit it alongside compatible backward Plans in one Executor and invoke it with
`executor.run(plan="optimizer").result()`. It takes no examples and consumes the
existing gradients. The caller controls whether a generation is ready to update;
this low-level operation does not infer participation from gradient values.

`instantiate_examples(program, template_indices)` selects or repeats saved
physical example templates on CPU. For example, `[0, 0, 0]` repeats the first
captured shape three times; `[1, 0, 1]` uses the second, first and second shapes.
Each occurrence has independent input, activation and output storage, while
model, gradient and optimizer identities stay shared. Compiled task forms and
their profile keys remain unchanged:

```python
from shadowspill import instantiate_examples

three_examples = instantiate_examples(step_program, [0, 0, 0])
count_plan = plan_program(
    apply_profiles(three_examples, step_profiles),
    memory_budgets=budgets,
    communication_info=communication,
)
```

This supports complete-step and forward/backward Programs with saved templates.
Missing fresh/additive forms or measurements require preparation; instantiation
does not compile or profile them. Replan each count and jointly admit its state
and pools before execution. Input-capacity and packing coverage checks are
separate: a count fitting device memory does not authorize padding beyond a
declared batch limit. Registered model mutation and RNG ordering constraints
remain in force.

Traversal is a planning choice; microbatch geometry is a build choice. For eight
instantiated calls, `StepDataOrdering(depth=2, breadth=4)` makes two passes of four
microbatches each. Within a pass, stage-major execution can reuse fetched weights
across those microbatches. `depth * breadth` must equal the captured call count.
`reverse_breadth=True` and `pair_loss=True` are its defaults. Model mutations or
RNG effects can restrict legal reordering.

```python
from shadowspill.planner import SearchOptions
from shadowspill.planner.search.algorithms.pressurefit import PressureFit, PressureFitOptions
from shadowspill.step import StepDataOrdering

search = SearchOptions(
    workers=8,
    algorithm=PressureFit(PressureFitOptions(
        resolution_options=("0", "1/4", "1/2", "3/4", "1"),
        tail_save_counts=(0, 1, 2, 4, 8, 16),
    )),
)
plan = plan_program(
    profiled, memory_budgets=budgets, communication_info=communication,
    ordering=StepDataOrdering(depth=2, breadth=4), search_options=search,
)
```

Each tail count retains that many final eligible forward invocations **per pass,
repeated at every depth**. Every tail choice crosses every recompute share for
the remaining flexible groups. Required saves are excluded; duplicate choices
are searched once. These are choices among existing graphs, so no rebuild is
needed. Retention still permits offloading. The default shares are quarters,
with `tail_save_counts=(0,)`; small spaces may be enumerated exhaustively.

`SearchOptions.workers=0` uses available logical CPUs; `1` is serial.
`generic=GenericPlanningOptions(deterministic=True)` requests worker-count-independent
candidate outcomes, with extra search work; it defaults to `False`. Search traces
are opt-in via `trace_directory`. See [search options](neutral.md#searchoptions)
and [PressureFit controls](neutral.md#pressurefit) for limits and advanced knobs.

## Compare Programs and budgets

`sweep_programs` compares **already built and profiled** candidates representing
equivalent useful work. It returns `SweepResult`: `cases` retains Plans or
capacity/search failures, and `best_per_budget` chooses lowest simulated time.
Invalid inputs and implementation errors propagate.

<!-- api-options: packages/engine/src/shadowspill/planning/sweep.py:sweep_programs -->

| Argument | Default | Meaning |
| --- | --- | --- |
| `programs` | Required | Mapping of candidate names to `ProfiledProgram` objects. |
| `memory_budgets` | Required | Sequence of device/spill budgets to compare. |
| `communication_info` | Required | Shared communication evidence/costs. |
| `pool_memory` | `None` | Optional calibrated pool evidence. |
| `search_options` | `None` | Search settings used for every candidate/budget. |

This convenience loop uses default traversal; it does not search microbatch
sizes, chunk policies or orderings. For explicit orderings, loop over
`plan_program`. For build/profile candidate generation as well, use the
[recipe runner](../../../packages/workloads/src/shadowspill_workloads/recipes/README.md) or
[quickstart](../../../benchmarks/quickstart.md). See the
[budget-sweep example](../../examples/reusable-planning.md) for saving winners.
