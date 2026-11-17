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

`backward_program(step_program)` derives the matching forward/backward Program
from that same capture. It retains every fresh and additive form, the original
model state and producer-gradient geometry, and persists gradients in spill.
Optimizer state, controls, reductions and publications stay in the optimizer
path. Apply the same profiles and plan normally; the resulting Plan includes
fresh and accumulating modes. Jointly admit both derived Plans before execution.
Saved derivation metadata distinguishes the operation kind from its original
capture provenance. Neither derivation rebuilds the model or changes weight
initialization.

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

This supports complete-step, forward/backward and forward Programs. An expanded
forward Program returns a tuple of ordinary forward results, one per selected
example, including when the selected count is one. An ordinary `build_forward`
Program retains its single-result return value. Repeating a forward Program
orders examples and snapshots returned input/state views so later mutations do
not overwrite earlier results. Already expanded Programs can be selected again.

Missing fresh/additive forms or measurements require preparation; instantiation
does not compile or profile them. Replan each count and jointly admit its state
and pools before execution. Input-capacity and packing coverage checks are
separate: a count fitting device memory does not authorize padding beyond a
declared batch limit. Registered model mutation and RNG ordering constraints
remain in force.

The input-preparation primitives also expose `certify_coverage` in
`shadowspill_client.inputs.coverage`. It derives a conservative geometry proof
from an `InputSchema`, `BatchConfig`, `InputPlacement` and explicitly requested
`PackingSchedule` records. Its two constructions are a buffer large enough for
the entire allowed call, or reserved worst-case slots for independent Datums.
The first also preserves arbitrary atomic groups. Missing count/dimension bounds
or unsupported coverage fail with a diagnostic; successful sample packings are
insufficient. Scheduled padding must fit every declared scheduled capacity.

A certificate's `pack` method uses that construction without heuristic search.
Its `select` method tries the supplied admitted family in cost order and retains
the certified construction if bounded search fails. The witness must remain in
the family, and saved certificates rederive their proof when loaded. These are
input geometry guarantees; model padding/independence, compiled metadata bounds,
memory admission and collective compatibility still require separate preparation
checks. For service training, `training.validate_batch(batch).result()` checks
the actual batch against prepared layouts, including packing and scheduled
padding. Declared batch limits alone do not guarantee that every batch fits.

Installed `ModelDefinition` records can supply an `input_adapter(config, layout)`
function returning an `InputAdapter`. `ResolvedModel.bind_inputs(InputLayout(...))`
resolves it on CPU. The default handles independent, fixed-shape stacked fields;
packed, variable-shape or grouped inputs require an explicit adapter. `InputView`
can rename fields or reshape contiguous views without copying. Metadata fields
declare their physical shapes, dtypes, padding and value bounds; the engine
reserves their host storage before calling the adapter's fill function with
caller-order regions, groups and logical-worker context. Views of sliced inputs
retain their strides, and unsupported reshapes fail before allocating a copy.

The resulting input signature and staging bound are inspectable before model
construction. `padding_safe` describes the model-side assertion; objective
masking and loss ownership must still be qualified when binding the operation.
Adapters use ordinary functions, never a trainer loop or model-name branch.

`training.outputs.OutputLayout` binds each declared tensor output to physical
buffer and worker geometry. Its destination records retain caller Datum order;
its piece records remove record/token padding and describe leader, summed or
concatenated worker views. Alignment comes from declared schema symbols, with an
explicit source required when physical interpretations differ. Global tensors
also require a declared reduction across microbatches, separate from the worker
reduction. Declare these in `TensorOutput(source=("inputs", "tokens"), ...)`
and `TensorOutput(..., ordering="global", reduction="max")`, respectively.
They travel in the resolved output contract. Saved layouts recompute their
derived mappings and reject discrepancies. The mappings retain no GPU tensors
and support bounded artifact writes. Service results expose explicit
[output downloads](../advanced/service-client.md#results-and-saved-metadata).

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
[service preparation controls](../advanced/service-client.md#preparation-and-local-planning) or
[quickstart](../../../benchmarks/quickstart.md). See the
[budget-sweep example](../../examples/reusable-planning.md) for saving winners.
