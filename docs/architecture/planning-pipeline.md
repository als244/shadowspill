# The planning pipeline

Planning consumes a `ProfiledProgram`, memory budgets and communication
information. It selects a schedule and physical layout without building a model,
profiling tasks, initializing an accelerator driver or creating runtime pools.

```text
build_forward / build_forward_backward / build_step
    -> Program: tasks, storage, alternatives and executable payloads
measure_program, or explicit task profiles
    -> Profiles: durations and missing memory evidence
apply_profiles
    -> ProfiledProgram
plan_program
    -> Plan: selected tasks, schedule, layout and simulated timeline
Executor in a fresh worker
    -> execution
```

The [phase API](offline-pipeline.md) describes the public calls and artifacts.
Build owns stage partitioning, operator-level recomputation and task
implementations. Profiling supplies costs. Planning owns example traversal,
optimizer placement, stage save/recompute selection and transfers. Execution
loads the saved decisions and binds actual worker resources.

## Planning one candidate

```python
from shadowspill import MemoryBudgets, plan_program, write_report
from shadowspill.step import StepDataOrdering

plan = plan_program(
    profiled_program,
    memory_budgets=MemoryBudgets(
        execution_bytes=40 * 2**30, spill_bytes=128 * 2**30,
    ),
    communication_info=communication_info,
    ordering=StepDataOrdering(depth=2, breadth=4),
    optimizer_ordering="stage_interleaved",
)
plan.save("plan.ssp")
write_report(plan, "report")
```

`execution_bytes` is the total per-rank GPU allowance, including recorded fixed
overhead and reserved headroom. `spill_bytes` is a separate per-rank spill
capacity. The planner subtracts overhead before fitting task tensors and
workspace; it does not silently allocate extra GPU memory beyond the allowance.

The returned Plan exposes `summary`, `schedule`, `selections`, `timeline`,
`memory_layout` and `diagnostics`. Infeasible requests raise
`PlanInfeasibleError` or `PlanSearchExhaustedError` with diagnostics. Invalid
inputs and implementation errors are not converted into infeasible results.

## The walk through the microbatches

Build fixes the number and geometry of example calls, while planning chooses
their traversal. `StepDataOrdering(depth, breadth)` describes `depth` passes of
`breadth` examples, with their product equal to the Program's example count.
Within each pass, forward visits one stage across all examples before advancing;
backward visits stages in reverse. A breadth of one is depth-first traversal.

Two default flags reduce live activations: `pair_loss` runs each example's final
forward stage and its backward together; `reverse_breadth` visits examples in
reverse during backward. The first backward that reaches a parameter creates
its gradient; subsequent contributions use the accumulation form. That choice
follows traversal, not the original example index.

Changing traversal reuses the built task forms and compatible profiles. Model
mutations and RNG effects restrict reorderings when necessary. Optimizer
placement also happens during planning: stage-interleaved updates follow all
required gradient writes and old-weight reads; tail updates follow backward.
`build_forward_backward` Plans include both fresh-gradient and accumulate-into-
existing-gradient paths, admitted together. Fresh gradients remain the runtime
default.

## Search and physical admission

Save/recompute alternatives remain open until search chooses a legal assignment.
The search also chooses fetch, eviction and residency decisions under the supplied
capacities and costs. `search_options` configures the existing native planner and
simulator; it performs no build or measurement work.

Logical fit is not sufficient for execution. Physical admission assigns ranges,
checks allocation contracts and range-reuse dependencies, and re-simulates the
result. The certified layout travels with the Plan. Estimated memory supports
simulation only; it cannot produce an executable admission certificate.
`plan.simulation_only_reasons` reports missing evidence or executable payloads.
See [physical admission](physical-admission.md) for the placement mechanics.

`sweep_programs` is a convenience loop over explicitly supplied candidates and
budgets. It retains each successful Plan, diagnostic failures and a
`best_per_budget` lookup. Callers can write the same loop themselves, including
additional traversal choices. The helper does not invent candidates or recapture
models.

## Communication costs

`CommunicationInfo` contains host fetch/evict rates and latencies, optional
concurrent rates, topology, scale-up/out rates and measured collective timings.
It is separate from compute Profiles, so a saved ProfiledProgram can be replanned
under different communication assumptions.

`measure_communication` is an optional, independent GPU calibration API. Given a
Program it derives the relevant groups, dtypes, sizes and transfer forms; callers
may override the probes. It also records communication memory evidence. Planning
prefers matching measurements and estimates gaps from the supplied rates and
topology, recording the assumptions. Unknown memory is never silently zero.

Optimizer reductions and publication remain ordered with local updates on the
compute resource. Native fetch includes host-shard copy and all-gather; native
gradient eviction includes reduction and host-shard copy. Their completion
boundaries cover all constituent work. Simulation distinguishes collective bytes
and time from host traffic. Ordinary DP compute profiling can therefore run on
one representative GPU, while real calibration/execution uses the declared
participants.

## Persistence and reports

`Program`, `Profiles`, `ProfiledProgram`, `CommunicationInfo`, `Plan` and
`SweepResult` have explicit save/load methods. Each uses one archive containing
descriptive metadata and packed payloads. ProfiledProgram includes its Program
and applied profiles; Plan includes the computation and decisions needed to load
it independently of the original preparation directory. Weights/checkpoints and
compatible runtime/provider libraries remain external.

No artifact store is required to pass results between phases. Compiler caches
remain separate from saved phase artifacts. `write_report` is a CPU-only
presentation step that writes summaries, timelines and inspection data; saving
an artifact does not require generating its report.

Every execution worker loads a Plan for its fixed logical parallelism
configuration: one saved Plan on one kind of GPU, or Plans built for each GPU
that agree on what the ranks exchange. A complete worker-to-logical-rank map
binds physical workers at startup. Build/profile artifacts carry implementation
identities; execution validates relevant builds and memory evidence before
admission.

Previous: [Fixed-offset placement](fixed-placement.md).
See also: [training composition](training.md) and [phase API](offline-pipeline.md).
