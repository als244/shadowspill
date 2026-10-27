# Phase artifacts

The public pipeline passes inspectable artifacts between independent phases:

```text
build_forward / build_forward_backward / build_step → Program
measure_program                                   → Profiles
apply_profiles(Program, Profiles)                  → ProfiledProgram
plan_program(ProfiledProgram, ...)                 → Plan
Executor(Plan)                                     → execution worker
```

Import these names from `shadowspill`. Building finalizes task implementations
and may require a representative GPU. Measurement uses the ordinary PyTorch
allocator. Applying profiles, planning, inspecting archives and writing reports
require neither an accelerator nor ShadowSpill pools. Executor sets up pools in a fresh
worker. See the [phase architecture](../../architecture/offline-pipeline.md)
for the full API and current qualification limits.

## Program

`Program` describes one concrete invocation: input signatures, task alternatives,
storage and dependencies. Its fields include `devices`, `alias_groups`, `objects`,
`tasks`, `task_definitions`, `initial_residency`, `final_residency` and
`task_alternative_groups`. Task definitions have compatibility identities and
known workspace information, but no invented durations.

`name` and `sources` carry semantic module/parameter names alongside internal IDs.
`parallelism` declares logical ranks/groups independently of live process groups.
`setup` identifies the target and relevant builds. Finalized code is packaged in
`executables`; `bindings` records its portable input, output and initialization
requirements. Checkpoint tensors and live backend resources remain external.

Training builders accept either `example_args`, `example_kwargs` and
`num_examples`, or optional `example_calls=[(args, kwargs), ...]` for different
shapes/static metadata. Planning chooses example and optimizer ordering from the
same built Program; changing ordering does not require rebuilding it.

## Profiles and TaskProfile

`Profiles` is an editable collection for one hardware/runtime setup. Its `tasks`
map uses the Program's profile keys; each `TaskProfile` has plain numeric
`duration_seconds` and `workspace_bytes`, optional resource/overlap costs, and
ordinary metadata. `fixed_device_bytes` accounts for retained context/provider
memory and declared worker headroom. Measurement records allocation/lifetime
contracts and output aliases in addition to peak sizes.

```python
from shadowspill import Profiles, TaskProfile

replacement = Profiles(
    setup=profiles.setup,
    tasks={key: TaskProfile(duration_seconds=0.0072)},
)
profiles = profiles.merge(replacement)
profiles.save("profiles.ssp")
```

Later supplied fields overwrite earlier ones; omitted fields are retained.
Manually supplied values default to estimates. Estimated memory supports
simulation only. Timing estimates can reuse complete measured memory evidence.
`apply_profiles` requires every compute task's duration and unresolved memory;
it never silently fills missing values. Framework collective costs and resources
come separately from `CommunicationInfo` during planning.

## ProfiledProgram

`apply_profiles(program, profiles)` returns a read-only `ProfiledProgram`.
It contains `program`, `task_profiles`, `setup`, `fixed_device_bytes` and metadata.
Its `profiles` property gives an editable copy. Later edits to the original
collection cannot change the applied artifact.

Save/load includes both Program and profiles, without references to their
original folders. The same artifact can be planned against different budgets,
communication assumptions and supported task orderings.

## Plan

`plan_program` returns a `Plan` or raises a diagnostic planning exception.
The Plan exposes `program`, `profiled_program`, `memory_budgets`,
`communication_info`, `schedule`, `selections`, `timeline`, `memory_layout`,
`diagnostics`, and a computed `summary`.

`MemoryBudgets.execution_bytes` is the total per-rank GPU allowance, including
fixed overhead; `spill_bytes` is separate. `CommunicationInfo` carries topology,
rates/latencies and optional measured communication evidence. Compatible measured
costs take precedence; supplied rates/topology fill gaps.

`simulation_only_reasons` explains why a Plan cannot execute. Workspace peaks
alone do not prove physical allocation order or safe storage reuse. With complete
memory evidence, planning certifies a fixed layout and completion dependencies;
execution still checks its target, dependencies, resources and physical budget.
Estimated memory never qualifies a Plan for execution.

```python
from shadowspill import Plan, write_report

plan.save("training-plan.ssp")
loaded = Plan.load("training-plan.ssp")
print(loaded.summary)
write_report(loaded, "report")
```

The saved bundle includes task payloads, profiles, schedule, layout and logical
bindings. Model code and preparation caches are not needed to load it. Checkpoint
weights and compatible runtime/provider libraries remain external. Distributed
execution also requires compatible communication memory records and backend
bindings; a logical-rank declaration alone is not execution admission. See the
[phase API guide](../../architecture/offline-pipeline.md) for worker startup.

## SweepResult

`sweep_programs` is an optional CPU loop over named ProfiledPrograms and memory
budgets. `SweepResult.cases` keeps each candidate/budget's Plan or diagnostic
failure. `best_per_budget` selects the fastest successful simulation for each
budget. Users can also loop over `plan_program` directly.

## Persistence and reports

`Program`, `Profiles`, `ProfiledProgram`, `Plan`, `SweepResult` and
`CommunicationInfo` provide `save(path)` and `load(path)`. Each save produces one
archive containing a versioned JSON description and checksummed payloads.
Shared executable payloads are deduplicated within that archive. Inductor/Triton
caches are separate and never required to pass artifacts between phases.

`write_report(plan_or_sweep, directory)` is separate from persistence and is
CPU-only. It writes the simulated timeline and memory/budget summaries. No
runtime allocator or automatic profiling is invoked to render a report.

## Lower-level planning records

`ShadowSpillPlanningProblem`, `AnnotatedProgramPlan` and `TransferBandwidths`
serve planner experiments and diagnostic tools. Their [JSON reference](../planning-json.md)
is separate from the phase archives above.
