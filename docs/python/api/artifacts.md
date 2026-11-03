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
requirements. `resources` describes provider resources by importable factory,
options and logical groups. Checkpoint tensors and the live resources those
factories return remain external.

Training builders accept either `example_args`, `example_kwargs` and
`num_examples`, or optional `example_calls=[(args, kwargs), ...]` for different
shapes/static metadata. Planning chooses example and optimizer ordering from the
same built Program; changing ordering does not require rebuilding it.

`bindings["task_metadata"][key]["numerical_contracts"]` lists versioned numerical
lowering contracts applied during compilation. An empty list means no additional
native scalar-rounding rule was applied. Composed tasks include contracts from
their children. A precision-emulated sigmoid-backward contract identifies the
provider's intermediate rounding and multiplication order for BF16/FP16.
These contracts participate in implementation identity; they are inspection
metadata, not a runtime setting for changing an already measured task.

`bindings["task_inputs"][key]["provenance"]` describes representative input
policies. Authentic control, constant and optimizer-hyperparameter samples use
`value` for small JSON values or `value_bytes` for larger arrays. `value_bytes`
is base64-encoded contiguous tensor bytes; dtype and shape come from the matching
tensor geometry, and restoration checks the byte count. These samples support
kernel finalization and standalone measurement. They do not initialize runtime
user inputs or embed model parameters. Caller-supplied profiling inputs override
the sample under the normal measurement API.

### Axis task metadata

For a chunked task definition, `Program.bindings["task_metadata"][key]["row_chunks"]`
records its resolved axis composition. The serialized names `row_chunks` and
`rows` denote a generic chunk contract and logical axis extent. `key` is the
task definition's `profile_key`,
not a stage occurrence or a microbatch index. The record has:

| field | meaning |
|---|---|
| `schema` | Axis-composition metadata version. |
| `graph` | Full explicit graph's structural compatibility digest. |
| `rows` | Full logical axis extent for this parent task. |
| `chunk_size` | Requested maximum axis positions per child call. |
| `domain` | Optional logical axis name for propagation between execution regions. |
| `inputs`, `outputs` | Flat axis layouts: `{dim, before, after}`, `null` for shared values, or `"sum"` for additive contributions. An input `"sum"` is a shared-output cotangent supplied once, with zeros in later children; an output `"sum"` accumulates all child contributions. |
| `rules` | Opaque operator contracts: `target`, `version`, `output_axes` and `reductions`; a derived axis stores `input_index` and `trailing_dims`, a factored axis stores `dim`, `before`, `after`, and a local scratch declaration stores `dim`. |
| `children` | Pairs of actual axis extent and structural child digest for ordinary calls. When fresh backward gradients are accumulated, this contains only the first-chunk specialization; otherwise it includes each required size and uneven tail. |
| `accumulation_leaves` | Ordered fresh shared backward output positions passed as running gradient inputs after the first chunk. Empty for forward tasks and tasks whose gradients already alias prior inputs. |
| `accumulating_children` | Pairs of actual axis extent and child digest for calls after the first chunk. Each receives ordinary sliced inputs followed by the running gradients in `accumulation_leaves` order, mutates those buffers, and preserves output order. Only invoked sizes are recorded. |
| `child_regions` | Optional nested execution contracts inside bounded slices. `creators` and `accumulators` each contain pairs of actual slice extent and the execution-composition record described below. These expose each inner region's own bounds and releases. |
| `storage_contract` | Physical full-output storage digest, including view offsets. |
| `output_literals` | Pairs of flat output position and portable scalar value; tensor outputs follow the storage contract and absent leaves remain `null`. |

The parent has one ordinary profile key. Its measured duration, workspace and
allocation evidence cover all chunks and assembly. A child digest is a build
reuse identity, not independently usable timing or memory evidence. Save and
recompute choices remain stage alternatives; chunk count does not change the
invocation's example count. These fields describe execution geometry and do
not force boundary residency or prescribe transfers.

For nested execution, `task_metadata[key]["execution_regions"]` contains:

| field | meaning |
|---|---|
| `schema` | Execution-composition metadata version. |
| `graph` | The uncomposed outer graph's structural digest. |
| `children` | Ordered structural child digests; a child can itself be an axis composition. |
| `regions` | Ordered instructions. `inputs` and `outputs` name local value slots; `release` drops slots after their last local use. `boundary` records the child's exact output storage. Optional `row_chunks` describes that region's selected chunk loop. |
| `outputs` | Outer output bindings, each a local `slot` or a portable `literal`. |
| `slots` | Number of local binding slots, including outer inputs. |
| `boundary` | Outer storage contract, input storage offsets, output count and scalar output literals. |
| `input_domains`, `output_domains` | Per-port maps from a logical domain name to `{dim, before, after}`. These retain known coordinate mappings across execution cuts and forward/backward saved-value bindings. |
| `domain_rules` | Declared opaque operator contracts per named domain, including versions and output mappings. These support coordinate propagation through whole regions and are inherited by backward and nested compositions. They do not select execution chunking or residency. |

A boundary record's `storage` is a task storage contract. `input_offsets` are
captured element offsets; `output_count` and `output_literals` describe the flat
output structure. Executable packaging also records observed
`host_output_roots`. Slots describe lifetimes inside one measured task; they
are not planner object identifiers and cannot request independent spilling.

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

`write_report(result, path)` is separate from persistence and is CPU-only. It
writes the simulated timeline and memory/budget summaries into the directory
`path`. No runtime allocator or automatic profiling is invoked to render a
report.

### AOT graph-pair cache

The frontend's `GraphPairStore` persists structural graph pairs separately from
finalized Programs. Its entry digest and `manifest.json` include
`metadata_policy: "source_precision"`, identifying the source-provenance and
precision-annotation contract. A mismatched entry requires capture under the
requested contract.
The manifest's `retention` records `save` and `recompute` policy identities,
with custom-operator classifications under `operators.save` and
`operators.recompute`; `retention_digest` keys their capture compatibility.
Each policy identity records `memory_bound_flops_per_byte`, includes `default`
when it is `retain` or `recompute`, and includes nonempty `rules` sorted by target
and optional module pattern. Each rule records `target`, `action` (`retain` or
`recompute`) and `module_pattern` when scoped. Scoped policies add
`retention.scoped.save` and/or `retention.scoped.recompute`: each contains
`rules` with target, action and resolved `source_indices`, plus per-source
custom `operator_occurrences`. These structural decisions participate in cache
identity before an entry is reused. A module path's occurrence name alone does
not prevent reuse when the same pattern selects the same structural positions.
Each variant records its `regenerated_operators` and `unknown_operators` from
the actual captured backward. These describe the cut; they are not residency
assignments or timing estimates. The variant's `memory_budget` labels the base
endpoint (`1.0` for save, `0.0` for recompute); zero can include retained additions
and is not a claim of zero internal saved bytes. Actual saved tensors and task
profiles determine memory admission.
The `emulate_precision_casts` boolean is also part of the entry key and manifest;
it records the requested pointwise rounding policy during differentiation.

Each serialized FX node retains its `name`, `op`, `target`, `args` and `kwargs`,
plus these explicit optional fields:

| Field | Meaning |
| --- | --- |
| `source_operation` | Source operation's zero-based `index`, FX `op` and `target`. The index counts source operations, excluding argument/output placeholders. `null` means no source attribution. |
| `partitioner_tag` | AOT partitioner's phase tag, such as `is_forward` or `is_backward`; `null` when no tag exists. |
| `low_precision_pointwise_barrier` | Boolean requiring the compiler to preserve the low-precision rounding boundary at this node; defaults to `false`. This participates in graph compatibility. |
| `saved_outputs` | Output-node records with flattened output `position`, AOT saved-value `index`, and `check_version` boolean. They identify saved-for-backward outputs whose precision the compiler must preserve for internal consumers. Defaults to an empty tuple and participates in graph compatibility. |

Restoration reconstructs the source tag in node `custom` metadata, the phase
tag in node `partitioner_tag` metadata, and enabled rounding barriers in node
`low_precision_pointwise_barrier` metadata. Saved-output records reconstruct the
saved-for-backward descriptors in output-node `desc` metadata; unsaved positions
receive neutral descriptors. Source module stacks resolve from the current
occurrence before policy selection; they are not persisted as names
belonging to a representative layer. Arbitrary node metadata and live tensors
are excluded. Code compatibility depends on graph structure and contracts,
not these occurrence-local names.
Saved-output positions exclude effect-token outputs removed during capture;
the corresponding ordered effects stay inside the task graph.

## Lower-level planning records

`ShadowSpillPlanningProblem`, `AnnotatedProgramPlan` and `TransferBandwidths`
serve planner experiments and diagnostic tools. Their [JSON reference](../planning-json.md)
is separate from the phase archives above.
