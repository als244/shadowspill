# Phase artifacts

The public pipeline passes inspectable artifacts between independent phases:

```text
build_forward / build_forward_backward / build_step → Program
measure_program                                   → Profiles
apply_profiles(Program, Profiles)                  → ProfiledProgram
plan_program(ProfiledProgram, ...)                 → Plan
Executor(Plan)                                     → execution worker
```

Optional [pool-memory calibration](pool-memory.md) produces a portable
`PoolMemoryProfile` for the exact spill capacity and compatible hardware/software.
GPU UUID, hostname and driver version record provenance without restricting
Plan portability to those exact values.
Pass it as
`pool_memory` to planning or candidate sweeps; the Plan retains both the original
process budget and the separately measured physical-device allowance.

Import these names from `shadowspill`. Building finalizes task implementations
and may require a representative GPU. Measurement uses the ordinary PyTorch
allocator. Applying profiles, planning, inspecting archives and writing reports
require neither an accelerator nor ShadowSpill pools. Executor sets up pools in a fresh
worker. See the [phase guide](../phases.md) for the full API and artifact contracts.

Artifact `load` also accepts `max_metadata_bytes` and `max_payload_bytes`. These
limits check declared decompressed sizes before reading archive contents; they
bound serialized bytes rather than decoded Python object overhead. Ordinary phase
artifacts have unbounded defaults. The
[training bundle](../advanced/definitions.md#preparation-and-capture) defaults to 8 GiB of metadata and 8 GiB of
deduplicated binary payloads, with explicit per-load overrides. All archives reject
duplicate JSON keys and nonfinite JSON values, in addition to verifying checksums
and the exact member inventory.

## Contents

- [Program](#program)
- [Profiles and TaskProfile](#profiles-and-taskprofile)
- [ProfiledProgram](#profiledprogram)
- [Plan](#plan)
- [SweepResult](#sweepresult)
- [Persistence and reports](#persistence-and-reports)
- [Lower-level planning records](#lower-level-planning-records)

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
contracts applied during compilation. Composed tasks include contracts from
their children. A precision-emulated sigmoid-backward contract identifies the
provider's intermediate rounding and multiplication order for BF16/FP16.
A precision-emulated SiLU-backward contract identifies FP32 internal arithmetic,
round-to-nearest division, and explicit fused multiply-add for supported
BF16/FP16/FP32 implementations. With `emulate_precision_casts=True`, these
operations remain eligible for one fused pointwise kernel. The contract changes
affected compiler and profile identities; it does not imply bitwise agreement for every
operation, provider, or chunk geometry.
The `torch.numerical_settings.v1:` entry contains a JSON snapshot of the caller's
backend settings. It participates in graph-pair cache, compiler-cache,
implementation and profile identity. These records describe an already built
task; editing them does not change its implementation.

### Numerical backend settings

ShadowSpill does not enable strict arithmetic or determinism by default. It
inherits the caller's PyTorch and provider configuration, including precision
emulation. Leave normal training settings in place for throughput measurements;
enable stricter settings explicitly for the numerical comparisons that need
them. Deterministic kernels can change both task time and workspace, so profiles
and memory budgets measured under one configuration do not describe the other.

Configure PyTorch before building, and use the same settings in finalization,
profiling and every execution worker. ShadowSpill records and validates:

- deterministic-algorithm and warn-only flags;
- device matrix multiplication precision, including the `highest`/`high`/
  `medium` control and the `fp32_precision` control;
- BF16/FP16 reduced-precision reduction, their split-K controls, FP16
  accumulation and preferred BLAS library;
- cuDNN enablement, determinism, benchmarks/benchmark limit, and convolution
  and RNN FP32 precision;
- `CUBLAS_WORKSPACE_CONFIG`, `CUBLASLT_WORKSPACE_SIZE`,
  `NVIDIA_TF32_OVERRIDE` and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` environment values.

ShadowSpill keeps the caller's settings unchanged. A mismatch identifies the
required and actual values and refuses the task. Restore the build configuration
or build, profile and plan again under the new configuration. Worker startup and
each `Executor.run()` validate the selected tasks before input staging or task
execution. Keep these process-global settings fixed until an asynchronous
invocation completes; this is not per-thread precision isolation. Resource-group
build/profiling and distributed execution coordinate configuration refusals.

Set environment variables before starting the process, since libraries may cache
them during initialization. This snapshot does not inventory every provider flag,
SDPA backend selector or workspace API override, nor guarantee bitwise equality
across different GEMM/chunk geometries. Such changes still require fresh build
and measurement and the appropriate numerical qualification. CPU loading and
planning carry these settings as plain data and never import PyTorch to check them.
An artifact without a snapshot is read without configuration validation;
rebuilding it records one.

### Representative task inputs

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

Task timing takes at least three samples and continues until the requested device
time is covered and median variability/drift pass their thresholds, subject to
wall-time and maximum-sample limits. The minimum count takes precedence over
the wall limit: three four-second invocations still take twelve seconds. Tiny
tasks stop at the count cap instead of filling the duration target at any cost.
The [profiling API](profiling.md) documents all controls and stop reasons.
Allocation validation separately
observes one cold invocation and three warmed invocations; reducing the timing
minimum does not remove those checks. Profiles retain actual sample counts,
durations and stability diagnostics. Task metadata also separates wall time for
cold observation, warmed observations, validation, conditioning, timing and
layout reconciliation. Each allocation observation reports execution/collection,
decoding and initial synchronization time.

Saved compiler memory descriptions separately record `allocation_behavior`:
static wrapper geometry, exact opaque call names and unresolved behavior. Fixed
row/region compositions inherit every child's requirements. An opaque provider
call is not an allocation guarantee, and observed traces alone do not prove a
bound over all admitted tensor values. These records describe evidence; they do
not replace measured profiles or physical Plan admission.
Prepared training captures additionally save an `allocation_coverage` record.
CPU rechecks require the complete capture identity, all compiled forms and exact
installed provider contracts. Unknown allocation behavior fails preparation even
if one measured run fits. Ordinary low-level profiling remains an observed-path
API; the stronger requirement applies to prepared training families.

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

`TaskProfileCache` shares compatible task measurements across candidates and
processes. Direct callers can share one explicitly, including across runs:

```python
from shadowspill import TaskProfileCache, measure_program, apply_profiles

cache = TaskProfileCache("/local-ssd/shadowspill-cache")
profiled_candidates = [
    apply_profiles(program, measure_program(program, cache=cache))
    for program in candidate_programs
]
```

A cache hit requires the finalized task digest (including selected generated
kernels and provider preparations), hardware/toolchain setup, tensor geometry
and aliases, representative input policy and producer lineage, numerical
contracts, resources/rank topology, allocator configuration and measurement
policy to match. More samples can satisfy a smaller requested minimum. Different
chunk implementations or kernel choices are distinct entries even when their
original graphs match. Caller-supplied `inputs` disable reuse for that call.

The cache saves raw task timing and complete allocation evidence. It validates
that evidence on load. Tasks that retain anonymous cold allocations or lack an
external-memory measurement are measured again. Every candidate still loads its
executables and provider resources and accounts for its own streams, retained
data and worker headroom. Cached external context memory supplies a conservative
floor; aggregate `fixed_device_bytes` is never copied from another candidate.
Planning, physical memory admission and overlap calibration remain independent.

`TaskProfileCache(root, mode="refresh")` ignores preexisting disk entries while
reusing new measurements within that cache instance. `reuse` reads without
writing; `require` refuses a miss. Default `contribute` reads hits and writes
misses. Pass one cache with `mode="refresh"` to remeasure a whole candidate
sweep. Distributed participants intersect their cache hits before
launching any task collectives. `Profiles.metadata["task_cache"]` records hits
and misses; original sample counts and timing diagnostics stay with each task.
Cache directories are optional; saved public Profiles remain self-contained.

`measure_overlap(profiled_program, communication_info, device=device,
rank_map=..., stages=None, inputs=None, trials=None, cache=None)` returns a `Profiles` overlay containing
paired task/collective progress measurements. Call it on all participating ranks
after communication calibration, then merge and apply it:

```python
from shadowspill import OverlapProfileCache, measure_overlap, apply_profiles

cache = OverlapProfileCache("/local-ssd/shadowspill-cache")
# Reuse this instance across the candidate sweep on each participating rank.
overlap = measure_overlap(profiled, communication, rank_map=rank_map, cache=cache)
profiles = profiled.profiles.merge(overlap)
profiled = apply_profiles(profiled.program, profiles)
```

The overlay preserves isolated task durations, allocation contracts and fixed
memory evidence. It restores saved code with ordinary allocations, measures each
unique form at up to ten of each route's actual transfer sizes, spread evenly in
log bytes from its smallest to its largest, and records paired
trial statistics in `TaskProfile.metadata["overlap_calibration"]`. It does not
build, create execution pools, fit full-step timings or choose a Plan. Both save
and recompute forms and complete composed/chunked tasks are measured as built.

Admission releases unused allocator blocks once per task, allocates one buffer
large enough for the admitted routes, then warms the task. Timing preserves that
allocator workspace. Compatible isolated task and transfer baselines are reused
within the live communicator session. By default, probes take three trials when
their predicted work together fits 100 ms, otherwise one conditioned trial.
`trials=N` requests a positive fixed count. One trial estimates progress; it does
not establish repeatability. Repeated trials must also pass the dispersion check.
The isolated transfer reference uses one warmed eight-transfer burst per actual
route, shared across task lengths and trial counts. Its first transfer is excluded
from the median. Paired bursts still cover the task's interference window; those
can be longer. This avoids repeating long transfer-only bursts for every task.

`OverlapProfileCache` stores supported task/route evidence under
`profiling/overlap/`, independently of ordinary task measurements. Reuse requires
matching finalized code, representative input lineage, numerical settings,
allocator/stream policy, rank mapping, topology, backend, route geometry and
sampling protocol. All ranks agree on identical cached evidence before skipping
collectives. Fully cached tasks skip input preparation and executable restoration.
Explicit `inputs` disable reuse; missing/unsupported probes are never negative
cache entries. The `contribute`, `reuse`, `require` and `refresh` modes have the
same meanings as `TaskProfileCache`. No candidate-wide memory evidence is reused.

The overlay records cache hits/misses and per-rank preparation, input/dispatch,
admission, buffer, warmup and probe wall times. Trial timing separates observer
setup, enqueue, drain and Python event decoding. Drain time includes GPU waiting
and native event conversion; these host intervals must not be added to device
durations as independent costs. Cached observations retain their original trial
statistics; cache metadata distinguishes those from newly measured work.
Native allocator counters also record device-allocation/free calls and retries
around each newly measured trial, outside its timed device interval.

Supported rates apply only while compute and the corresponding collective phase
actually overlap. Route identity includes operation, dtype, ordered groups,
in-place layout, topology and backend setup. Recalibrate when those change.
Calibration excludes the first collective and tails after local compute when
estimating collective progress; unsupported or unstable trials remain missing.
Insufficient memory and tasks that communicate are explicitly reported as skipped.
Programs with model-owned collective resources and heterogeneous task forms are
not measured by this calibration. Host-copy interference is not inferred.
No sharded collective transfers means an empty overlay requiring no GPU work.

Service preparation performs this enrichment during communication measurement
and retains the overlap evidence in saved Profiles/ProfiledProgram before CPU
planning. Direct callers supplying CommunicationInfo can attach compatible
overlap profiles with `apply_profiles`.

## ProfiledProgram

`apply_profiles(program, profiles)` returns a read-only `ProfiledProgram`.
It contains `program`, `task_profiles`, `setup`, `fixed_device_bytes` and metadata.
Its `profiles` property gives an editable copy. Later edits to the original
collection cannot change the applied artifact.

Save/load includes both Program and profiles, without references to their
original folders. The same artifact can be planned against different budgets,
communication assumptions and supported task orderings.

## Plan

[`plan_program`](planning.md) returns a `Plan` or raises a diagnostic planning exception.
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
[phase API guide](../phases.md) for worker startup.

## SweepResult

`sweep_programs` is an optional CPU loop over named ProfiledPrograms and memory
budgets. `SweepResult.cases` keeps each candidate/budget's Plan or diagnostic
failure. `best_per_budget` selects the fastest successful simulation for each
budget. Users can also loop over `plan_program` directly. See [planning options](planning.md#compare-programs-and-budgets).

## Persistence and reports

`Program`, `Profiles`, `ProfiledProgram`, `Plan`, `SweepResult` and
`CommunicationInfo` provide `save(path)` and `load(path)`. Each save produces one
archive containing a versioned JSON description and checksummed payloads.
Shared executable payloads are deduplicated within that archive. Inductor/Triton
caches are separate and never required to pass artifacts between phases.

Prepared training families and local Plan proposals use the same archive envelope
with compact binary metadata. Shared records are encoded once and referenced by
index, preserving sharing when loaded. Checksums cover the metadata and binary
payloads; loading remains CPU-only. Use the artifact methods to inspect or copy
these values rather than depending on their on-disk representation.

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
The `numerical_settings` field carries the same versioned backend snapshot as
compiled task metadata. Differentiation can select a different autograd formula
under changed settings even when its input forward graph is identical. Both
memory and disk graph-pair caches therefore require matching settings.

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
serve planner experiments and diagnostic tools. Their [JSON reference](../advanced/planning-json.md)
is separate from the phase archives above.
