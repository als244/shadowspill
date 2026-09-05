# Artifact store

`artifact_store_dir` selects one content-addressed artifact store shared by
`plan_step()`, `plan_forward()`, `make_step_program()`, and
`pressurefit_program()`.

```text
artifact_store_dir/
└── v1/                   one tree per store format version
    ├── layout.json
    ├── README.md
    ├── pytorch/
    │   ├── exports/      normalized Export archives and manifests
    │   └── inductor/     PyTorch Inductor and Triton caches
    ├── graphpairs/       structural AOT graph pairs
    ├── profiling/
    │   ├── compiled_manifests/
    │   └── measurements/
    ├── pressurefit/
    │   ├── programs/     canonical PressureFit inputs
    │   ├── selections/   selected schedules
    │   └── requests/     budget/bandwidth request indexes
    └── plans/            readable request-to-artifact manifests
```

There is exactly one version for the store and everything in it:
`shadowspill.schema.ARTIFACT_VERSION`. It names the version directory, and
every stored file, and every structure embedded in one (programs, schedules,
plans, graph pairs, compiled manifests, profiles, selections, requests,
manifests), carries it in its schema string as `shadowspill.<kind>/v<N>`. It is
bumped whenever any stored structure changes, so one `artifact_store_dir` can
be kept across ShadowSpill updates: an update writes a fresh `v<N>` tree beside
the old one and replans, and nothing inside one tree is ever read as a stale
version. A file whose schema does not match inside a tree is corruption and
raises. Documents that live outside the store, such as step and plan
diagnostics, qualification results, and fixtures, version themselves. Callers
should use manifests and artifact diagnostics rather than constructing leaf
paths.

## Identity

Every artifact is found by a digest of its inputs, and each layer hashes only
what changes its own answer. Two rules decide what goes in.

**A key holds everything that changes the artifact.** If two requests would
produce different bytes, they must not share a key. This is why a profile
depends on the hardware and a plan depends on the bandwidths it was made for.

**A key holds nothing else.** Anything a consumer chooses, rather than
something the artifact is, stays out, or every such choice invalidates
everything downstream of it. The clearest case is a `PressureFitProgram`: it
records a problem, so a search *policy* is not part of it. Options are passed
by whoever plans, and the same saved program answers any of them.

| Artifact | Its key holds | Deliberately excluded |
|---|---|---|
| Export | callable semantics, graph signature, fixed input geometry, implementation revision | |
| Graph pair | normalized stage semantic contract, differentiation options, partition inputs | |
| Compiled manifest | graph-pair contract, compiler and provider identity, physical storage contract | |
| Profile | compiled manifest, hardware, representative-value policy, `profiling_metadata`, allocation-probe policy | |
| PressureFit program | the canonical `Program` and its measured task costs, initial and final residency, admission facts, the device and its simulated capacities and calibrated transfers, the capacity contract, and the pool budgets it was profiled under | `PressureFitOptions`. A program is a problem, not a search |
| Planned program | the program digest above, both residency lists, every device's capacity, both bandwidths and both latencies, the spill capacity, every `PressureFitOptions` field, and the admission and placement digests | nothing |
| Plan manifest | the complete planning request and all artifact dependencies | |

The planned-program key is built from the inputs *resolved for that call*, not
from what a saved program happens to record. A caller that overrides budgets or
bandwidths is asking a different question and gets a different key, which is
what lets one corpus of programs be planned across a sweep of budgets without
any point reading another point's answer.

`profiling_metadata` describes data-dependent measurement effects that are not
fully expressed by tensor geometry. For packed variable-length workloads, for
example, the same `[T, D]` activation can use metadata that distinguishes one
sequence from several shorter sequences. The value participates in profile
and downstream plan identity but is never passed into execution.

## What is saved beside a key

A stored artifact carries more than its key. The extra fields are provenance:
they record how the artifact came to exist, and they are never read back as
inputs, so adding one cannot invalidate anything.

- A `PressureFitProgram` records the budget and machine model in effect when
  it was collected. Planning it uses what the caller passes.
- A program corpus records, in each case manifest, the collection's runtime
  configuration, model, geometry, and seed, beside the program itself.
- A planned program records the options and resolved inputs it was searched
  under, which is what makes a stale entry detectable rather than silently
  reused: reading one re-derives the same fields and rejects a mismatch.

The distinction is worth keeping deliberately. A field that is hashed is a
question; a field that is only saved is a note about the answer.

`profiling_metadata` describes data-dependent measurement effects that are not
fully expressed by tensor geometry. For packed variable-length workloads, for
example, the same `[T, D]` activation can use metadata that distinguishes one
sequence from several shorter sequences. The value participates in profile
and downstream plan identity but is never passed into execution.

## Where each artifact lands

One rule covers every content-addressed artifact:

```text
<kind>/<first two characters of the digest>/<digest>/<document>
```

A directory named for the key, sharded so no directory grows unbounded,
holding one file per document. A kind that needs a second document later adds
it beside the first rather than inventing a path, and every store builds this
path through the same helper, `digest_directory`.

| Kind | Path under `v<N>/` |
|---|---|
| Export | `pytorch/exports/<2>/<digest>/` |
| Graph pair | `graphpairs/<2>/<digest>/graph_pairs.pt` |
| Compiled manifest | `profiling/compiled_manifests/<2>/<digest>/manifest.json` |
| Profile measurement | `profiling/measurements/<2>/<digest>/measurement.json` |
| PressureFit program | `pressurefit/programs/<2>/<digest>/program.json` |
| Selection request | `pressurefit/requests/<2>/<digest>/request.json` |
| Selection | `pressurefit/selections/<2>/<digest>/selection.json` |

The digest in a path is the key described above, so a path is a question and
its contents are the answer. A graph pair is the only entry that is not JSON,
because it holds compiled graphs; its key covers the structural contract and
the differentiation options together, so one entry is one digest like
everything else.

Two directories are deliberately not content-addressed, and both say why in
their names. `pytorch/inductor/` is PyTorch's own cache, laid out by PyTorch.
`plans/<qualified callable>/<program digest>/<plan digest>/` groups plan
manifests under the callable they were planned for, because a person reading
a store wants the plans for one model rather than a digest they would have to
compute.

## What each record contains

The four planning documents -- canonical `Program`, `PressureFitProgram`,
`StepProgram`, and `AnnotatedProgramPlan` -- are specified field by field in
[Program and annotated-plan JSON](planning-json.md). The rest of the store is
summarized here.

Every record names its own schema and, where it is content-addressed by a key
this store computed, repeats that key so a file found at the wrong path is
detected rather than trusted.

**Compiled manifest** records what compiling one structural task produced:

```text
schema, graph_digest, profile_key_digest, manifest{
  compatibility_digest, semantic_contract_digest,
  storage_contract, optimized_storage_contract,
  root_allocations, contract_capture_ns }
```

The two storage contracts are what a later profile and admission bind
against; `compatibility_digest` is what a task is matched by.

**Profile measurement** records what running that task cost:

```text
schema, key_digest, measurement{
  runtime_ns, samples_ns, timing_half_drift,
  allocation_contract, allocation_trace, allocation_path_observations,
  persistent_extent_bytes, output_input_bindings,
  representative_inputs, provenance, phase_timings_ns, profiling_wall_time_ns }
```

`runtime_ns` with its samples is the measured cost the planner schedules
against. The allocation contract and trace are what physical admission
replays, and `provenance` records the hardware and policy the measurement was
taken under.

**Selection request** is the question put to PressureFit:

```text
schema, program_digest, initial_residency, final_residency,
simulation, admission, options
```

**Selection** is the answer, keyed by that request:

```text
schema, key_digest, program_digest, initial_residency, final_residency,
simulation, admission_digest, options, selections, schedule,
resident_slice, diagnostics
```

`selections` is the task-alternative choice per group and `schedule` the memory
schedule it implies. Reading one re-derives the request fields and rejects a
mismatch, which is what makes a stale entry an error rather than a silent
wrong answer.

**Plan manifest** is the readable record of one planning call, beside the
`execution_plan.json` it produced:

```text
schema, mode, model, capture_identity,
execution_device, execution_pool, execution_budget_bytes,
allocation_probe_seeds, allocation_probe_repetitions,
implementation_revision, execution_plan_digest, initial_execution_plan,
artifacts, phase_timings_ns
```

`artifacts` lists every store entry the call depended on, which is how a plan
is traced back to the profiles and programs behind it.

## Cache policy

| Argument | Behavior |
|---|---|
| `save_plan=True` | Persist artifacts and readable manifests. |
| `force_fresh=True` | Do not read cached artifacts; use isolated compiler caches. |
| `overwrite_plan=True` | Replace matching saved artifacts; requires both `save_plan=True` and `force_fresh=True`. |
| `implementation_revision="..."` | Invalidate compiler/profile identity when an implementation changes without changing graph semantics. |

Export is performed on each planning call so Python objective and signature
semantics are freshly validated. A matching Export archive is retained as
evidence; it is not treated as permission to skip capture.

When `artifact_store_dir` is omitted, the package uses a user cache location.
Long-running or reproducible work should pass an explicit local-filesystem
directory. Network filesystems are unsuitable for compiler caches and
high-frequency atomic artifact publication.

## Plan diagnostics

`PlanReport.diagnostics.cache_artifacts` records every managed, matched, read,
or written artifact with its category, kind, digest, absolute path, schema,
and dependency digests. Cache directories are also recorded, so a report is a
complete provenance index for the planning call.
