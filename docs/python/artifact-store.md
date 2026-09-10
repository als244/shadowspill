# Artifact store

One content-addressed store, shared by `plan_step()`, `plan_forward()`,
`build_step_program()`, and `plan_program()`. It holds two independent trees:

```text
artifact_store/
└── v1/                   one tree per store format version
    ├── layout.json
    ├── README.md
    ├── build/            what a run pays for, and another run can reuse
    │   ├── exports/      normalized Export archives and manifests
    │   ├── inductor/     PyTorch Inductor and Triton caches
    │   ├── graphpairs/   structural AOT graph pairs
    │   ├── optimizers/   traced recurrent optimizer updates
    │   └── profiling/
    │       ├── compiled_manifests/
    │       └── measurements/
    └── planning/         what a run decided
        ├── programs/     the canonical ShadowSpillProgram each planning call was given
        ├── requests/     what each search was asked for
        ├── results/      its answer: resolution, schedule, diagnostics
        └── plans/        the ExecutionPlan a callable runs, with its lineage
```

A build writes nothing under `planning/`, and a planning call nothing under
`build/`. That is what lets one build store serve many runs that each keep
their own plans. The program archive is in the planning tree for the same
reason: a planning call archives the program it was handed, so a plan's
lineage points at an immutable copy of what was planned, and that copy is
evidence about the planning rather than something a build produced.

There is exactly one version for the store and everything in it:
`shadowspill.schema.ARTIFACT_VERSION`. It names the version directory, and
every stored file, and every structure embedded in one (programs, schedules,
plans, graph pairs, compiled manifests, profiles, selections, requests,
manifests), carries it in its schema string as `shadowspill.<kind>/v<N>`. It is
bumped whenever any stored structure changes, so one `artifact_store` can
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
everything downstream of it. The clearest case is a `ShadowSpillPlanningProblem`: it
records a problem, so a search *policy* is not part of it. Options are passed
by whoever plans, and the same saved program answers any of them.

| Artifact | Its key holds | Deliberately excluded |
|---|---|---|
| Export | callable semantics, graph signature, fixed input geometry, implementation revision | |
| Graph pair | normalized stage semantic contract, differentiation options, partition inputs | |
| Compiled manifest | graph-pair contract, compiler and provider identity, physical storage contract | |
| Profile | compiled manifest, hardware, representative-value policy, `profiling_metadata`, allocation-probe policy | |
| `ShadowSpillPlanningProblem` | the canonical `ShadowSpillProgram` and its measured task costs, the role, initial and final residency, admission facts, the device and its simulated capacities and calibrated transfers, and the capacity contract | `SearchOptions`. A program is a problem, not a search |
| Planned program | the canonical `ShadowSpillProgram` digest, both residency lists, every device's capacity, both bandwidths and both latencies, the spill capacity, the admission and placement digests, which search ran, and everything that search was told | `workers`, which changes how long an answer takes and not which answer is right; and the plan handed in as the one to beat, which is provenance rather than the question |

The plan manifest is the one document not found by a key. It is filed by the
callable it was planned for, and it names the whole request and every artifact
the call depended on, so it is where a plan is read rather than looked up.

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

- A `ShadowSpillPlanningProblem` records the budget and machine model in effect when
  it was collected. Planning it uses what the caller passes.
- A program corpus records, in each case manifest, the collection's runtime
  configuration, model, geometry, and seed, beside the program itself.
- A planned program records the options and resolved inputs it was searched
  under, which is what makes a stale entry detectable rather than silently
  reused: reading one re-derives the same fields and rejects a mismatch. It
  also records the plan the search was handed to beat, which never enters the
  key: a run that replans a budget without that plan in hand must still read
  back the answer the sweep chose.

The distinction is worth keeping deliberately. A field that is hashed is a
question; a field that is only saved is a note about the answer.

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
| Export | `build/exports/<2>/<digest>/exported_program.pt2` |
| Graph pair | `build/graphpairs/<2>/<digest>/graph_pairs.pt` |
| Optimizer capture | `build/optimizers/<2>/<digest>/optimizer_capture.pt` |
| Compiled manifest | `build/profiling/compiled_manifests/<2>/<digest>/manifest.json` |
| Profile measurement | `build/profiling/measurements/<2>/<digest>/measurement.json` |
| Canonical program | `planning/programs/<2>/<digest>/program.json` |
| Selection request | `planning/requests/<2>/<digest>/request.json` |
| Planned program | `planning/results/<2>/<digest>/selection.json` |

The digest in a path is the key described above, so a path is a question and
its contents are the answer. The program archive is the one entry keyed by
something it wraps rather than by a composite: its digest is `ShadowSpillProgram.digest`,
because the archive's job is to hold one immutable copy of each distinct
program a plan can be traced back to. Graph pairs and optimizer captures are
the entries that are not JSON, because they hold compiled graphs and traced
tensors; a graph pair's key covers the structural contract and the
differentiation options together, so one entry is one digest like everything
else.

Two directories are deliberately not content-addressed, and both say why in
their names. `build/inductor/` is PyTorch's own cache, laid out by PyTorch and
subdivided by `implementation_revision`.
`planning/plans/<model class>/<capture identity>/<plan digest>/` groups plan
manifests under the qualified name of the class they were planned for, because
a person reading a store wants the plans for one model rather than a digest they
would have to compute. The last two segments are the leading sixteen characters
of each identity.

## Rooting the two trees apart

| Argument | Effect |
|---|---|
| `artifact_store` | Roots both trees under one directory's own `v<N>/`. `None` uses a user cache location. |
| `build_store` | Roots `build/` elsewhere, overriding `artifact_store` for that tree. |
| `plan_store` | Roots `planning/` elsewhere, overriding `artifact_store` for that tree. |

The common shape is a build store several runs read and a plan store each run
keeps to itself, so runs share the capture, compilation and profiling they
paid for while each searches every point itself rather than reading back a
plan another run found. A plan store gets its own `layout.json` naming the
artifact store it was searched over.

`build_step_program()` takes only `artifact_store` and `build_store`: it
produces a program and plans nothing. `plan_program()` takes only
`artifact_store` and `plan_store`: it plans a program it is given and builds
nothing.

Long-running or reproducible work should pass an explicit local-filesystem
directory. Network filesystems are unsuitable for compiler caches and
high-frequency atomic artifact publication.

## Store modes

`build_store_mode` and `plan_store_mode` each say what this run does with one
tree: whether it reads what is there, and what it does about what is not.
A mode sets all four gates at once -- read, write, overwrite, refuse a miss --
so no combination that means nothing can be asked for.

| Mode | Reads a hit | On a miss |
|---|---|---|
| `contribute` (default) | yes | builds it and writes it back |
| `reuse` | yes | builds it and persists nothing |
| `require` | yes | refuses, naming the mode that would allow it |
| `refresh` | no | rebuilds and overwrites what was there |

`reuse` is what makes a shared store safe to read from many runs at once, and
`require` is what makes one a fixed reference: two results compared against a
`require` store are known to have stood on the same artifacts rather than on
whatever each rebuilt.

The two trees take separate modes because they are shared for different
reasons. A build artifact is what a run *paid for* and any run may reuse; a
plan is what a run *decided*, and two runs comparing planners must not read
each other's. So a run can keep its plans to itself and still contribute the
builds it paid for.

`build_store_mode` also decides where PyTorch compiles. Only `contribute`
points Inductor and Triton at the store's own cache. Every other mode runs
them in a private temporary directory, with the process-local compiler caches
cleared on the way in and out, so an earlier plan in this process cannot
serve an entry the mode was told not to read. `refresh` publishes what it
built there back into the store afterwards; `reuse` and `require` discard it.

`implementation_revision` is the other invalidation control. It marks the
lower-level implementations a build was measured against, so a kernel change
that does not change the exported graph still gives compiler and profile
artifacts a new identity. It also names the Inductor cache subdirectory, so a
fresh revision starts a fresh compiler cache. It reaches a plan only through
the profiles behind it: a planned program's key does not name it.

Export runs on every planning call, so Python objective and signature
semantics are freshly validated. A matching Export archive is retained as
evidence; it is not permission to skip capture.

## What each record contains

The four planning documents -- canonical `ShadowSpillProgram`, `ShadowSpillPlanningProblem`,
`StepProgram`, and `AnnotatedProgramPlan` -- are specified field by field in
[program and annotated-plan JSON](planning-json.md). The rest of the store is
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
  runtime_ns, samples_ns,
  timing_relative_mad, timing_half_drift, timing_unstable,
  workspace_requested_bytes, workspace_charged_bytes, workspace_extent_bytes,
  persistent_extent_bytes,
  allocation_contract, allocation_trace, allocation_path_observations,
  output_input_bindings, representative_inputs,
  provenance, phase_timings_ns, profiling_wall_time_ns }
```

`runtime_ns` with its samples is the measured cost the planner schedules
against, and the three `timing_*` fields say how much to trust it. The
allocation contract and trace are what physical admission replays, and
`provenance` records the hardware and policy the measurement was taken under.

**Selection request** is the question a search was put:

```text
schema, program_digest, initial_residency, final_residency,
simulation, search, search_options, admission, incumbent
```

**Planned program** is the answer, keyed by that request:

```text
schema, key_digest, program_digest, initial_residency, final_residency,
simulation, search, search_options, admission_digest, incumbent,
schedule, selections, resident_slice, diagnostics
```

`search` names the algorithm and `search_options` is everything it was told,
both halves in full, so a plan searched over one candidate space is never read
back for another. The search that ships carries its `resolution_options` there,
the shares of the flexible groups to recompute as exact fractions (`"1/4"`).
`selections` is the task-alternative choice per group and `schedule` the memory
schedule it implies. Reading one re-derives the request fields and rejects a
mismatch, which is what makes a stale entry an error rather than a silent wrong
answer. `incumbent` is provenance in both documents and in neither key: it
names the plan the search was handed to beat.

**Plan manifest** is the readable record of one planning call, beside the
`execution_plan.json` it produced:

```text
schema, mode, model, capture_identity,
execution_device, execution_pool, spill_pool,
execution_budget_bytes, spill_budget_bytes,
requested_dynamic_scratch_reserve_bytes,
allocation_probe_seeds, allocation_probe_repetitions,
implementation_revision, execution_plan_digest,
execution_plan, initial_execution_plan,
artifacts, phase_timings_ns
```

`execution_plan` and `initial_execution_plan` are the filenames beside the
manifest, the second `null` when the plan has no distinct first step.

`artifacts` lists every store entry the call depended on, which is how a plan
is traced back to the profiles and programs behind it.

## Plan diagnostics

`PlanReport.diagnostics.store_directories` names the roots this call used as
name/path pairs -- `root`, `build`, `build.inductor`, `planning`, `plan_store`
-- and `cache_artifacts` records every artifact it touched with its category,
kind, digest, absolute path, schema, and dependency digests. Each carries one
disposition:

| Access | Meaning |
|---|---|
| `read` | Its bytes were loaded and used as planning authority. |
| `matched` | It agreed with a freshly produced in-memory value, which was used instead. |
| `write` | This call produced it. |
| `improved` | A planned program this call replaced: a request handed a plan faster than the one on record searched again and won. |
| `managed` | A directory owned by another component, such as the Inductor cache. |

Together they make a report a complete provenance index for the planning call.
