# Artifacts and caches

Use `save(path)` and `load(path)` on `Program`, `Profiles`, `ProfiledProgram`,
`Plan` and `SweepResult` to pass results between phases. Each archive is
self-contained; a `ProfiledProgram` includes its Program, and a Plan includes
its selected code and applied profiles. Checkpoint tensors and compatible
runtime/provider libraries remain external. See the
[artifact API](api/artifacts.md#persistence-and-reports).

`write_report(result, directory)` writes human-readable tables and timelines.
Reports and disposable compiler caches are separate from saved phase artifacts.
Neither is required to load a Plan.

## Compiler caches

PyTorch and Triton manage their own caches. Their normal temporary locations
apply; ShadowSpill does not redirect them into a report or artifact directory.
Set `TORCHINDUCTOR_CACHE_DIR` or `TRITON_CACHE_DIR` explicitly only when an
alternate location is useful. Compiler intermediates can contain many small
files, so node-local temporary storage is preferable to cold storage.

Build finalizes task implementations and packages the required code in the
Program. A fresh execution worker restores that payload; it does not depend on
the build process's live compiler state. Build may use a temporary isolated
cache to replace an incompatible AOT entry, and removes that directory afterward.

## Internal content-addressed store

`shadowspill.store.ArtifactStore` supports lower-level capture/profiling adapters
and planner experiments. It is an optional cache, not a phase handoff mechanism.
The public phase APIs do not require an ArtifactStore.

```text
artifact_store/v<N>/
├── layout.json
├── README.md
├── build/
│   ├── graph_pairs/                 structural AOT graph pairs
│   └── optimizers/                  captured optimizer updates
└── planning/
    ├── programs/                   costed IR inputs
    ├── requests/                   search inputs and settings
    └── results/                    schedules, diagnostics and certificates
```

Entries use `<kind>/<digest[:2]>/<digest>/...`. Identity includes the inputs
that affect the result: graph structure, compilation choices, measurement setup,
or planning budgets and communication costs. Planner worker count affects wall
clock time, not the planning identity. Files use schema strings derived from
`shadowspill.schema.ARTIFACT_VERSION`; incompatible schemas are rejected.

The lower-level [planning JSON reference](planning-json.md) describes
`ShadowSpillProgram`, `ShadowSpillPlanningProblem` and `AnnotatedProgramPlan`.
The store's `artifacts()` method returns the files touched by that cache instance;
`diagnostics()` lists its roots. These are cache records, not fields of Plan.

## Roots and policies

`ArtifactStore.resolve(root, build_store=..., plan_store=...)` can place the
build and planning trees under separate roots. Omitting `root` uses
`~/.cache/shadowspill/v<N>`. `export_bypass_key` adds an explicit caller-provided
identity to adapters that consume it; it does not control compiler cache paths.

`build_store_mode` and `plan_store_mode` apply independently:

| Mode | Read | Write | On miss |
| --- | --- | --- | --- |
| `contribute` | Yes | Yes | Compute and save |
| `reuse` | Yes | No | Compute without saving |
| `require` | Yes | No | Raise with the missing key |
| `refresh` | No | Yes | Recompute and replace |

Entries are written atomically. A matching key with invalid content is an error,
not a reason to accept stale evidence. Do not edit content-addressed entries.
A planning experiment can share build data while keeping its search results in a
separate `plan_store`; this does not require copying compiler caches.
