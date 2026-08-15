# Planning Cache

ShadowSpill uses one inspectable cache root for PyTorch compiler artifacts,
graph pairs, task measurements, PressureFit inputs and outputs, and resolved
plans. Pass it explicitly to `plan_step()` or `plan_forward()`:

```python
model = relocate_model_state(
    model, runtime=runtime, pool="spill", release_source=True
)
train_step = plan_step(
    model,
    ...,
    planning_cachedir="/mnt/planning-cache/shadowspill",
)
```

The directory can contain artifacts for multiple models, input geometries,
workload classes, budgets, software revisions, and devices. Leaf artifacts
are content addressed; readable names under `plans/` are indexes, not cache
identities.

## Layout

```text
planning_cachedir/
├── README.md
├── layout.json
├── pytorch/
│   ├── exports/v1/<prefix>/<export-digest>/
│   └── inductor/<implementation-revision>/...
├── graphpairs/
│   └── v4/<prefix>/<structural-task-abi>/<selection-digest>/
├── profiling/
│   ├── compiled_manifests/v2/<prefix>/<compiler-key>.json
│   └── measurements/v15/<prefix>/<profile-key>.json
├── pressurefit/
│   ├── programs/v1/<prefix>/<program-digest>/program.json
│   └── selections/v3/<prefix>/<selection-key>.json
└── plans/
    └── <model-name>/<capture-prefix>/<execution-plan-prefix>/
```

Every successful callable exposes the exact paths touched by its planning
call:

```python
for artifact in train_step.plan_report.diagnostics.cache_artifacts:
    print(artifact.category, artifact.access, artifact.path)
```

`access` is `read`, `write`, `matched`, or `managed`. `matched` means a fresh
in-memory semantic artifact had the same identity as an existing entry.
`managed` identifies PyTorch's opaque Inductor cache directory.

## Identities

The cache uses a different key at each conversion boundary:

| Artifact | Identity |
|---|---|
| Export archive | Canonical Export graph, signature, tensor geometry, aliases, mutations, and PyTorch runtime |
| AOT graph-pair portfolio | Structural task ABI, differentiated root positions, terminal-unit-tangent specialization, and portfolio schema |
| Compiled manifest | Compiled task ABI, compiler/device environment, and `implementation_revision` |
| Task measurement | Compiled task ABI, compiler/device environment, representative-value policy, `implementation_revision`, and `profiling_metadata` |
| Program | Canonical objects, tasks, profiles, alternatives, and resource identities |
| PressureFit selection | Program, initial/final residency, simulator capacities/routes, and planner options |
| Execution plan | Selected Program variants, schedule, entrypoints, and physical admission |

The structural task ABI describes the tensor and graph interface presented to
AOTAutograd: normalized FX topology and static arguments, tensor geometry and
gradient requirements, alias groups, and explicit mutations. Differentiated
root positions and terminal-unit-tangent specialization complete graph-pair
lookup identity. It excludes tensor values, stage occurrence IDs, task names,
timing profiles, runtime budgets, and physical allocation sizes. Equivalent
stage occurrences therefore share one persisted portfolio.

## Value-sensitive profiling

`profiling_metadata` distinguishes data distributions that can change measured
kernel cost without changing the compiled tensor ABI. For packed sequences,
both examples below have activation geometry `[4096, D]`, but may execute
value-sensitive kernels differently:

```python
profiling_metadata=[
    {"sequence_lengths": [4096]},
    {"sequence_lengths": [512] * 8},
]
```

Training requires one metadata value per example microbatch position. Forward
planning accepts one value. Values must be JSON-compatible and are
canonicalized before hashing.

The metadata is key-only. It is never passed to the model, compiled task,
allocator, or runtime. Concrete `example_inputs` still provide the actual
`seq_lens` tensor used by task-local profiling. The metadata prevents a
measurement collected for one value distribution from being reused for a
different distribution. The selected measurement then naturally affects the
Program and PressureFit result.

## Freshness and invalidation

- `save_plan=True` writes generated artifacts and the readable plan index.
- `force_fresh=True` reads no ShadowSpill or PyTorch compiler artifacts. The
  call compiles into an isolated, initially empty PyTorch cache; when
  `save_plan=True`, that newly populated cache is published for the next call.
- `overwrite_plan=True` requires both `force_fresh=True` and `save_plan=True`;
  it replaces entries at an existing lookup identity with fresh evidence.
- `save_plan=False` uses a temporary PyTorch compiler directory and writes no
  planning artifacts.

ShadowSpill reruns Export on every planning call. Export validates current
Python graph semantics and signatures, while later content-addressed stages
can still be reused.

A custom operation's schema may remain unchanged when its lower-level kernel
implementation changes. Export cannot detect that invisible change. Supply a
stable package version, build ID, or source revision through
`implementation_revision`; changing it creates a new compiler/profile
namespace without changing graph-pair identity:

```python
train_step = plan_step(
    model,
    ...,
    implementation_revision="mlops-0.8.1+git.a17c4ef",
)
```

Planning caches contain executable compiler artifacts and Python graph-pair
serialization. Treat the cache as trusted local build data; do not load a
cache supplied by an untrusted party.
