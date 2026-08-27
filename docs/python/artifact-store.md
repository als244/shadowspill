# Artifact store

`artifact_store_dir` selects one content-addressed artifact store shared by
`plan_step()`, `plan_forward()`, `make_step_program()`, and
`pressurefit_program()`.

```text
artifact_store_dir/
├── pytorch/
│   ├── exports/          normalized Export archives and manifests
│   └── inductor/         PyTorch Inductor and Triton caches
├── graphpairs/           structural graph-pair graph pairs
├── profiling/
│   ├── compiled_manifests/
│   └── measurements/
├── pressurefit/
│   ├── programs/         canonical PressureFit inputs
│   ├── selections/       selected schedules
│   └── requests/         budget/bandwidth request indexes
└── plans/                readable request-to-artifact manifests
```

Schema-specific leaf directories may appear beneath these stable categories.
Callers should use manifests and artifact diagnostics rather than constructing
leaf paths.

## Identity

Artifact keys compose only the inputs relevant to that layer:

- Export: callable semantics, graph signature, fixed input geometry, and
  implementation revision.
- Graph pair: normalized stage semantic contract, differentiation options, and
  partition inputs.
- Compiled manifest: graph-pair contract, compiler/provider identity, and
  physical storage contract.
- Profile: compiled manifest, hardware, representative-value policy,
  `profiling_metadata`, and allocation-probe policy.
- PressureFit program: canonical `Program`, residency, capacity contract,
  transfer bandwidths, and options.
- Selection request: program digest, budgets, transfer bandwidths, and
  PressureFit options.
- Plan manifest: the complete planning request and all artifact dependencies.

`profiling_metadata` describes data-dependent measurement effects that are not
fully expressed by tensor geometry. For packed variable-length workloads, for
example, the same `[T, D]` activation can use metadata that distinguishes one
sequence from several shorter sequences. The value participates in profile
and downstream plan identity but is never passed into execution.

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
