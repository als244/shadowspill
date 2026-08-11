# Canonical intermediate representation

ShadowSpill exchanges three immutable records between capture, simulation,
planning, and execution. They contain logical identities and measured costs,
but never tensors, callables, pointers, streams, events, or backend handles.

## `Program`

`Program` describes work that may be planned. Declaration order assigns stable
dense identities and is therefore part of the scheduling contract.

- A device has a process identity, backend kind, and local index.
- A resource selects a device, a compute/communication/control kind, and a
  lane. A communication operation is an ordinary task rather than an
  optimizer-specific exception.
- An alias group is the unit of residency and versioning. Object records are
  byte extents within it, so views do not become independently transferable.
- A task names its resource, structural profile, dependencies, object edges,
  mutations, and whether a frontend entrypoint is required.
- A recomputation group lists mutually exclusive task sets. Selection resolves
  the program to a conventional DAG by removing inactive tasks and their
  dependency edges.

The current wire schema is `shadowspill.program/v1`.

## `MemorySchedule`

`MemorySchedule` contains declared initial residency, an ordered action stream,
and required final residency. An action occurs after its trigger task:

- `release` removes a device generation without retaining its contents;
- `offload` copies the authoritative generation to host and releases device
  residency;
- `prefetch` creates a device generation from authoritative host contents.

Validation replays the selected task and action order. Every input alias must
be device-resident when its task begins. This makes invalid planner output a
construction error rather than a runtime recovery path.

The current wire schema is `shadowspill.memory_schedule/v1`.

## `ExecutionPlan`

`ExecutionPlan` binds one program, recomputation selection, memory schedule,
frontend entrypoints, physical admission, and simulator prediction. Admission
states the complete device and host budgets and accounts separately for
context, provider headroom, the slab, workspace reserve, host reservation, and
predicted fragmentation.

`digest` identifies the complete record. `scheduling_digest` intentionally
excludes entrypoint and physical-admission metadata so descriptive frontend
changes do not invalidate schedule caches.

The current wire schema is `shadowspill.execution_plan/v1`.

## Serialization and compiled projections

`to_json()` emits sorted compact JSON with integer byte and nanosecond fields.
`from_json()` validates the schema and all cross-record references. Re-encoding
a valid record is byte-identical.

`project_dense`, `project_dense_schedule`, and
`project_dense_execution_plan` convert string identities to stable integer
indices and flattened offset/value arrays. These lossless projections define
the data handed to compiled simulator, planner, and runtime boundaries; the
compiled components do not invent or reorder identities.
