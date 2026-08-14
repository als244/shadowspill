# Architecture

ShadowSpill separates framework capture, planning, deterministic simulation,
and memory execution.

```text
Program + profiles ──► PressureFit ──► ExecutionPlan
       │                     │
       └──────► Simulator ◄──┘
                                ExecutionPlan
                                     │
                                     ▼
PyTorch caller ──► task boundaries ──► neutral C runtime
       │                                  │
       └──────── launches compute         ├── named MemoryPools
                                          ├── directed TransferRoutes
                                          ├── fetch/evict lanes
                                          └── one worker thread
```

## Dependency rules

- IR records contain no framework objects, pointers, or backend handles.
- The simulator accepts an explicit schedule and never invokes the planner.
- The planner may call the simulator; the reverse dependency is forbidden.
- The runtime consumes an admitted execution plan without interpreting model,
  optimizer, or operation semantics.
- A frontend captures tasks, profiles executable ABIs, and binds framework
  storage at task boundaries.
- Pool storage, route copies, events, and profiler integration come from
  backend vtables. Neutral targets build and test without an accelerator SDK.
- `MemoryPool` owns only range geometry and causal lease ownership. Transfer
  actions own queueing, routes, copy submission, and object readiness; neither
  component writes the other's state machine.
- Models and operation libraries are clients and cannot become core
  dependencies.

## Runtime topology

`Runtime` is initialized explicitly before planning. It owns a registry of
named pools and directed routes. Each route has independent measured latency
and bandwidth; calibration publishes an immutable generation-tagged matrix.
Plans select one execution pool and one spill pool, take an exact matrix
snapshot, and record the selected fetch/evict profiles in `PlanReport`.

The initial implementation registers one accelerator execution pool and one
pinned-memory spill pool. The public pool/route representation does not assign
host or accelerator meaning to `MemoryPool` or `MemoryLease`; future peer,
remote-memory, and storage providers can implement the same contracts.

The expected first distributed deployment is one process and runtime per
execution device. Device and communication identity already exist in the IR,
so DDP, expert, pipeline, tensor, and context parallel work does not require
model-specific allocator behavior.

## PyTorch boundary

PyTorch owns compute streams and numerical dispatch. One narrow, version-pinned
adapter provides allocator callbacks and storage rebinding. The neutral runtime
owns allocation policy, object generations, readiness events, transfers,
failure propagation, and teardown.

One runtime context currently owns one worker thread and two route lanes. In
the NVIDIA provider these streams appear in NSYS as `shadowspill_fetch` and
`shadowspill_evict`; the worker appears as `shadowspill_worker`. Profiling names
and ranges use a neutral profiler vtable, with NVTX confined to the NVIDIA
implementation and a future ROCm implementation able to provide rocTX.

## PyTorch planning and lowering layout

The frontend cold path is organized around immutable artifacts rather than a
stateful planning coordinator:

```text
pytorch/
├── api.py             public plan_forward()/plan_step() orchestration
├── callables.py       lifecycle-owning planned callable objects
├── contracts.py       public errors and input/objective value contracts
├── guards.py          fixed-shape input signature validation
└── cache.py           central cache root, policy, and artifact ledger

pytorch/capture/
├── aot.py             Export and AOTAutograd capture boundaries
├── artifacts.py       immutable semantic graph artifacts
├── storage.py         offline alias/view/mutation storage contract
├── schema.py          dispatcher-schema normalization
├── fake.py            geometry-only FakeTensor construction
└── live_storage.py    live frontend tensor/view identity helpers

pytorch/partition/
├── api.py             short Stage-construction orchestrator
├── policy.py          built-in and custom contiguous partition policies
├── split.py           FX graph splitting and observed stage calls
├── provenance.py      boundary sources, mutations, and output projection
├── sources.py         representative root/stage value resolution
└── artifacts.py       Stage, StageExample, and PartitionedExport

pytorch/graph_pairs/
├── artifacts.py       arbitrary GraphPairVariant portfolios
├── build.py           portfolio construction policy
├── capture.py         occurrence → differentiated portfolio binding
├── rebind.py          occurrence-local value rebinding
├── repository.py      structural-ABI persistence and reuse
└── training.py        training-only composition after partitioning

pytorch/compilation/
├── compiler.py        stateless explicit-task executable construction
├── inductor.py        narrow version-pinned Inductor adapter
└── layout.py          semantic-contract/physical-layout reconciliation

pytorch/profiling/
├── inputs.py          deterministic value-bearing task inputs
├── metadata.py        value-sensitive profiling metadata
├── records.py         immutable keys, measurements, and allocation events
├── executables.py     warmed callable ownership and occurrence values
├── profiler.py        isolated CUDA timing/workspace orchestration
├── manifests.py       physical-manifest reconciliation and lookup
├── manifest_repository.py compiled-manifest persistence
├── runner.py          unique-ABI profile orchestration
└── repository.py      profile artifact serialization and lookup

pytorch/optimizer/
├── artifacts.py       optimizer tensor/task records
├── capture.py         readable validation → discovery → recurrent capture
└── staging.py         parameter ownership at semantic stage boundaries

pytorch/planning/
├── artifacts.py   immutable values passed between planning phases
├── forward.py     forward capture → profile → Program → PressureFit composition
├── training.py    training capture → profile → Program → PressureFit composition
├── repositories.py typed artifact-repository construction
├── admission/
│   ├── physical.py physical-cap checks and pool sealing
│   └── spatial.py exact annotated slab-timeline replay
├── reporting.py   PlanReport construction and lineage publication
└── common.py      validation, phase timing, and capacity calculations

pytorch/lowering/
├── catalog.py       canonical objects, aliases, and model registration
├── task_binding.py  one semantic/compiled task ABI → canonical objects
├── profiles.py      shared physical-layout and TaskProfile resolution
├── program.py       shared canonical Program publication
├── forward/
│   ├── program.py   readable partitioned-forward orchestrator
│   ├── objects.py   model and root-input registration
│   ├── tasks.py     stage binding and task emission
│   ├── residency.py initial/final residency derivation
│   └── artifacts.py immutable phase values
└── training/
    ├── program.py   readable accumulated-training orchestrator
    ├── objects.py   model/input/gradient/optimizer registration
    ├── bindings.py  stage boundaries and graph-pair variants
    ├── tasks.py     forward/backward/optimizer task emission
    ├── residency.py initial/final residency derivation
    └── artifacts.py immutable phase values

pytorch/materialization/
├── forward.py         forward model/input storage ownership
└── training.py        accumulated-training and optimizer storage ownership

pytorch/state/
├── model.py           public model relocation/externalization lifecycle
├── model_copy.py      payload-free module copy over runtime spill storages
├── optimizer.py       optimizer-state relocation/externalization
├── storage.py         generic tensor-storage/runtime-object operations
├── records.py         persistent storage, view, and source-owner records
└── registry.py        runtime-scoped persistent frontend ownership

pytorch/execution/
├── forward.py         ordinary forward task dispatch
├── training.py        centralized before/run/after training dispatch
└── records.py         immutable predecoded repeated-path records

pytorch/diagnostics/
├── plan.py            immutable PlanReport/PlanDiagnostics values
├── builders.py        deterministic report inventory construction
├── execution.py       immutable task/allocator/transfer timing records
└── step.py            StepResult and deferred DiagnosticsHandle

pytorch/runtime_adapter/
├── runtime.py         public PyTorch runtime configuration
├── bridge.py          neutral-runtime execution bridge
├── allocator.py       pluggable allocator installation
├── abi.py             declarative compiled ABI definitions
├── telemetry.py       allocation trace decoding
├── trace.py           runtime trace decoding
└── transfer_labels.py semantic fetch/evict labels
```

Partitioning ends when ordered `Stage` occurrences and their boundary
provenance exist. It has no AOTAutograd, graph-pair, profile, cache, or planner
dependency. Graph-pair construction is a subsequent training-only conversion:
it keys the geometry-specialized structural ABI and shares one arbitrary
portfolio of legal differentiation choices across equivalent stage
occurrences. Forward capture converts stage occurrences directly into
inference task artifacts and never enters the graph-pair package.

`plan_forward()` and `plan_step()` resolve public runtime/cache policy and then
call these phase functions in order. Every phase has a typed input and
immutable result, so tools may stop after Program construction, rerun
PressureFit for different capacities, or perform physical admission later
without reimplementing capture logic.

Semantic and physical authority remain separate. `lowering/task_binding.py`
binds offline task-storage contracts to the objects in
`lowering/catalog.py`. Both modes use the same profile/layout resolver and
Program publisher; only their real semantic differences live under
`lowering/forward/` and `lowering/training/`.
Compiler measurements supply extents, workspace, and time but never redefine
semantic identity. `pytorch/cache.py` is the only cache-policy module.
Artifact-specific repositories retain their domain names and serialization
contracts instead of introducing more generic `cache.py` modules. Cache code
cannot capture, compile, lower, plan, or admit a model—it only resolves or
persists artifacts at those explicit boundaries.

The package root deliberately contains no implementation catch-all such as
`public.py`, `runtime_bridge.py`, or `training_executor.py`. Its six files are
only public orchestration, public contracts/guards, cache policy, and stable
exports. Test-only numerical oracles live under `tests/`, not in the installed
frontend.
