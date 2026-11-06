# Preparing external operator kernels

`OperatorPreparation` lets an operator provider attach its selected binaries to
a Program. Pass descriptions through `operator_preparation` on `build_forward`,
`build_forward_backward`, or `build_step`:

```python
from shadowspill import OperatorPreparation, build_step

program = build_step(
    model,
    objective=objective,
    optimizer=optimizer,
    example_args=examples,
    operator_preparation=(
        OperatorPreparation("my_provider.prepared:factory", {"implementation": "v1"}),
    ),
)
```

The factory is supplied by the provider. The default empty tuple preserves the
existing build behavior. This interface does not discover or prepare arbitrary
external compilers automatically. A provider must qualify the entire dispatch
surface it claims, including generated epilogues, tails, accumulation forms,
helper kernels and native launchers.

## Contract

The importable factory accepts keyword arguments `phase`, `device` and `options`.
`options` contains finite JSON values. The builder opens the provider before
graph capture and compilation; the loader opens it before task measurement or
the executor's fixed-layout Plan admission and final physical-budget seal.

For `phase="build"`, return an object with:

- `capture()`: a context manager yielding provider-owned capture state. It
  surrounds the task's existing kernel-finalization invocation. It must not
  invoke the task itself or retain its activation tensors after export.
- `export(captured)`: return `(manifest, blobs)`. The manifest contains finite
  JSON values; blobs maps SHA-256 digests to immutable bytes. Capture and export
  must identify the implementations actually selected, including warm cache
  hits. Benchmark candidates must not masquerade as selected implementations.
- `close()`: release build-owned state. Constructors must clean up their own
  partial initialization if they raise before returning an owner.

For `phase="load"`, return an object with:

- `bind(manifest, read_blob)`: validate source, target, ABI, specialization and
  binary integrity; load and initialize the executable modules; return a task
  binding. `read_blob(sha256)` reads the Program's archived immutable bytes.
  Missing or incompatible assets must refuse, without compiling, tuning or
  falling back to an eager cache. Share modules across compatible task bindings
  owned by this loader. Loading cannot execute the tensor computation.
- `close()`: release owned modules and host state after execution streams have
  been quiesced. Process-lifetime resources such as Python native extensions
  require an explicit, validated sharing contract in the provider.

A task binding supplies `activate()`, a context manager for dispatching only
through that task's validated implementations. Activation may not load, compile
or benchmark. It must preserve numerical behavior and the ordinary operator's
live device/stream queries. Missing specializations refuse even if another task
has warmed the process's eager caches. Bindings may be activated on executor
threads; use task-local state rather than unscoped process-global cache changes.

ShadowSpill hashes the factory module. Providers must additionally record and
validate transitive source dependencies and their own compiler/runtime contract;
the factory-module hash alone is not that contract.

The contract covers implementations managed by the provider. Prebuilt framework
libraries can have their own first-use initialization, such as native PyTorch
or cuBLAS kernels. Their external memory remains part of profiling and physical
admission; preparing one provider does not certify all task code free of library
initialization or eliminate every source of first-invocation latency.

## Artifact, measurement and execution behavior

The Program stores manifests under `bindings["operator_preparation"]`, keyed by
unique executable/profile key. `operator_binaries` holds digest-addressed blobs,
deduplicated in Program, ProfiledProgram and Plan archives. Neither local build
cache paths nor warm caches are required to recover archived binaries. Existing
Programs without preparation retain their serialized field layout.

Each task's compatibility digest includes its provider descriptions, factory
source and selected manifests. A changed implementation requires corresponding
measurement; saved profiles cannot silently describe different binaries.
Program and Plan loading remains CPU-only and does not invoke factories. Device
preparation occurs explicitly during measurement or executor startup.

Profiling loads all the Program's prepared tasks, because control-input
reconstruction can use tasks outside a requested stage subset. Retained PyTorch
allocations are included in fixed device bytes; non-PyTorch module/context
memory is included through the existing external-device observation. Each
executor first installs its allocator and opens the cold pool, reserving the
profiles' measured mature external footprint, which includes provider modules.
It then loads only its selected task keys, including continuing-accumulation
and named Plans, before admitting their fixed layouts and sealing the physical
budget. Loading before allocator installation would initialize PyTorch CUDA too
early and is refused. Shared loader ownership lasts
until the executor drains work and removes its task callbacks. Loading or closing
while a prepared task is active is refused.

Checkpoint cuts, execution regions, arbitrary-axis chunks, saved/recompute graph
policies, and PressureFit residency remain independent controls. Each candidate
build captures its own selected implementations; planning compares measured
costs and admits the resulting physical memory layout. Preparation does not
alter gradient initialization/accumulation or replay expensive computation an
additional time. Logical distributed ownership is unchanged: each rank loads
its own selected artifacts on its assigned local device, and preparation
failures participate in the existing coordinated startup error protocol.
