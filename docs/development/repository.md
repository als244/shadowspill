# Repository structure and validation

## Top-level layout

See [workload organization and adding a model family](workloads.md) for the
structure inside `packages/workloads` and its extension pattern.

```text
shadowspill/
├── packages/client/       lightweight service/training clients and portable records
├── packages/engine/       installable engine distribution
│   ├── src/shadowspill/   Python phase APIs and runtime bindings
│   ├── csrc/             native planner, runtime, backends and adapter
│   ├── CMakeLists.txt    native build entry point
│   └── pyproject.toml    engine metadata and dependencies
├── packages/workloads/    optional models, objectives and registered setup hooks
├── examples/              public-client demos and training applications
│   ├── demos/             short runnable clients with explicit training settings
│   ├── extensions/        standalone extension package
│   └── training/          shared corpus readers and CLI application loops
├── tools/                 trace analysis, source checks and corpus applications
├── tests/                 repository and cross-package checks
├── qualification/         numerical/performance campaigns and independent references
├── docs/                  architecture, APIs, guides and development
├── scripts/               environment setup
└── pyproject.toml         workspace lint, type, and test configuration
```

`qualification/` owns the complete gate implementation and its documentation.
There is no forwarding package under `tools/`. `qualification/gates/`
runs the selected gates; the numerical and performance gates' `matrix/`
packages select and launch their cases, and the SSD and remote gates run those
same matrices; `run.py` implements the single-case command. Product behavior
remains under `packages/engine/src/shadowspill/`. The numerical gate is a package,
`qualification/numerical/`, holding one case end to end: the
request and its identity, the reference arm, the planned run, the two
comparisons, the artifact, the verdict, and the matrix that launches cells.
The throughput gate is the package beside it,
`qualification/performance/`, holding one cell in the order it
runs: the manifest and its budgets, the readings taken off the runtime, the
phases (calibrate, plan, warm, measure), and the verdict each gate reaches.

`shadowspill_workloads` supplies optional architectures, registered components,
input/objective implementations and reusable numerical policies. Qualification budgets and
acceptance rules live under `qualification/`. Build accepts ordinary user models
and call examples. The public [training client](../python/client.md) submits global
batches to the installed service, which owns preparation and execution. Generic
saved-phase preparation lives in `shadowspill.training.preparation`, and bounded
initialization/copy helpers live in `shadowspill.checkpoint`. The catalog installs
independently from `packages/workloads`; custom components need no catalog.

`shadowspill_client` supplies portable settings and Datum records with NumPy as
its only required dependency. Install `./packages/client` independently of the
engine or catalog; see its [installation guide](../../packages/client/README.md).

Within the engine, `server/http.py` owns authenticated, bounded socket exchanges.
`server/routes/` separates artifact transfers, run creation and saved Plan names,
request submission/polling, and metrics/reports. These routes call the existing
owners in `server/records/` and `training/`; they contain no model construction or
training loop. Server startup remains separate from the live request path.

A library adapter belongs to the package that uses the library: the workloads
package's `shadowspill_workloads.integrations` translates MLOps operator facts.
Operator libraries own their output-axis, reduction, mutation and layout facts;
adapters translate those facts into generic build settings. Core compilation,
planning and execution do not discover libraries or select policies by their
names. Architecture documentation describes these generic mechanisms; setup and
examples for a particular library belong in its integration guide.

`shadowspill.orchestration` owns installed process infrastructure used by the service,
benchmarks and developer tools. Its launch, capture and monitoring modules take
explicit worker counts and rank maps.
The engine imports no repository clients or model catalog. See the
[ownership map](../../packages/engine/src/shadowspill/orchestration/README.md).

`tools/planning/` groups GPU corpus collection and CPU planner evaluation.
`tools/measure_plan/` measures an existing executable Plan; `tools/stage_traces/`
inspects individual compiled stages. Ordinary training and hypothetical memory
sweeps use the public APIs shown in the client demos and Advanced guides.
Training-system comparisons live in the separate `shadowspill_baselines` project.

## Python package

```text
packages/engine/src/shadowspill/
├── artifacts/             portable Program, Profiles, ProfiledProgram and Plan archives
├── build/                 capture, compile and lower finalized task definitions
│   ├── compilation/       package task code and physical storage contracts
│   │   └── inductor/      version-specific lowering, aliases and allocation geometry
│   └── lowering/          bind compiled tasks into portable IR
│       ├── forward/       forward objects, tasks and residency
│       └── training/      boundaries, all backward forms and optimizer templates
├── communication/         calibration and reusable communication information
│   └── binding/           resolve logical groups into calibration participants
├── execution/             saved-plan workers, initialization and bounded tensor export
│   ├── diagnostics/       no-update startup calls, state preservation and reports
│   ├── recording/         opt-in task/transfer recording for Executor
│   ├── distributed/       collective and shard-storage execution bindings
│   └── observations/      optional bounded runtime norm collection
├── orchestration/         installed process launch, capture and phase diagnostics
│   ├── launch/            process commands, rank/device isolation and host groups
│   ├── nodes/             assigned hardware inventory and compatible device selection
│   ├── capture/           external profiler launch and runtime capture boundaries
│   └── monitoring/        worker phase journals
├── checkpoint/            streamed checkpoint format over Executor tensor access
├── parallelism/           logical groups, parameter storage and input placement
├── resources/             portable provider descriptions, setup and cleanup
├── planning/              CPU-only profile binding, scheduling and physical layout
│   ├── admission/         resolve saved memory evidence without a live pool
│   ├── training/          bind example/update orderings and gradient completion
│   ├── memory_search/     search schedules and device/spill residency
│   │   ├── admission/     prove physical placement and reuse dependencies
│   │   │   └── layout/    fixed offsets, lease lifetimes and certificates
│   │   ├── diagnostics/   search evidence, counters and summaries
│   │   ├── algorithms/    concrete memory search implementations
│   │   │   └── pressurefit/
│   │   │       └── candidates/  native problem records and decoded results
│   │   └── toolkit/       shared search traversals and graph-pair resolution
│   ├── plan_store/        request identity, validated cache I/O and search reuse
│   └── serialization/     neutral planning artifact encode/decode
├── operator_preparation/  provider binary capture and saved task binding
├── reporting/             CPU-only reports and timeline/summary output
├── server/                service storage and transport boundaries
│   ├── startup/           preparation, restored state and worker startup
│   ├── records/           durable bounded journal and committed-prefix recovery
│   └── routes/            run, request, artifact and observation HTTP adapters
├── ir/                    framework-neutral values and indexed projections
│   └── program/           the program: its vocabularies, specs, alternatives,
│                          and the checks it must satisfy
├── step/                  microbatch ordering descriptions for planning
│   └── ordering.py        StepDataOrdering: the walk through the microbatches
├── diagnostics/           what one traced step turned out to cost, as
│   │                      immutable evidence, and the reading of it
│   ├── step.py            the summary, the two timelines, and the clock each
│   │                      is read against
│   ├── timing.py          what one invocation measures about itself: the
│   │                      markers, the armed records, the invocation timelines
│   ├── collection/        task/transfer joins, native evidence and timing summaries
│   ├── transfer_phases/   event matching, lane regions and payload throughput
│   └── occupancy/         pool attribution, clocks and timeline rendering
├── profiling/             ordinary-allocator timing and memory measurement
│   └── observation/       allocator observations and lifetime decoding
├── task/                  one task's shape and provenance, framework-free
│   ├── storage.py         the storage contract: roots, output views, mutations
│   ├── allocations/      allocator events and the stable task contract
│   ├── profiles.py        task timing, memory and allocation evidence
│   ├── inputs.py          what a task argument is, and the value profiled for it
│   ├── slots.py           where a task's objects sit in its contract
│   ├── layout.py          the compiled task's layout, and what a transition costs
│   └── manifest.py        what a compiled task promises, and the digest of the
│                          compilation behind it
├── store/                 content-addressed artifacts, and the modes that gate
│   │                      reading and writing them
│   ├── artifacts.py       the store itself
│   └── policy.py          StoreMode and the four gates it implies
├── serialization/         compact typed binary records for saved artifacts
├── simulator/             the simulator and diagnostic timeline
│   └── indexing/          the template, one schedule bound onto it, the result
│                          decoded back, and the buffers all three hand to C
├── network/               pool configurations whose memory is on another
│                          machine, and the library that serves them
├── frontend.py            the twelve methods a framework frontend implements,
│                          and nothing else
├── runtime/               one runtime in this process, framework-neutral:
│   │                      bootstrap, core, configuration, plan, occupancy,
│   │                      residue, calibration, the admission replay, the
│   │                      trace and telemetry it exposes, the markers a
│   │                      caller times with, and the failures it reports
│   ├── abi/               the ctypes projection of the C API, one module per
│   │                      record kind
│   ├── objects/           runtime objects: the reference, and how one is made
│   ├── plan/              worker bindings, admission, lifecycle and optional tracing
│   │   └── layout/        certified offsets and completion dependencies
│   └── timing.py          an instant recorded on a stream, and the time between two
├── training/              service preparation, sessions and worker commands
│   ├── components/        public model/objective functions and CPU schema resolution
│   ├── inputs/            owned global input validation and packing
│   ├── session/           confirmed operation boundaries and participation
│   ├── worker/            bounded commands and fixed prepared worker bindings
│   └── preparation/       resolve setup and compose the existing phase APIs
│       ├── configuration/ schemas, limits, policies and layout candidates
│       ├── capture/       build bindings, templates and allocation checks
│       ├── phases/        build/profile/communication/plan process jobs
│       ├── planning/      concurrent searches and compatible Plan families
│       ├── artifacts/     saved bundles, measurements and validation
│       └── offline/       local sweeps and submitted schedule reconciliation
└── pytorch/
    ├── capture/           Export/AOT capture and semantic storage contracts
    │   ├── artifacts/     graph records, geometry, identity and output schemas
    │   ├── gradients/     accumulation, precision and padded gradient storage
    │   └── storage/       one task's contract: the records, the roots, the
    │                      capture, the symbolic replay
    ├── chunking/          axis contracts, explicit graph slicing and portable loops
    ├── partition/         stage policies, splitting, provenance, authentic controls
    ├── graph_pairs/       differentiation alternatives by structural contract
    ├── tuning/            finalized provider choices, saved and replayed per task
    ├── operators/         runtime-importable task operations, independent of capture
    ├── optimizer/         capturable optimizer updates: discovery, sandbox,
    │                      bindings, trace and task definitions
    ├── accelerator.py     the device PyTorch drives, named once
    ├── bindings.py        who holds a range, in PyTorch's terms, and detaching them
    ├── frontend.py        PyTorch's answers to what the runtime cannot do itself
    ├── runtime.py         a runtime opened with that frontend
    ├── runtime_adapter/   PyTorch's half of the task boundary: the operators
    │                      that rebind a plan's leases onto storages
    └── representations.py physical components of wrapped tensors
```

`planning` implements the public Plan and sweep phases. Its `memory_search`
subpackage solves the lower-level schedule and memory residency problem.
`operator_preparation` saves finalized provider kernels for later execution; it
is separate from the training preparation workflow.

`server/startup` coordinates creation and resumption. `configuration.py` owns
deployment settings, `initialization.py` resolves model/checkpoint sources,
`workers.py` binds ranks, and `publication.py` writes preparation artifacts.
`creation.py` keeps admission, cancellation and retirement in one resource owner.

High-level modules expose small orchestration functions. Detailed algorithms
live in domain-named submodules; generic `utils`, `helpers`, and `common`
modules are avoided.

## The C tree

[csrc/README.md](../../packages/engine/csrc/README.md) draws the layout of `csrc/`: the
public headers, the one library under `src/` (common, simulator, planner,
runtime with a directory per runtime component), the backends, and the
PyTorch adapter.

Everything under `csrc/src/` builds into one library. The backends share one
`CMakeLists.txt`, which builds the mock backend always and every provider
whose toolchain is installed, or the providers named in `SHADOWSPILL_BACKENDS`
(`-DSHADOWSPILL_BACKENDS=<provider>`, or through pip
`--config-setting cmake.define.SHADOWSPILL_BACKENDS=<provider>`). The PyTorch
adapter has its own `CMakeLists.txt` and links no provider: it loads a backend
library by name at bootstrap. Provider dependencies stay inside
`backends/<provider>/`.

A private header named `internal.h` belongs to the directory holding it and is
included by path from anywhere else, so `"internal.h"` always means this
directory's. Each one compiles on its own, and a source includes the narrowest
one that covers what it touches.

## Tests

The test tree mirrors the boundary under test, all the way down: a test for
`packages/engine/src/shadowspill/profiling/` lives in
`packages/engine/tests/engine_tests/profiling/`.

```text
tests/
├── integration/    application checks across installed packages
├── tools/          reusable source-tool tests
├── qualification/  gate runner and acceptance-rule tests
└── repository/     packaging, naming, documentation and boundary checks
```

Engine and workload tests live under their respective package roots. The engine
keeps native and fresh-process canaries under `tests/engine_tests/native` and
`tests/engine_tests/integration`; [test ownership](../../tests/README.md) describes
the complete layout. Root `conftest.py` and `pyproject.toml` configure all three
test roots, including dtype selection and subprocess imports.

Long numerical and throughput runs belong in `qualification/`, not the unit
suite.

## Setup

```bash
./scripts/setup.sh
```

The script creates `.venv`, installs PyTorch with the machine accelerator
backend, builds and installs the library, its backends and the adapter,
installs development dependencies and those of the training harness, runs the
workloads package's `packages/workloads/scripts/setup_mlops.sh`, and verifies
the device backend, component libraries, ABI loading, and PyTorch storage
adapter. `setup_mlops.sh` installs the mlops operation library with its
implementation providers: from a checkout named `mlops` beside this repository,
as it is, when there is one, with a warning when that checkout is not at the
pinned revision or has uncommitted changes; otherwise at the pinned revision,
cloned into the user cache. To use an existing virtual or Conda environment:

```bash
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python"
```

`--expert-parallel quack`, `te` or `both` also installs mlops's optional
expert-parallel backends, which the expert-parallel workloads use. mlops's own
setup installs them after everything else, from whichever mlops checkout
`setup_mlops.sh` installs.

Building also needs the compiler of the accelerator toolkit PyTorch was built
against. PyTorch's wheel carries that toolkit's runtime but not its compiler,
and PyTorch's CMake package, which the adapter's build loads, enables
compilation for its accelerator as it loads. CMake looks for the compiler
through its environment variable, on `PATH`, and under the toolkit's path
variable; when those find none, the adapter's build takes the one the toolkit's
own CMake module finds in the toolkit's default install location, so a toolkit
installed there need not be on `PATH`. When no compiler is installed, the
script stops before building and names the one to install.

## Validation

Run Python tests, formatting/lint checks, and strict typing:

```bash
pytest
ruff check .
mypy
```

`pytest` runs the default suite. The options in `pyproject.toml` deselect two
markers: `slow`, the end-to-end service runs of minutes each,
which `pytest -m slow` runs alone; and `fresh_process`, the tests that assert
their own process's allocator state, which CTest runs one per process.
`packages/engine/tests/engine_tests/native/test_canaries.py` runs the editable build tree's CTest canaries
from the suite: those that need neither the device backend nor a peer, those
that need the device backend, and those that need a peer when
`SHADOWSPILL_NETWORK_PEER` names one.

The CMake build enables warnings as errors and registers with CTest the C
canaries, the PyTorch canaries and the `fresh_process` tests, the last two
listed in `packages/engine/tests/engine_tests/integration/CMakeLists.txt`. Each one's labels name what it
needs: the device backend, a network peer, an SSD, or PyTorch. `ctest -L` and
`-LE` take one regular expression, so excluding two labels joins them with `|`
(`ctest -LE 'network|ssd'`); a `;` list excludes nothing. The PyTorch adapter's
storage operations build against PyTorch's CMake package, which
`CMAKE_PREFIX_PATH` names; without it the adapter builds without them and every
PyTorch canary fails:

```bash
cmake -S packages/engine -B packages/engine/build/dev -DBUILD_TESTING=ON \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build packages/engine/build/dev --parallel
ctest --test-dir packages/engine/build/dev --output-on-failure
```

An unnamed build type builds `RelWithDebInfo`, matching the editable install;
CMake's own default is empty, which is no `-O` at all, and an unoptimized
planner is several times slower at exactly the work this project measures.
Pass `-DCMAKE_BUILD_TYPE=Debug` when that is what you want.

Python never picks up a build directory in the working tree on its own. It
searches the installed package and the editable `packages/engine/build/{wheel_tag}` location, in
that order, and nothing else unless `SHADOWSPILL_LIBRARY_DIRECTORY` names a
directory to try first:

```bash
SHADOWSPILL_LIBRARY_DIRECTORY=packages/engine/build/dev pytest packages/engine/tests/engine_tests
```

A build the loader finds without being told to is the kind of mistake that
reads as the code being slow or subtly broken.

Device integration tests require a matching provider/PyTorch toolchain.
Accelerator-free runtime tests use the mock backend.

For a CPU planner build, `-DSHADOWSPILL_CPU_ONLY=ON` disables both provider
discovery and the PyTorch adapter, even on a machine with a device toolchain.
The corresponding package setting is
`--config-setting cmake.define.SHADOWSPILL_CPU_ONLY=ON`. Do not request provider
backends alongside it. This is a build configuration for offline planning and
inspection; it does not introduce a CPU model-preparation phase.

## Documentation changes

When a public export, C header, cache category, or ownership contract changes,
update the corresponding architecture and API page in the same commit.
Repository tests validate local links and public API name coverage. Exact
function signatures and ABI constants remain in source/header definitions to
avoid duplicate authorities.

The API reference is generated by Sphinx from the public docstrings, written
in NumPy style; `docs/conf.py` configures it and `docs/reference/` lists what
each page shows. Build it with the `dev` extra's tools:

```bash
python -m sphinx -W -n -b html docs docs/_build/html
```

The build checks every public docstring for its summary, parameters and return
value, and every cross-reference; `tests/repository/test_api_reference.py`
runs it. Read the Docs builds the same configuration (`.readthedocs.yaml`).
