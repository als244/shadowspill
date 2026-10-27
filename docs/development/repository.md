# Repository structure and validation

## Top-level layout

```text
shadowspill/
├── src/
│   ├── shadowspill/       installed Python package
│   └── tools/             naming checks and sanitizer support
├── csrc/                  the C library, its backends and public C headers,
│                          the network extension, and the memory daemon
├── tests/                 fast tests mirroring product/tool boundaries
├── workloads/             model and data clients
├── reference/             executable reference implementations
├── benchmarking/          quickstart, profiled-program corpora, and planning evaluation
├── qualification/         gate commands, implementations, configurations, and docs
├── docs/                  architecture, Python, C, and development
├── scripts/               one-command setup
├── CMakeLists.txt         build orchestrator for the C tree
└── pyproject.toml         Python build, dependency, lint, type, and test configuration
```

`qualification/` owns the complete gate implementation and its documentation.
There is no forwarding package under `src/tools/`. `qualification/gates/`
runs the selected gates; each gate's `matrix.py` launches its cases, and its
`run.py` implements the single-case command. Product behavior remains under
`src/shadowspill/`. The numerical gate is a package,
`qualification/numerical/`, holding one case end to end: the
request and its identity, the reference arm, the planned run, the two
comparisons, the artifact, the verdict, and the matrix that launches cells.
The throughput gate is the package beside it,
`qualification/performance/`, holding one cell in the order it
runs: the manifest and its budgets, the readings taken off the runtime, the
phases (calibrate, plan, warm, measure), and the verdict each gate reaches.

`workloads/` supplies optional model and data recipes; qualification budgets and
acceptance rules live under `qualification/`. Build accepts ordinary user models
and call examples. `shadowspill.training.Trainer` drives an Executor from saved
Plans and iterable inputs, without importing recipes.

`benchmarking/quickstart/` composes the public phase APIs in separate workers. The
`bandwidth_frontier/` package consumes `ProfiledProgram` archives and runs CPU
planning/plotting only. Corpus collection/evaluation remain
separate: collection builds and profiles, while evaluation reads the saved
corpus and plans budgets and transfer assumptions. Shared benchmark host-memory
preflight lives in `benchmarking/host_memory/`; it checks physical/cgroup limits
before the baseline worker creates spill pools.

## Python package

```text
src/shadowspill/
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
│   ├── recording/         opt-in task/transfer recording for Executor
│   ├── distributed/       collective and shard-storage execution bindings
│   └── observations/      optional bounded runtime norm collection
├── checkpoint/            streamed checkpoint format over Executor tensor access
├── parallelism/           logical groups, parameter placement and storage declarations
├── resources/             portable provider descriptions, setup and cleanup
├── planning/              CPU-only profile binding, scheduling and physical layout
│   ├── admission/         resolve saved memory evidence without a live pool
│   └── training/          bind example/update orderings and gradient completion
├── reporting/             CPU-only reports and timeline/summary output
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
│       └── run/           stored-run discovery, trace matching and report indexes
├── profiling/             ordinary-allocator timing and memory measurement
│   └── observation/       allocator observations and lifetime decoding
├── task/                  one task's shape and provenance, framework-free
│   ├── storage.py         the storage contract: roots, output views, mutations
│   ├── allocations.py     what a task promised about its allocations, and the
│   │                      path the profiled run observed
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
├── planner/               the planning question and its C bindings
│   ├── plan_store/        request identity, validated cache I/O and search reuse
│   ├── admission/         neutral physical admission, after a search answers
│   │   ├── physical.py    what a plan needs physically, before a runtime exists
│   │   └── layout/        leases to fixed offsets, and the certificate
│   ├── diagnostics/       search evidence, counters and plan summaries
│   ├── search/            the search seam
│   │   ├── toolkit/       what any search may call
│   │   └── algorithms/
│   │       └── pressurefit/   the search that ships
│   │           └── candidates/  the records the search returns, decoded
│   └── serialization/     neutral artifact encode/decode
├── simulator/             the simulator and diagnostic timeline
│   └── indexing/          the template, one schedule bound onto it, the result
│                          decoded back, and the buffers all three hand to C
├── network/               pool configurations whose memory is on another
│                          machine, and the library that serves them
├── frontend.py            the twelve methods a framework frontend implements,
│                          and nothing else
├── runtime/               one runtime in this process, framework-neutral:
│   │                      bootstrap, core, configuration, plan, occupancy,
│   │                      residue, calibration, teardown, the admission replay,
│   │                      the trace and telemetry it exposes, the markers a
│   │                      caller times with, and the failures it reports
│   ├── abi/               the ctypes projection of the C API, one module per
│   │                      record kind
│   ├── objects/           runtime objects: the reference, and how one is made
│   ├── plan/              worker bindings, admission, lifecycle and optional tracing
│   │   └── layout/        certified offsets and completion dependencies
│   └── timing.py          an instant recorded on a stream, and the time between two
├── training/              saved-Plan training loops, schedules and checkpoints
│   ├── diagnostics/       no-update startup calls, state preservation and reports
└── pytorch/
    ├── capture/           Export/AOT capture and semantic storage contracts
    │   ├── artifacts/     graph records, geometry, identity and output schemas
    │   ├── gradients/     accumulation, precision and padded gradient storage
    │   └── storage/       one task's contract: the records, the roots, the
    │                      capture, the symbolic replay
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

High-level modules expose small orchestration functions. Detailed algorithms
live in domain-named submodules; generic `utils`, `helpers`, and `common`
modules are avoided.

## The C tree

[csrc/README.md](../../csrc/README.md) draws the layout of `csrc/`: the
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
`src/shadowspill/profiling/` lives in
`tests/shadowspill/profiling/`.

```text
tests/
├── shadowspill/           package tests matching src/shadowspill subpackages
├── csrc/                  C, mock-backend, sanitizer, and device canaries, one directory per library
├── integration/           fresh-process framework/backend integration
├── tools/                 source-tree tool tests
├── workloads/             workload definition tests
├── benchmarking/          corpus/frontier harness tests
├── repository/            packaging, naming, links, and API documentation drift
└── fixtures/              immutable test-only inputs
```

Long numerical and throughput runs belong in `qualification/`, not the unit
suite.

## Setup

```bash
./scripts/setup.sh
```

The script creates `.venv`, installs PyTorch with the machine accelerator
backend, builds and installs the library, its backends and the adapter,
installs development dependencies and those of the training harness, installs
the mlops operation library with its implementation providers, and verifies
the device backend, component libraries, ABI loading, and PyTorch storage
adapter. To use an existing virtual or Conda environment:

```bash
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python"
```

`--expert-parallel quack`, `te` or `both` also installs mlops's optional
expert-parallel backends, which the expert-parallel workloads use. mlops's own
setup installs them, after everything else, from the mlops checkout beside this
one, so the option needs that checkout.

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

The CMake build enables warnings as errors and registers the C canaries with
CTest. The PyTorch adapter's storage operations build against PyTorch's CMake
package, which `CMAKE_PREFIX_PATH` names; without it the adapter builds
without them and every PyTorch canary fails:

```bash
cmake -S . -B build/dev -DBUILD_TESTING=ON \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build/dev --parallel
ctest --test-dir build/dev --output-on-failure
```

An unnamed build type builds `RelWithDebInfo`, matching the editable install;
CMake's own default is empty, which is no `-O` at all, and an unoptimized
planner is several times slower at exactly the work this project measures.
Pass `-DCMAKE_BUILD_TYPE=Debug` when that is what you want.

Python never picks up a build directory in the working tree on its own. It
searches the installed package and the editable `build/{wheel_tag}` location, in
that order, and nothing else unless `SHADOWSPILL_LIBRARY_DIRECTORY` names a
directory to try first:

```bash
SHADOWSPILL_LIBRARY_DIRECTORY=build/dev pytest tests/shadowspill
```

A build the loader finds without being told to is the kind of mistake that
reads as the code being slow or subtly broken.

Device integration tests require a matching provider/PyTorch toolchain.
Accelerator-free runtime tests use the mock backend.

## Documentation changes

When a public export, C header, cache category, or ownership contract changes,
update the corresponding architecture and API page in the same commit.
Repository tests validate local links and public API name coverage. Exact
function signatures and ABI constants remain in source/header definitions to
avoid duplicate authorities.
