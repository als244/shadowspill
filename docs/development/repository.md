# Repository structure and validation

## Top-level layout

```text
shadowspill/
├── src/
│   ├── shadowspill/       installed Python package
│   └── tools/             reusable source-tree diagnostics and qualification tools
├── csrc/                  the compiled library, its backends, and public C headers
├── tests/                 fast tests mirroring product/tool boundaries
├── workloads/             model and data clients
├── benchmarking/          reusable Program datasets and planning evaluation
├── qualification/         thin numerical and performance release gates
├── docs/                  architecture, Python, C, development, and investigations
├── scripts/               one-command setup
├── CMakeLists.txt         compiled-component build orchestrator
└── pyproject.toml         Python build, dependency, lint, type, and test configuration
```

`qualification/` contains launchers and protocol descriptions only. Reusable
logic lives under `src/tools/qualification/`, and product behavior remains
under `src/shadowspill/`.

`benchmarking/` has two independent phases: Program collection performs
capture/compilation/profiling once, while planning evaluation consumes saved
Programs across budgets and transfer bandwidths.

## Python package

```text
src/shadowspill/
├── ir/                    framework-neutral values and indexed projections
├── planner/               PressureFit orchestration and compiled bindings
├── simulator/             compiled simulator and diagnostic timeline
├── runtime/               physical admission and replay bindings
└── pytorch/
    ├── capture/           Export/AOT capture and semantic storage contracts
    ├── partition/         stage policies, splitting, provenance, authentic controls
    ├── graph_pairs/       differentiation alternatives by structural contract
    ├── compilation/       Inductor adapter and executable storage manifests
    ├── profiling/         representative inputs, timing, allocation contract, workspace
    ├── lowering/          ObjectCatalog and task binding into Program
    ├── optimizer/         optimizer graph capture and ordering
    ├── planning/          forward/training orchestration and physical admission
    ├── materialization/   selected callable and runtime state publication
    ├── execution/         before/compiled/after task skeletons
    ├── runtime_adapter/   Python-to-C runtime and allocator boundary
    ├── diagnostics/       PlanReport and StepDiagnostics
    ├── state/             persistent model/optimizer import
    └── program_serialization/
```

High-level modules expose small orchestration functions. Detailed algorithms
live in domain-named submodules; generic `utils`, `helpers`, and `common`
modules are avoided.

## Compiled components

```text
csrc/
├── include/shadowspill/   public headers: shadowspill.h, status.h, simulator.h,
│                          planner.h, runtime.h, admission_replay.h, backend.h,
│                          profiler.h
├── src/
│   ├── common/            shared by all three components
│   ├── simulator/
│   ├── planner/           and planner/admission/
│   └── runtime/           and memory/ objects/ tasks/ transfers/ sync/ plan/
│                          telemetry/, each with its own internal.h
├── backends/{mock,<provider>}/
└── adapter/pytorch/       include/shadowspill/pytorch_adapter.h and the
                           allocator/storage/profiler bridge sources
```

Everything under `src/` builds into one library. Backends and the PyTorch
adapter are separate targets with their own `CMakeLists.txt`, because each is
compiled elsewhere against a toolchain the rest of the tree must not require.
Provider dependencies stay inside those boundaries.

A private header named `internal.h` belongs to the directory holding it and is
included by path from anywhere else, so `"internal.h"` always means this
directory's.

## Tests

The test tree mirrors the boundary under test:

```text
tests/
├── shadowspill/           package tests matching src/shadowspill subpackages
├── csrc/                  C, mock-backend, sanitizer, and device canaries
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
backend, builds and installs every compiled component, installs development
dependencies, and verifies the device backend, component libraries, ABI
loading, and PyTorch storage adapter. To use an existing virtual or Conda
environment:

```bash
./scripts/setup.sh --python "$CONDA_PREFIX/bin/python"
```

## Validation

Run Python tests, formatting/lint checks, and strict typing:

```bash
pytest
ruff check .
mypy
```

The CMake build enables warnings as errors and registers compiled canaries with
CTest:

```bash
cmake -S . -B build/dev -DBUILD_TESTING=ON
cmake --build build/dev --parallel
ctest --test-dir build/dev --output-on-failure
```

Device integration tests require a matching provider/PyTorch toolchain.
Accelerator-free runtime tests use the mock backend.

## Documentation changes

When a public export, C header, cache category, or ownership contract changes,
update the corresponding architecture and API page in the same commit.
Repository tests validate local links and public API name coverage. Exact
function signatures and ABI constants remain in source/header definitions to
avoid duplicate authorities.
