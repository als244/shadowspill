# Repository structure and validation

## Top-level layout

```text
shadowspill/
├── src/
│   ├── shadowspill/       installed Python package
│   └── tools/             reusable source-tree diagnostics and qualification tools
├── csrc/                  the C library, its backends, and public C headers
├── tests/                 fast tests mirroring product/tool boundaries
├── workloads/             model and data clients
├── reference/             executable reference implementations
├── benchmarking/          reusable Program datasets and planning evaluation
├── qualification/         thin numerical and performance release gates
├── docs/                  architecture, Python, C, and development
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
│   ├── admission/         neutral physical admission
│   ├── diagnostics/       PlanReport values
│   ├── pressurefit/       the search itself
│   ├── recomputation/     resolved-program enumeration
│   └── serialization/     neutral artifact encode/decode
├── simulator/             the simulator and diagnostic timeline
├── runtime/               physical admission and replay bindings
├── plots/                 step-run and step-search figures
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
    ├── sharing/           runtime-owned TensorRef handles
    ├── state/             persistent model/optimizer import
```

High-level modules expose small orchestration functions. Detailed algorithms
live in domain-named submodules; generic `utils`, `helpers`, and `common`
modules are avoided.

## The C tree

[csrc/README.md](../../csrc/README.md) draws the layout of `csrc/`: the
public headers, the one library under `src/` (common, simulator, planner,
runtime with a directory per runtime component), the backends, and the
PyTorch adapter.

Everything under `src/` builds into one library. The backends share one
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
`src/shadowspill/pytorch/profiling/` lives in
`tests/shadowspill/pytorch/profiling/`.

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
backend, builds and installs the library, its backends and the adapter, installs development
dependencies, installs the mlops operation library with its implementation
providers, and verifies the device backend, component libraries, ABI
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

An unnamed build type builds `RelWithDebInfo`, matching the editable install;
CMake's own default is empty, which is no `-O` at all, and an unoptimized
planner is several times slower at exactly the work this project measures.
Pass `-DCMAKE_BUILD_TYPE=Debug` when that is what you want.

Python never loads out of this directory on its own. It searches the installed
package and the editable `build/{wheel_tag}` location, in that order, and
nothing else unless `SHADOWSPILL_LIBRARY_DIRECTORY` names a directory to try
first:

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
