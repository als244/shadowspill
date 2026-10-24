# Source layout and migration: review proposal

Status: the user accepted phase-based directories and the small top-level public
API, and requested implementation on a new branch. The user also requires small,
focused files/functions, with hierarchy inside a phase where it helps. The user
selected migrating repository callers and then removing superseded APIs and
duplicate implementations, without deprecated public compatibility wrappers.
No source files have been moved or deleted.

## Selected public surface

Use one canonical top-level import surface for the main API, with the
implementation organized by phase. The root exports names; it does not contain
another orchestration implementation or a mandatory Pipeline class.

```python
from shadowspill import (
    Program, Profiles, ProfiledProgram, Plan,
    build_forward, build_forward_backward, build_step,
    measure_program, apply_profiles, measure_communication,
    plan_program, sweep_programs, write_report,
    Parallelism, MemoryBudgets, CommunicationInfo,
    Executor, Trainer,
)
```

The example shows discoverable entry points, not a requirement to import all of
them. Root exports must load implementations only when needed so importing CPU
planning/artifact/reporting APIs does not initialize CUDA, install the allocator,
load GPU providers, or start compilation. CPU planning must not depend on an
Executor or on a Trainer instance.

Keeping a separate public PyTorch frontend API was not selected as the canonical
surface. Top-level calls expose the complete pipeline without requiring callers
to learn backend package structure. This does not imply that the initial
builders accept arbitrary frameworks; their model frontend is PyTorch.

## Selected implementation ownership

```text
shadowspill/
  __init__.py       # Small, lazy public exports.
  artifacts/       # Program, Profiles, ProfiledProgram, Plan; archive encoding.
  build/           # Export, partition, compile, lower; finalized task forms.
  profiling/       # GPU measure_program; CPU apply_profiles in a separate file.
  planning/        # CPU plan_program/sweep_programs; solver and simulation internals.
  communication/   # CPU topology/cost data; separate GPU calibration implementation.
  parallelism/     # CPU logical groups, parameter/input placements, storage policies.
  execution/       # Executor, live bindings, initialization/export, invocation handling.
  training/        # Trainer, checkpoint/data/scheduler helpers; composes public phases.
  reporting/       # CPU reports, timelines, plots from existing artifacts/results.
  ir/              # Shared low-level task, storage, dependency and effect definitions.
  runtime/         # Allocator/runtime bindings used for execution.
```

These are responsibility boundaries, not a command to create empty directories
or split each small class into its own file. Preserve useful internal compiler,
solver, simulator, and runtime subdivisions where they have a clear owner.
The C++ planner/simulator/runtime implementations remain engines behind their
respective phases; this does not require rewriting their algorithms.

The user explicitly requests the simplest practical public surface and small,
focused files/functions. Give each function and module a clear responsibility;
keep related helpers together. Add subpackages when a phase has real components,
such as compiler adapters or simulation, rather than adding forwarding layers.
Public re-exports should not grow into an additional orchestration framework.
Avoid giant orchestration/configuration objects and duplicate record hierarchies.
Apply this structure during implementation as well as final cleanup.

Examples of important file separation within those owners:

- `profiling/measure.py` performs GPU work; `profiling/apply.py` validates and
  attaches supplied records on CPU. They share profile data structures, not a
  function with hidden measurement behavior.
- `communication/calibration.py` needs hardware; cost evaluation/topology data
  can be used independently on CPU.
- `parallelism` describes logical participation and placement; `execution`
  creates live groups from the full rank map. Build never takes those live groups.
- `artifacts` owns public phase results and one archive codec. Internal IR
  structures remain the authoritative graph/storage facts, rather than parallel
  copies in several artifact wrappers.
- PyTorch/provider-specific adapters live with the phase that needs them.
  Do not create another cross-phase frontend orchestration path underneath the
  public one merely to preserve the current package layout.

`build` may use a GPU when finalization requires it, as already selected. The
important separation is independent phase invocation and no runtime pools in
build/planning; the folder name does not promise universally CPU-only compilation.

## Current entry points mapped to the new responsibilities

| Current surface | Proposed replacement/composition |
| --- | --- |
| `plan_forward` | `build_forward`, supplied/measured Profiles, `apply_profiles`, `plan_program`, then fresh-worker `Executor`. |
| `plan_step` | `build_step` followed by the same independent profile/plan/execute phases. Trainer/quickstart compose them for usual workflows. |
| `build_step_programs` | Builder output plus planning-time ordering/candidate selection; do not preserve a builder that silently profiles or creates one Program for every task ordering. |
| `prepare_step` / `PreparedStep` | Explicit Program and reusable Profiles / ProfiledProgram artifacts. Inventory their retained metadata/code/ownership before replacing them. |
| `plan_step_search` | Orchestrator builds/measures desired candidates explicitly, then invokes CPU-only `sweep_programs`. |
| `search_prepared_steps` | `sweep_programs` over supplied ProfiledPrograms. Retain core incumbent reuse through explicit planner inputs. |
| `admit_step` / `PlannedTrainStep` / `PlannedForward` | Shared Plan plus Executor binding/validation and the selected result-handle interface. No model recapture as a hidden part of execution. |
| `Distributed`, `DataParallel`, transfer-sharding controls | The central Parallelism description, resolved once during build, plus live execution binding. Preserve required model-specific placement/communication semantics. |
| Mandatory `ArtifactStore`, framework/planning store wrappers, step archives | One explicit save/load path for public artifacts; optional caching that is not required for phase handoff. |
| Several public planning-report wrappers | Inspect Plan/SearchResult fields and call `write_report`; preserve useful diagnostics without multiple competing result owners. |

This is a responsibility map, not a claim that each old function can be renamed
one-to-one. Runtime allocation admission remains required even though
`admit_step` would cease to be the user-facing preparation path.

## Concrete callers to migrate

The current Trainer ShadowSpill backend invokes both ordinary planning and
prepared-step APIs. Migrate both onto one implementation of the new public
phases. Quickstart and baseline preparation should compose those calls, rather
than retain private copies of capture/profile/search logic.

Observed callers include:

- `src/shadowspill/training/backends/shadowspill.py`;
- `workloads/recipes/text/quickstart.py` and model/precision recipe helpers;
- `baselines/engines/shadowspill/preparation.py`;
- `workloads/models/glm53_flash/forward.py`;
- integration, API, training, distributed-profiling, search, and repository tests.

Existing source inspection also shows that `shadowspill.pytorch.__init__`
exports runtime pools alongside planning and prepared/execution objects, while
`shadowspill.pytorch.planning` exports many intermediate phase wrappers. The
refactor should simplify those ownership boundaries, not recreate all of their
types as additional public wrappers around the four selected phase artifacts.

## Selected migration and removal policy

Migrate repository callers and then remove the superseded public entry points
and duplicate orchestration/store implementations. The user selected this
approach rather than retaining deprecated public compatibility wrappers.
Temporary internal adapters may support migration, but are removed when callers
use the new phases; they must not become a permanent second API or implementation.

Before removal, map existing capabilities such as tied/shared storage, partial
checkpoint operations, parameter observations, optimizer/precision controls,
and failure cleanup to their new owners. Do not equate deleting redundant
wrappers with deleting unreviewed advanced behavior. Mark genuinely unsupported
cases explicitly instead of silently weakening tests to make migration pass.

## Verification and remaining design work

After the API/layout are agreed, acceptance should cover independent CPU phase
use, no-runtime-pool build/planning, profiling's chosen allocation-observation
path, self-contained archive round trips with original caches absent, executable
compatibility failures, bounded shard initialization/export, one-GPU DP task
profiling, and a shared distributed Plan with a nonidentity complete rank map.
Preserve useful existing numerical and memory-admission checks.

The GLM trace example remains an acceptance scenario: supply known task timings
and explicit missing estimates, plan on CPU, and render timelines. Estimated
memory must keep the result simulation-only.

A final complete API example pass must settle remaining record/default/extension
details; a high-level folder choice does not implicitly approve their signatures.
The allocator-observation/provider-preparation investigations remain engineering
tasks. No GPU jobs or source refactor implementation are part of this review.
