# Independent build, profile and planning phases

The refactor is being implemented in phases. The CPU path described here works;
new model builders, GPU measurement, executable bundles and caller migration are
still in progress on `codex/offline-pipeline-refactor`.

```python
from shadowspill import (
    Program, Profiles, apply_profiles, plan_program,
    MemoryBudgets, CommunicationInfo, write_report,
)

program = Program.load("program.ssp")
profiles = Profiles.load("profiles.ssp")
profiled = apply_profiles(program, profiles)
plan = plan_program(
    profiled,
    memory_budgets=MemoryBudgets(
        execution_bytes=40 * 2**30, spill_bytes=128 * 2**30,
    ),
    communication_info=CommunicationInfo(
        fetch_bytes_per_second=24e9, evict_bytes_per_second=20e9,
    ),
)
plan.save("plan.ssp")
write_report(plan, "report")
```

All the calls in this example run on CPU. They do not import PyTorch, initialize
CUDA, install an allocator, open pools, or require ArtifactStore. Planning uses
the existing native search/simulator library. Loading and inspecting an artifact
does not load that native library.

`Program` stores task dependencies, storage extents, alternative task groups,
source names and implementation identities without durations. `Profiles` stores
plain numeric task durations and unresolved workspace sizes for one setup.
`apply_profiles` checks complete coverage and snapshots the supplied values.
An explicit zero is valid; a missing workspace or runtime/provider overhead is
an error, not zero. Known compiler memory is retained as a lower bound.

```python
from shadowspill import TaskProfile

overrides = Profiles(
    setup=profiles.setup,
    tasks={profile_key: TaskProfile(duration_seconds=0.0072)},
)
profiles = profiles.merge(overrides)
```

Later supplied values overwrite earlier values. Omitted fields are preserved,
so this timing override preserves existing memory information. Supplied values
without classification metadata are estimates. Task compatibility and setup
mismatches are errors. Changes to editable Profiles never alter an already
created ProfiledProgram or Plan.

The current offline milestone supports one logical rank and host transfers.
Explicit latency is charged as a transfer phase. Concurrent fetch/evict rates
default to independent full-duplex lanes; callers can provide different rates.
The communication description is saved alongside the plan, independently of
compute measurements. Collective calibration, topology-aware collective estimates,
and distributed transfer layouts are deferred to the distributed milestone.

The GPU allowance includes fixed runtime/provider overhead. Planning subtracts
that overhead before scheduling objects/workspace, and reporting adds it back to
the total peak. Spill capacity is separate. No physical memory is allocated to
perform this calculation.

`sweep_programs({name: profiled_program, ...}, memory_budgets=[...],
communication_info=...)` is an optional loop. It keeps every successful Plan,
diagnostic failures, and a `best_per_budget` lookup. It does not construct or
profile candidates. Invalid arguments and implementation errors propagate.

Each phase result saves as one atomic archive with a versioned JSON description
and checksum. ProfiledProgram contains its Program and profiles; Plan contains
the selected schedule, simulation, assumptions and diagnostics. No compiler cache
directories are included. Reporting is explicit and separate from persistence.
The native simulator currently records memory peaks rather than a complete
allocation history. Reports therefore show peak-versus-capacity bars alongside
the compute/transfer timeline; they do not invent a memory curve.

Workspace totals permit logical simulation. They do **not** establish allocation
order, storage handoffs, charged alignment, persistent provider allocations or
the task code required for execution. The initial CPU artifacts therefore expose
`simulation_only_reasons` and no physical `memory_layout`. A later milestone
adds compiler/measurement allocation evidence and executable payloads without
promoting estimates into execution facts.

Implementation ownership:

- `artifacts/`: immutable phase results, editable profile collections, archive codec.
- `profiling/apply.py`: CPU validation/application of supplied profiles.
- `planning/`: public CPU planning/sweep and capacity/cost conversion.
- `communication/`: saved communication assumptions; calibration will be separate.
- `reporting/`: semantic task/transfer tables and timeline/memory figures.
- `ir/` and existing native planner/simulator: shared structural validation and engines.

The internal `_legacy.split_costed_program` imports old captured IR for validation
while callers migrate. It is not a second public build path and does not claim
that imported workspace totals carry allocator evidence.
