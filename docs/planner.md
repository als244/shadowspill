# PressureFit planner

`shadowspill.planner.pressurefit` converts a validated `Program` and physical
simulation configuration into a simulator-verified memory schedule.

```python
from shadowspill.planner import pressurefit

result = pressurefit(
    program,
    initial_residency=initial_residency,
    final_residency=final_residency,
    config=simulation_config,
)
execution_plan = result.to_execution_plan(entrypoints=entrypoints)
```

Every PyTorch callable retains this exact framework-neutral boundary.  This
makes budget sweeps independent of Export, AOTAutograd, Inductor compilation,
and profiling:

```python
from dataclasses import replace

from shadowspill.planner import pressurefit

report = train_step.plan_report
baseline = report.pressurefit_result
device = baseline.simulation_config.devices[0]
simulation_config = replace(
    baseline.simulation_config,
    devices=(replace(device, capacity_bytes=new_object_capacity),),
)

alternative = pressurefit(
    report.program,
    initial_residency=baseline.initial_residency,
    final_residency=baseline.final_residency,
    config=simulation_config,
    options=baseline.options,
)
```

The simulator capacity above is PressureFit's admitted object capacity, not the
PyTorch API's complete physical execution-pool budget.  The latter additionally
accounts for context/provider allocation, fixed slab use, workspace reserve,
and spatial admission.  Direct PressureFit sweeps deliberately do not mutate or
re-admit the live callable.

The API accepts only framework-neutral immutable IR. It knows nothing about
PyTorch, optimizers, model families, operation names, CUDA handles, or tensor
pointers.

## Planning stages

For every bounded recomputation selection, PressureFit:

1. derives stable dense task, alias, access, mutation, and output-reservation
   facts;
2. checks the irreducible task geometry and workspace floor;
3. chooses a deterministic initial device placement;
4. reduces residency spans until capacity fits;
5. realizes releases, dirty offloads, and packed FETCH prefetches;
6. asks the standalone simulator to accept or reject the schedule;
7. selects the shortest valid makespan, using portfolio order only as a tie
   break.

Candidate evaluation may run concurrently. `PressureFitOptions.workers=0` is
the default and uses the available logical CPUs; `workers=1` provides an
explicit serial control. Worker count never enters candidate identity or
ordering. Once a candidate is selected, its action locations are immutable;
physical admission may reject the result but cannot move a trigger.

## Selection cache

PyTorch planning persists a complete selected PressureFit result beneath the
explicit `planning_cachedir/pressurefit/selections` tree. When no directory is
supplied, `~/.cache/shadowspill` is used. The content key includes the Program
digest, initial/final residency, simulator capacities and transfer calibration,
and every behavior-bearing planner option. Evaluation worker count is excluded
because it cannot affect the deterministic result. ShadowSpill does not read an
environment variable to redirect this cache.

A cache hit does not trust stale timing evidence: ShadowSpill validates the
schedule against the current Program and selections, replays it through the
standalone simulator, and requires the cached selected makespan to match. The
cache restores complete candidate diagnostics and never reruns or modifies a
selected transfer trigger. Writes use an atomic replacement.

## Results and errors

`PressureFitResult` owns the selected schedule, recomputation choices, exact
simulation result, and observational candidate diagnostics. It can be bound to
frontend entrypoints and physical admission as an `ExecutionPlan`.

`PressureFitInfeasibleError` exposes a stable `kind` plus optional device,
task boundary, required bytes, capacity, and complete candidate diagnostics.
Callers should not parse its human-readable message.

The current planner uses bounded enumeration for small recomputation spaces
and a deterministic endpoint/prefix/single-deviation portfolio for larger
spaces. This keeps search finite without introducing operation-specific rules.

## Compiled boundary

Python resolves the bounded recomputation portfolio and projects each selected
task topology into the simulator ABI. `libshadowspill_planner.so` derives the
dense access, mutation, reservation, capacity, and seed records directly from
that topology, then evaluates the complete policy portfolio: residency
reduction and repair, interval refinement, action emission, simulator replay,
schedule hashing, diagnostics, and context-local selection. Only the globally
selected schedule is materialized as Python IR.

The Python candidate path remains an independently executable differential
oracle when the compiled planner is unavailable. Native and Python paths must
agree on every recomputation choice, action and trigger, byte count, repair,
failure diagnostic, stall, peak, schedule digest, and makespan. Parallel native
contexts preserve declaration order and use the same deterministic
makespan/ordinal tie-break as serial execution.
