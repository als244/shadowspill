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

Candidate evaluation may run concurrently. Worker count never enters candidate
identity or ordering. Once a candidate is selected, its action locations are
immutable; physical admission may reject the result but cannot move a trigger.

## Selection cache

PyTorch planning persists a complete selected PressureFit result in
`~/.cache/shadowspill/recomputation`, or in
`SHADOWSPILL_RECOMPUTATION_CACHE` when configured. The content key includes the
Program digest, initial/final residency, simulator capacities and transfer
calibration, and every behavior-bearing planner option. Evaluation worker
count is excluded because it cannot affect the deterministic result.

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

The readable Python policy constructs candidates. `libshadowspill_planner.so`
owns the framework-neutral compiled selection boundary: it independently
replays every materialized candidate through `libshadowspill_simulator.so` and
selects by makespan and declaration order. Differential tests require this
selection to agree with the Python policy. Keeping schedule construction in
the policy layer avoids a second, divergent implementation of PressureFit's
heuristics while retaining a stable C handoff for other language frontends.
