# Simulation

The simulator deterministically replays an explicit `ShadowSpillProgram`,
`MemorySchedule`, graph-pair selection, device configuration, and optional
physical admission. It never invokes the planner. A [search](search.md) calls
it while evaluating candidates, and the same public API can evaluate a
supplied schedule independently.

Runtime-global shared aliases retain their true sizes in the `ShadowSpillProgram`, but
are projected out of the callable's movable alias set. Their execution and
retained-spill footprints are subtracted from available capacity before the C
simulation runs and added back to decoded physical peaks. They therefore
consume real budget exactly once without acquiring plan-owned actions.

## Resources and intervals

The model includes:

- ordered compute tasks on execution resources;
- independent fetch and evict lanes;
- route latency and calibrated directional bandwidth;
- object residency and task input readiness;
- physical allocation deltas and reuse dependencies.

Each `TaskInterval` records ready, start, and end time, and what it waited
for. Each `TransferInterval` records the action and boundary that triggered
it, its direction, queue and wire timing, and bytes. `SimulationResult`
carries the makespan, the task and transfer intervals, per-device and spill
peaks, and every capacity shortfall the plan waited on.

## Memory actions

The four action kinds of the [IR](ir.md#memory-schedule) have one simulated
meaning each, and the runtime's executor keeps the same contract:

| action | lane | requires | leaves |
|---|---|---|---|
| fetch | fetch | a current spill copy and no device copy | a current device copy; the spill copy is dropped unless the alias retains it |
| write-back | evict | a current device copy with no copy of it in flight | the spill copy current; the device copy kept, allocated and authoritative |
| release | none | a current device copy, and a current spill copy when the value is still needed | the device copy dropped at once |
| evict | evict | a current device copy with no copy of it in flight | the spill copy current and the device copy dropped when the copy lands |

A write-back whose spill copy is already current completes at its trigger
without occupying the lane. A release behind a pending write-back of the
same alias waits for the copy to land, the way a fetch with nowhere to land
waits for room, and holds the actions behind it while it waits. A value is still
needed after a release when a later task reads it or the final residency
names it; a release that would drop its only current copy is refused as
an `invalid-release`, at the release rather than at the fetch, task or
final residency that would miss the value. A value nothing needs any more
is released freely, whatever its spill copy holds.

A device-to-spill copy carries the version it started from. A task that
writes the alias while the copy is in flight leaves the spill copy stale,
never current by fiat, so a later fetch or the final residency reports the
loss instead of the simulation hiding it.

Each `TransferInterval` names the action kind that issued it beside its
direction, so write-backs are distinguishable from evictions on the evict
lane.

## Trigger-time capacity

Transfer capacity is charged at the directive trigger, not when the copy
reaches the wire. This matches runtime behavior: `after_task()` reserves the
destination before returning, while the worker may submit the transfer later
after earlier lane work completes.

A reservation with nowhere to land does not fail the simulation. It waits for
room and is retried, exactly as the runtime does, so a plan that comes up short
is slower rather than rejected. The wait appears as a `device-capacity` stall
and the shortfall as a `CapacityViolation` beside it: the stall says when and
for how long, the violation says by how much.

The simulator receives `ActionPhysicalDelta` values for these reservations and
`TaskPhysicalDelta` values for task allocations/releases. It also consumes
`MemoryReuseDependency` edges emitted by physical admission. A successor that
reuses an evicted range cannot start before that eviction completes, even when
the nominal object schedule would otherwise permit it.

## Compiled production path

`simulate()` always uses the installed C simulator and fails closed if the
library or ABI is unavailable. The readable Python implementation is a
non-installed differential oracle under `reference/python/simulator`; it is
never selected by production configuration or a diagnostic flag.

## Fidelity

Plan profiles predict task compute spans; transfer calibration predicts copy
duration; the scheduler predicts readiness and lane overlap. Runtime tracing
can compare, execution by execution and transfer by transfer:

- simulated and real start time;
- simulated and real duration;
- profiled and real task-event duration;
- simulated and real inter-task gaps;
- simulated and real selected-task span.

Simulator fidelity is evaluated on warmed execution. Startup fetches and final
cooldown are reported separately so their transfer and readiness costs remain
visible rather than being folded into selected-task timing. [Step
boundaries](step-boundaries.md) defines that cycle — which synchronization
points separate repeated invocations, and which of the boundary costs the
makespan does and does not price.

The [StepResult diagnostics guide](../python/step-diagnostics.md) defines the
real-versus-simulated task and transfer fields, clock domains, selected-span
summary, trace-integrity checks, and investigation workflow.

Previous: [Planning orchestration](planning.md). Next:
[Memory runtime](memory-runtime.md).
