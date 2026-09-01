# Simulation

The simulator deterministically replays an explicit `Program`,
`MemorySchedule`, recomputation selection, device configuration, and optional
physical admission. It never invokes the planner. [PressureFit](pressurefit.md)
calls it while evaluating candidates, and the same public API can evaluate a
supplied schedule independently.

Runtime-global shared aliases retain their true sizes in the `Program`, but
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

Each `TaskInterval` records ready, start, and end time. Each
`TransferInterval` records action trigger, queue/wire timing, bytes, and
completion. `SimulationResult` contains makespan, per-device peaks, transfer
utilization, stalls, and task/transfer intervals.

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
