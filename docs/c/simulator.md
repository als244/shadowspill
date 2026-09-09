# Simulator C API

Include `<shadowspill/simulator.h>`. The simulator is a deterministic,
standalone evaluator for an already selected schedule.

## Input

`ShadowSpillSimulationProgram` contains indexed arrays for:

- one `ShadowSpillSimulationDevice` per device, holding its capacity and its
  per-direction bandwidth and latency;
- aliases, sizes, versions, and retained spill copies;
- task resource, duration, workspace, dependencies, inputs, outputs, and
  mutations;
- memory actions, each a `ShadowSpillMemoryActionKind` (release, evict, fetch,
  write-back), and initial and final residency, each a
  `ShadowSpillMemoryLocation` (device or spill);
- task/action physical byte deltas;
- physical memory-reuse dependencies.

When `use_admission_accounting` is set, physical deltas replace synthetic
workspace accounting. Fetch destination bytes are charged at action trigger
and released or published at the declared completion transition.

## Output

`ShadowSpillSimulationResult` reports status, structured infeasibility fields,
makespan, spill peak, task intervals, transfer intervals, one
`ShadowSpillDevicePeak` per device holding its object, workspace and total
peak, and capacity shortfalls. Caller-provided
interval, peak, and violation buffers remain caller-owned.

A `ShadowSpillTaskInterval` and a `ShadowSpillTransferInterval` both carry
readiness, start, end and a stall mask; a transfer interval also carries its
bytes, its `ShadowSpillTransferDirection` and the kind of the action
that issued it, which tells a write-back from an eviction on the evict
lane. An action whose preconditions fail ends the simulation with the
matching status (`SHADOWSPILL_STATUS_INVALID_RELEASE`, `_INVALID_EVICT`,
`_INVALID_FETCH`, `_INVALID_WRITE_BACK`);
[Simulation](../architecture/simulation.md#memory-actions) states the
preconditions.
Stall masks distinguish input residency, device capacity, source readiness,
spill capacity, and physical memory reuse. The last two of those sound alike
and are not: memory reuse is an ordering wait the plan created, for the
eviction or release that frees the allocation about to be reused, while device
capacity is a shortfall -- no room at all for a fetch to land or for a task's
outputs and workspace. A plan that arranges its own capacity waits on the
first and never reaches the second.

A fetch or task launch that does not fit waits for room and is retried,
rather than ending the simulation; only a plan that can never make room
fails, as a deadlock. Each shortfall is reported once, at its first refusal,
as a `ShadowSpillCapacityViolation` carrying the time, device, task, alias,
location, capacity, used and requested bytes, and a
`ShadowSpillCapacityViolationReason`. The reason says what a repair would have
to change: `INITIAL_DEVICE` and `INITIAL_SPILL` mean the plan is over budget
before it starts and no scheduling helps, while `FETCH_DEVICE`,
`EVICT_SPILL` and `TASK_DEVICE` each name one fetch, one eviction or
write-back, or one task launch that asked for too much. The paired
`SHADOWSPILL_STALL_DEVICE_CAPACITY` mask says the plan waited;
the violation says by how much it was short.
`capacity_violation_count` is the true total even when it exceeds the buffer,
so a truncated list is distinguishable from a complete one, and a null buffer
counts without storing.

## Functions

- `shadowspill_simulate()` validates and evaluates one indexed program.
`shadowspill_abi_version()` and `shadowspill_status_string()` cover loading
and diagnostics for this boundary as for every other; see the
[C API guide](README.md#abi-use).

The call performs no I/O, owns no external storage after return, and uses no
global mutable state. Distinct result buffers can be evaluated concurrently.
