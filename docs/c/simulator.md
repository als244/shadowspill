# Simulator C API

Include `<shadowspill/simulator.h>`. The simulator is a deterministic,
standalone evaluator for an already selected schedule.

## Input

`ShadowSpillSimulationProgram` contains indexed arrays for:

- device capacity, directional bandwidth, and latency;
- aliases, sizes, versions, and retained spill copies;
- task resource, duration, workspace, dependencies, inputs, outputs, and
  mutations;
- memory actions and initial/final residency;
- task/action physical byte deltas;
- physical memory-reuse dependencies.

When admission accounting is enabled, physical deltas replace synthetic
workspace accounting. Fetch destination bytes are charged at action trigger
and released or published at the declared completion transition.

## Output

`ShadowSpillSimulationResult` reports status, structured infeasibility fields,
makespan, spill peak, task intervals, transfer intervals, per-device
object/workspace/total peaks, and capacity shortfalls. Caller-provided
interval, peak, and violation buffers remain caller-owned.

Task and transfer intervals include readiness, start, end, and stall masks.
Stall masks distinguish input residency, device capacity, source readiness,
spill capacity, and physical memory reuse.

A fetch or task launch that does not fit waits for room and is retried,
rather than ending the simulation; only a plan that can never make room
fails, as a deadlock. Each shortfall is reported once, at its first refusal,
as a `ShadowSpillCapacityViolation` carrying the time, device, task, alias,
capacity, used and requested bytes, and a reason. The paired
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
