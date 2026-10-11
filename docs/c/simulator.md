# Simulator C API

Include `<shadowspill/simulator.h>`. The simulator is a deterministic,
standalone evaluator for an already selected schedule.
`SHADOWSPILL_SIMULATOR_NO_INDEX` is its absent-index sentinel.

## Input

`ShadowSpillSimulationProgram` contains indexed arrays for:

- one `ShadowSpillSimulationDevice` per device, holding its capacity and its
  `fetch_solo_bandwidth_bytes_per_second`,
  `fetch_concurrent_bandwidth_bytes_per_second`,
  `evict_solo_bandwidth_bytes_per_second`,
  `evict_concurrent_bandwidth_bytes_per_second`, and per-direction latency;
  all four rates must be positive. Latency fields remain calibration metadata
  and do not affect timing. Payload progress switches rates when the reverse
  lane becomes active or finishes. See [timing semantics](../architecture/simulation.md#solo-and-concurrent-bandwidth);
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
spill capacity, physical memory reuse, and a busy lane. Two of those sound
alike and are not: memory reuse is an ordering wait the plan created, for the
eviction or release that frees the allocation about to be reused, while device
capacity is a shortfall -- no room at all for a fetch to land or for a task's
outputs and workspace. A plan that arranges its own capacity waits on the
first and never reaches the second.

A busy lane is the one wait a plan does not arrange and cannot avoid: the
copy is eligible and the lane is carrying another. It is set on the copy at
the head of that lane's queue, which is the one actually waiting, and which
every copy behind it becomes before it starts.

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

`shadowspill_simulate(const ShadowSpillSimulationProgram *program,
ShadowSpillSimulationResult *result)` validates and evaluates one indexed
program, returning a `ShadowSpillStatus`. `program` is borrowed; every buffer on
`result` is the caller's, and each is filled up to its capacity field.

`shadowspill_check_schedule(const ShadowSpillSimulationProgram *program,
const uint8_t *alias_flags, ShadowSpillScheduleCheck *result)` holds the
schedule bound into `program` to the schedule's own rules, independent of
timing, and writes the first refusal to `result`, or
`SHADOWSPILL_SCHEDULE_REFUSAL_NONE`. The rules are the ones
`MemorySchedule.validate` states, in the same order: the initial residency
names no runtime-owned group; every action moves a movable group that holds
bytes, in trigger order; every input is on the device when its task reads it;
a fetch needs a current spill copy and no device copy, and an evict, release
or write-back needs a device copy; a release may not drop the only current
copy of a value a later task reads or the final residency names; and the final
residency is reached. `alias_flags` holds one `ShadowSpillAliasCheckFlags`
value per alias for what the compiled program does not carry: whether the
runtime owns the group's residency, whether it holds no bytes, and whether it
keeps its spill copy. `index` names the initial or final residency entry, the
action, or the task the refusal points at, and `alias` the group. The call
returns `SHADOWSPILL_STATUS_OK` once the check has run, whatever its verdict,
and `SHADOWSPILL_STATUS_INVALID_ARGUMENT` for a malformed program.

`shadowspill_abi_version()` and `shadowspill_status_string()` cover loading
and diagnostics for this boundary as for every other; see the
[C API guide](README.md#abi-use).

Neither call performs I/O, owns external storage after return, or uses global
mutable state. Distinct result buffers can be evaluated concurrently.
