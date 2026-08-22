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
makespan, spill peak, task intervals, transfer intervals, and per-device
object/workspace/total peaks. Caller-provided interval and peak buffers remain
caller-owned.

Task and transfer intervals include readiness, start, end, and stall masks.
Stall masks distinguish input residency, device capacity, source readiness,
spill capacity, and physical memory reuse.

## Functions

- `shadowspill_simulate()` validates and evaluates one indexed program.
`shadowspill_abi_version()` and `shadowspill_status_string()` cover loading
and diagnostics for this boundary as for every other; see the
[C API guide](README.md#abi-use). `shadowspill_simulator_abi_version()` and
`shadowspill_simulation_status_string()` remain as
thin aliases of them.

The call performs no I/O, owns no external storage after return, and uses no
global mutable state. Distinct result buffers can be evaluated concurrently.
