# Planner C API

The public compiled planner interface is declared by
`planner/include/shadowspill/planner.h` and implemented by
`libshadowspill_planner.so`.

## Purpose

`shadowspill_select_plan` independently simulator-verifies a materialized
PressureFit candidate portfolio and returns the valid candidate with the lowest
makespan. Input order is the deterministic tie-break. Candidate and selection
identifiers are opaque integers copied to the result.

The policy constructing candidate schedules is intentionally outside this ABI.
Consequently the ABI stays independent of evolving search heuristics while
making the final choice reproducible from any language.

`shadowspill_reduce_residency` executes PressureFit's deterministic analytic
residency reduction over caller-provided dense alias-by-boundary arrays. It is
an acceleration of the documented policy, not a second planning policy:
schedule emission, repair, simulation, and final selection remain separate.
The result contains the reduced resident bitmap and logical span breaks. The
function never moves or emits transfer directives.

## Ownership and thread safety

- Call `shadowspill_planner_abi_version()` before submitting records. The
  current ABI is version 2.
- Candidate arrays, simulation programs, and every array reachable from a
  simulation program are borrowed only for the call.
- `candidate_results` is optional caller-owned storage and, when supplied,
  needs one entry per candidate.
- The library retains no pointer and owns only temporary simulator buffers.
- Concurrent calls are safe with distinct result buffers.

For residency reduction, every input pointer is borrowed for the duration of
the call. The caller supplies `alias_count * boundary_count` bytes for both the
resident and break result buffers. A break marks a logical split after a
resident boundary; the last column is unused. On analytic infeasibility, the
result identifies the device, task boundary, required bytes, and capacity.

A nonzero status leaves all caller storage owned by the caller. When no
candidate is feasible, the result includes the first failing candidate index
and simulator status as compact diagnostic context.
