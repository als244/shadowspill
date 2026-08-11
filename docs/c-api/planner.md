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

## Ownership and thread safety

- Call `shadowspill_planner_abi_version()` before submitting records. The
  current ABI is version 1.
- Candidate arrays, simulation programs, and every array reachable from a
  simulation program are borrowed only for the call.
- `candidate_results` is optional caller-owned storage and, when supplied,
  needs one entry per candidate.
- The library retains no pointer and owns only temporary simulator buffers.
- Concurrent calls are safe with distinct result buffers.

A nonzero status leaves all caller storage owned by the caller. When no
candidate is feasible, the result includes the first failing candidate index
and simulator status as compact diagnostic context.
