# Planner C API

The public compiled planner interface is declared by
`planner/include/shadowspill/planner.h` and implemented by
`libshadowspill_planner.so`.

## Purpose

`shadowspill_evaluate_pressurefit_program_context` derives the analytic
residency problem from one resolved simulation topology and runs the complete
deterministic PressureFit policy portfolio. It
reduces residency, places and repairs transfers, validates candidates through
the simulator, computes canonical schedule digests, and returns every
candidate diagnostic plus the fastest valid schedule. Input policy order is
the deterministic tie-break.

The caller resolves recomputation alternatives and supplies one immutable
`ShadowSpillSimulationProgram` containing the selected task/object topology,
declared initial/final residency, and empty action arrays. The evaluator
derives the alias-by-boundary matrices and required/greedy seed internally.

The lower-level `shadowspill_evaluate_pressurefit_context` remains available
for differential tests and callers that already own two immutable dense views
of the same selected program:

- `ShadowSpillResidencyProblem` contains alias-by-boundary residency facts.
- `ShadowSpillSimulationProgram` contains task, object, resource, and transfer
  topology with empty schedule arrays.

`ShadowSpillPressureFitContext` binds those views to the same dense task and
alias identities. JSON-escaped task and alias name payloads are used only to
produce schedule digests identical to the canonical IR serialization.

`shadowspill_reduce_residency` executes PressureFit's deterministic analytic
residency reduction over caller-provided dense alias-by-boundary arrays. It is
an acceleration of the documented policy, not a second planning policy:
schedule emission, repair, simulation, and final selection remain separate.
The result contains the reduced resident bitmap and logical span breaks. The
function never moves or emits transfer directives.

`shadowspill_select_plan` remains available for independently verifying and
selecting a caller-materialized candidate portfolio.

## Ownership and thread safety

- Call `shadowspill_planner_abi_version()` before submitting records. The
  current ABI is version 5.
- Candidate arrays, simulation programs, and every array reachable from a
  simulation program are borrowed only for the call.
- `candidate_results` is optional caller-owned storage and, when supplied,
  needs one entry per candidate.
- The library retains no pointer and owns only temporary simulator buffers.
- Concurrent calls are safe with distinct result buffers.

Both complete context evaluators borrow their context, options, simulation
topology, and identifier arrays only for the call. The lower-level evaluator
also borrows residency facts and seed arrays. On `SHADOWSPILL_PLANNER_OK` or
`SHADOWSPILL_PLANNER_NO_FEASIBLE_CANDIDATE`, its result owns the selected
schedule arrays and candidate array. Release them exactly once with
`shadowspill_pressurefit_context_result_destroy`. Destruction is also safe for
a zero-initialized or partially constructed result.

Neither evaluator creates worker threads. Concurrent context calls are safe
when their input and result records are distinct; the Python planner uses this
property to evaluate recomputation selections in parallel. Timing and cache
counters in `ShadowSpillPressureFitContextResult` are diagnostic only and do
not participate in candidate selection.

For residency reduction, every input pointer is borrowed for the duration of
the call. The caller supplies `alias_count * boundary_count` bytes for both the
resident and break result buffers. A break marks a logical split after a
resident boundary; the last column is unused. On analytic infeasibility, the
result identifies the device, task boundary, required bytes, and capacity.

Successful residency output is canonical: each logical split is encoded at
the final resident cell of the preceding span, including when a nonresident
gap currently separates the spans. This invariant allows later interval-entry
extension to operate directly on dense storage without first materializing
Python span objects.

A nonzero status leaves all caller storage owned by the caller. When no
candidate is feasible, the result includes the first failing candidate index
and simulator status as compact diagnostic context.
