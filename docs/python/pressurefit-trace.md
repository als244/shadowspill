# PressureFit cut traces

PressureFit can record every cut in selected candidates, including the base
reduction and later repairs. Tracing is off by default. The recording is a
single-device, deterministic rerun after the ordinary planning API obtains
its answer. The public phase API reads existing ProfiledPrograms and runs
entirely on CPU; no allocator, model, or process group is required.

The fixed ascending key is `E, writeback, -slack, -size`, then alias, start,
and cut ID. Signed slack is maximized without clamping; exposure sums separately
clamped eviction and fetch delays. Transfer strategies omit `E`. See the
[formulas and policy definitions](../architecture/pressurefit.md).

```python
from shadowspill import CommunicationInfo, MemoryBudgets, ProfiledProgram, plan_program
from shadowspill.planner import GenericPlanningOptions, SearchOptions

answer = plan_program(
    ProfiledProgram.load("measured.ssp"),
    memory_budgets=MemoryBudgets(execution_bytes=16 << 30, spill_bytes=112 << 30),
    communication_info=CommunicationInfo.load("communication.ssp"),
    search_options=SearchOptions(
        generic=GenericPlanningOptions(deterministic=True),
        trace_directory="/tmp/pressurefit-recordings",
    ),
)
```

Each call creates a unique `pressurefit-*` subdirectory. The directory contains `program.json`, `manifest.json`, and a binary
`.pftrace` file per requested candidate. `trace_candidates=((selection_id,
candidate_id), ...)` selects explicit pairs from the original diagnostics;
omitting it records the winner. It does not narrow the ordinary search.

`trace_directory` and `trace_candidates` are outside `SearchOptions.to_dict()`,
stored requests and plan keys. The API returns the original plan and verifies
each rerun's candidate status, makespan and schedule digest against its original
diagnostic. Internal step recording does not change the caller's keyed
`record_reduction_steps` option. Failed recording or a mismatched rerun raises
an error and leaves a manifest marked `failed`, never `complete`.

Set `deterministic=True` explicitly when recording traces, as above.
An isolated replay cannot reproduce the shared placement gate of a
nondeterministic search, so that mode is refused. A winning incumbent has no
local cut history: automatic winner tracing refuses it, while explicit
`trace_candidates` can record evaluated local candidates. Multi-device tracing
is refused. Existing files are never overwritten by the C recorder.

## Read and replay

```python
from pathlib import Path
from shadowspill.diagnostics.pressurefit_trace import PressureFitTrace, read_manifest

directory = Path("/tmp/pressurefit-recordings/pressurefit-...")
manifest = read_manifest(directory)
candidate = manifest["candidates"][0]
trace = PressureFitTrace.read(directory / candidate["file"])
trace.verify()                         # checks every reduction against C digests
seed = trace.state(0, 0)
after_first_cut = trace.state(0, 1)     # when this reduction has at least one cut
final = trace.state(trace.answer_reduction)  # when this candidate has an answer
for frame in trace.iter_states(0):
    draw(frame.spans, frame.pressure, frame.capacity)
```

`state(reduction, cut_count=None)` returns that reduction's final state.
`iter_states` yields the seed and each successive cut without retaining a dense
cuts-by-boundaries history. A `TraceState` has `reduction`, `cut_count`, per-alias
inclusive `spans`, `pressure` and `capacity`. Its arrays are read-only.

Every reduction restarts from the seed. Its capacity and extra-pressure changes
carry forward; its cuts do not. `step_reductions` links each simulated candidate
step to the reduction it used. A fetch-only repair can reuse a reduction.
`answer_step` and `answer_reduction` explicitly identify the returned answer,
which can precede the last attempted reduction. Both are `None` for a candidate
without an answer. Residency alone does not describe fetch timing changes.

Pressure is the reducer's analytic charge, including output reservations and
extra repair pressure. Capacity is the boundary capacity after workspace and
resident-slice reservations. The current search represents capacity refinement
as uniform extra pressure; do not subtract it from capacity a second time.
These arrays are not a physical execution occupancy trace or the full process
memory budget.

The binary cut stream describes reducer decisions and analytic pressure.
It does not record emitted actions or their simulated completion times. Join
the emitted schedule for a candidate step to study its execution timing;
the final answer alone does not describe every earlier reduction.

Before each cut the trace records up to three **over-budget boundaries**, sorted
by excess descending and then position ascending; the first is the trigger.
It also records the selected cut and up to three best rejected active cuts
relieving that trigger. `trace.comparisons(reduction, cut)` names the first
ranking criterion distinguishing each alternative from the winner, with both
values. It explains the local greedy decision, not a global optimality claim.

## Semantic labels and positions

Each candidate's `tasks` records `profiled_duration_ns` from the selected
profile. For candidates with an answer, it also records `plan_ready_ns`,
`plan_start_ns`, `plan_end_ns`, `plan_duration_ns` and `plan_stall_ns` from
simulating that candidate's returned schedule at the requested capacity.
These plan-time fields are `None` when the candidate has no answer.
`plan_makespan_ns` identifies this clock; it can differ from the C diagnostic's
makespan when the search evaluated the schedule under a refined capacity.
These are simulated times, not runtime measurements or times of cut events.

The saved input program also retains unselected alternative task profiles.
The generic renderer matches the save/recompute backward tasks within each
alternative group and shades a lime prefix of length
`max(0, recompute_backward_ns - save_backward_ns)`. Task-index views use the
same fraction of the task width. Both segments carry the stage and microbatch
label, such as `R16 · mb14` and `B16 · mb14`. The analysis keeps the signed difference and
matching baseline task ID; it does not substitute another stage's duration
when a unique pair is unavailable. This applies to any model exposing those
alternatives and does not require model-specific task names.

The analyzer also derives each selected task's `required_memory`: input,
output, and mutated-object counts and their unique alias bytes, plus profiled
workspace. Its total unions the three alias sets before adding workspace;
shared views and in-place mutations do not double count. The video decision
panel displays this breakdown for the task immediately after its boundary.

The collection tools add an optional `collection_context.json` sidecar with
the same program digest, workload/geometry/ordering labels, and physical and
usable planning budgets. Those annotations identify the winning configuration
in the video; they do not alter the program or plan. See the
[tool guide](../../src/tools/pressurefit_trace/README.md) for physical-budget
overrides and recovery from capacity failures during geometry profiling.

When semantic stage metadata is available, `read_manifest` adds `stages`
grouped by `(phase, stage_index, microbatch)`. Each group retains its `task_ids`,
summed `profiled_duration_ns` and `plan_duration_ns`, earliest `plan_start_ns`,
latest `plan_end_ns`, and `plan_span_ns` (end minus start). Summed compute
duration and elapsed span have different meanings; a span can include gaps.

Binary array **position 0** precedes task 0. Position `p > 0` follows task
`p - 1` and, unless terminal, precedes task `p`. The algorithm's signed boundary
number is **position minus one**. Cut endpoints use that signed convention;
span endpoints and pressure-operation ranges use array positions.

The manifest's `contexts` stores each selected resolution's `selection_id`,
`task_ids` in resolved execution order, and `aliases` table once. Candidates
refer to it with `context_index`; `read_manifest` exposes `task_ids` and
`aliases` on each candidate as well, sharing the same alias table in memory.
Older manifests with these tables directly on candidates remain readable.
Current contexts also include `pressure_accounting`: the resident alias indices
and resident slice size returned by the search, and the aliases required to
finish on the device. `resident_reservations` holds rows of alias index,
generation count, and reserved bytes including alignment padding;
`resident_alignment_bytes` records the alignment. Their sum is checked against
the C resident slice size. These facts let the visualization attribute initial
pressure to roles and check the result against the binary trace. Subsequent
pressure operations identify their alias through the associated cut; repair
pressure remains a separate component. The binary format is unchanged. Older
contexts without this metadata still support exact total-pressure replay.
The visualization joins task workspace from the captured program's profiles.
It adds workspace as a separate stack layer beneath a flat planning budget,
preserving the recorded excess at each boundary. The default view also includes
resident reservations by role, including aliases hidden by the drawing filters.
Optimizer-state rows are retained even when they have only a required-use
interval, subject to the size threshold. Their pressure counts only the current
device residency and reservations, not all optimizer storage across the model.
Candidate `tasks` retain phase/profile IDs and individual plan timings.
The reader derives `boundaries` with both `after`
and `before` tasks; timing records are stored only once per task.
Never join through the unresolved program's task position. An alias index maps
through the indexed template's `alias_ids`, not through the position of the
alias group in the input document. Each alias entry contains its `alias_id`,
display `label`, `role`, `category`, and all member `objects`, including their
producer, consumers, size, offset and persistence. The primary label uses the
occupancy tool's role priority; all shared-storage member identities remain.
Persistent parameters, optimizer state, gradients, and buffers take precedence
over saved activation views. The renderer applies the same rule to existing
recordings, preserving the storage's stable name and role across task-local
views. Required-use bands still include every member's actual task references;
a shared buffer is counted once, even when several microbatches save views of it.

Saved Programs retain semantic source names and task metadata. The trace
collection tools join those names to the selected task IDs and write
`semantic_labels.json` beside each trace manifest. Missing names remain absent;
numeric IDs are not interpreted as module names.

Other producers may supply an explicit annotation map:

```python
manifest = read_manifest(directory, semantic_labels=labels)
```

Place that object in `semantic_labels.json` beside the manifest;
`read_manifest` reads it automatically. The `program_digest` must match. An
activation's output origin is joined to the candidate's active producer, so
save/recompute alternatives are labelled correctly. `boundary_labels(tasks,
task_metadata)` exposes the same ID-based join separately. Other frontends can
provide the same `program_digest`, `tasks` and `objects` annotation maps.

## Binary format, version 4

The reader accepts the fixed split-exposure/slack comparator. The descriptive
properties `PressureFitTrace.cut_score` (`split-exposed`) and `cut_ranking`
(`slack-after-writeback`) identify it; they are not planner options.
`ranking_criteria` and `ranking_key(reduction, cut, rank=0)` expose the exact
ascending comparator, including the final cut-ID fallback.
Version 4 adds the allocation or placement that prompted each reduction.
Version 3 recordings remain readable; their reduction reason is unknown.
`object_capacity_bytes(reduction)` returns the object budget recorded in that
pass's search steps. Repair-only passes may inherit the last recorded value;
capacity refinements do not. It returns `None` when no placement budget was
recorded or an unlinked pass's capacity cannot be established exactly. This is
distinct from the per-boundary capacity after workspace/residency accounting.

For source-tree capture, analysis, and animation commands, see the
[PressureFit trace tools](../../src/tools/pressurefit_trace/README.md). The tools
provide profiled-time and equal-width task-index views, semantic object labels,
per-cut pressure replay, and explanation pauses comparing the selected cut with
recorded alternatives.

The same [tools directory](../../src/tools/pressurefit_trace/README.md) also
contains separate renderers for emission/placement and simulated execution.
Those [plan videos](../../src/tools/pressurefit_trace/video/plan/README.md) read the
saved `AnnotatedProgramPlan`, optionally joined with semantic labels. A cut
trace by itself does not provide physical addresses or copy timings. Both
guides list input formats, commands, timing controls, and encoding arguments.

Eight magic bytes `PFTRACE\0`, then array blocks. Each block header is little
endian `(kind: u32, width: u32, rows: u64)`, followed by `rows * width` little
endian u64 words, with no native struct padding. Signed fields use two's
complement i64. No device column is stored: the trace has exactly one device.

| Kind | Width | Contents |
| --- | ---: | --- |
| 0 | 8 | One row: version (4), local problem index, strategy, fetch rule, reserved (0), alias count, boundary count, comparator identifier (2). Policy codes are the public C enums. |
| 1 | 1 | One alias size in bytes per alias index. |
| 2 | 3 | Seed rows: alias index, inclusive span start/end positions. Touching spans retain their break. |
| 3 | 2 | Once: capacity, base pressure per boundary position for this strategy. |
| 4 | 1 | Begin reduction: sequential ID; 0 is base, later IDs are repair attempts. |
| 5 | 3 | Sparse updates: position, capacity, extra pressure; unchanged entries carry forward. Extra starts at zero. |
| 6 | 96 | Cut rows, described below; up to 128 rows per streamed block. |
| 7 | 4 | Pressure ops: cut index within reduction, first/last inclusive positions, signed sign (+1/-1). Multiply by the selected alias size. Up to 512 rows per block. |
| 8 | 7 | End reduction: ID, status, signed error boundary, required bytes, capacity bytes, residency digest, pressure digest. Error fields are meaningful on analytic infeasibility. |
| 9 | 2 | Candidate step index, reduction ID used by that step. |
| 10 | 7 | Final step diagnostics: makespan ns, capacity bytes, required bytes, flags, repairs, simulator status, capacity violations. Flags are `SHADOWSPILL_STEP_*`; required bytes is meaningful when measured. |
| 11 | 3 | Candidate status, answer step (UINT64_MAX for none), makespan ns. |
| 12 | 9 | Immediately after kind 4: reason kind, failure status, task index, alias index, memory location, requested bytes, used bytes, capacity bytes, shortfall bytes. |

Reason kinds are 0 (initial seed), 1 (simulation pressure repair), 2 (admission
pressure repair), 3 (object-pool layout refinement), and 4 (escalation rollback).
Task/alias indices can be absent (`UINT32_MAX` or `UINT64_MAX`). Location uses
the memory-location enum: execution 0, spill 1. `reduction_reason(index)` decodes
these fields. For task-start failures, the alias identifies the **first output**;
the failed request covers all new outputs and workspace together, so it is not
evidence that this output alone caused the shortfall. Layout refinements concern
the whole object pool and do not name an individual object.

Cut row columns:

| Columns | Meaning |
| --- | --- |
| 0, 1 | Trigger position and its used bytes before applying the cut. |
| 2 | Number of rejected alternatives (0–3). |
| 3–50 | Four 12-word choices: winner, then alternatives, then zero padding. Each is cut ID, alias index, signed start/end, exposed ns, writeback, two reserved words, negative alias size, one reserved word, alias index, signed start. Reserved words are ignored. |
| 51 | Number of recorded over-budget boundaries (1–3). |
| 52–63 | Three 4-word boundary entries: position, used bytes, capacity bytes, excess bytes; zero padding after the count. |
| 64–95 | Eight words for each of the same four choices: prior-use end, next-use begin, eviction duration (zero for release), fetch duration, eviction overlap window, fetch overlap window, exposed eviction, exposed fetch. All are nanoseconds; missing choices are zero-padded. |

`exposed_ns` is the saturated u64 sum of the two exposed
components for stall strategies, and zero for transfer-only strategies.
The reader validates each component's `max(0, duration - slack)` and their sum.

A cut with `start <= end` clears cells `start + 1 .. end + 1`. A cut with
`start > end` splits a span after position `end + 1`; it can change pressure
without removing residency cells. Pressure changes must be replayed from the
operations, not guessed from removed span lengths. Add/subtract operations
use u64 modular arithmetic, matching C.

Stall strategies compare `exposed_ns` first; transfer strategies omit it.
Both then compare writeback, negative signed slack, negative alias size, alias
index, signed start, and cut ID. Signed slack is computed exactly from the
recorded components as eviction window + fetch window − eviction duration −
fetch duration, without rounding or clamping. Exposure is a local estimate,
not the final simulation's transfer stall.

Kinds 6 and 7 have independent buffers and can interleave; concatenate each
kind in its own order inside a reduction. Every operation names its cut.
Each reduction ends with independent C FNV-1a digests: offset basis
14695981039346656037 and prime 1099511628211, bytewise over little-endian u64
words. Residency hashes each alias's span count followed by its start/end
pairs. Pressure hashes every final boundary value from the reducer's tree,
including extra pressure. The reader checks block structure and rejects
truncated/unknown formats; `verify()` additionally checks replay digests,
trigger/top-boundary rankings and recorded score ordering.

The manifest's candidate entries also retain the original selection and policy
IDs, original problem index and rerun-local index, recompute share, candidate
diagnostic and `verified_against_original`. Its top-level fields identify the
schema, completion status, source program/digest, original search options and
simulation configuration, and the original answer's IDs/makespan/digest.
The input program and original configuration are provenance, not extra event
columns. `read_manifest` rejects recordings that did not complete.

`resolution_recompute_fraction` is the exact recompute fraction among flexible
groups, using the planner's resolution definition (for example `1/4`). The
`recompute_share` field counts all groups, including forced choices,
and can therefore differ. Videos use the resolution fraction when available.
