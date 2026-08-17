# Canonical Runtime and Legacy-Path Removal

## Summary

Develop this work independently of the active 2,520-point sweep:

- Base commit: `519b655`
- Branch: `cleanup/canonical-runtime`
- Worktree: `/home/shein/Documents/grad_school/research/shadowspill-cleanup`
- Environment: a separate `shadowspill-cleanup` conda environment
- Plan/log: `docs/internal/plans/cleanup_legacy_stuff/`

The running sweep remains in the original worktree and `shadowspill` environment. The cleanup branch will use separate build outputs, caches, and qualification results; it will not modify the corpus or active sweep outputs.

The result will have one production implementation for runtime construction, task boundaries, worker progress, PressureFit, simulation, serialization, and diagnostics. Backward compatibility and old artifact migration are explicitly out of scope.

## Canonical Architecture

### Generic runtime topology

Replace the two-pool monolithic backend construction with:

- An explicit registry of generic `MemoryPool` backends.
- An explicit registry of directed transfer routes.
- A separate synchronization backend for streams and events.
- A backend-neutral profiler interface.
- Immutable plan-owned bindings selecting the execution pool, spill pool, fetch route, and evict route.

Python `Runtime` will accept arbitrary pool registries rather than requiring exactly one `DevicePool` and one `PinnedHostPool`. The current device and pinned-host implementations remain the initially supported backends.

Runtime creation validates all pools and routes, calibrates supported directions, and performs reverse-order cleanup on partial failure.

`Runtime.close()` will:

1. Reject new work.
2. Wait for active callables and persistent state to be released.
3. Stop and join the C worker.
4. Close routes and pools exactly once.
5. Unregister/free pinned memory and release device memory.
6. Leave the permanently installed PyTorch allocator shim in a closed state that raises a clear error on future allocation.

The same cleanup is invoked idempotently at process exit.

### One admitted task API

Plan adoption creates one immutable task handle per execution task containing:

- Execution ID and semantic name.
- Direct retained input-object references.
- Duplicate-input expansion maps.
- Mutation, output, and handoff records.
- Allocation ABI and memory envelope.
- Task-local retirement storage.
- Predecoded actions and resolved pool/route references.
- Preallocated worker-submission state.

Delete the raw task API and task-ID lookup API. The sole C task interface becomes handle-based `before_task()` and `after_task()`.

### Real frontend boundaries

Training and forward execution share:

```python
prepared = self._before_task(run, entrypoint)
try:
    outputs = self._run_compiled_task(prepared)
    return self._after_task(prepared, outputs)
except BaseException:
    self._abort_task(prepared)
    raise
```

Python `_before_task()` includes runtime acquisition, readiness-event insertion, batched storage rebinding, argument assembly, and all corresponding diagnostics.

Python `_after_task()` includes output classification, mutation/output publication, dematerialization, runtime action submission, binding cleanup, and diagnostics.

Profiling allocation attribution, initial actions, and caller-output handoff receive dedicated APIs instead of creating fake task boundaries.

## Concurrency Contract

### `before_task()`

The C function reads as a short orchestration sequence:

```text
validate handle and failure state
snapshot distinct inputs
insert waits for published readiness events
expand duplicate bindings
open allocation scope
return bindings
```

It must never:

- Acquire a global runtime mutex.
- Perform table or allocation-population scans.
- Wait for the worker.
- Sleep or use a condition variable.
- Allocate host memory.
- Call a backend while holding an object or pool lock.

A nonready input without a generation-matched published event is an immediate runtime invariant error.

### `after_task()`

The C function performs:

```text
validate allocation contract
publish outputs, mutations, and handoffs
record the compute-completion event
attach task-local retirements
reserve action destinations
publish a preallocated worker batch
wait for worker submission acknowledgement
close allocation scope
```

The acknowledgement means asynchronous route operations and their completion events have been submitted. It never waits for transfer completion.

For action-free tasks, no acknowledgement round-trip is required.

### Worker acknowledgement

Use monotonic atomic submission and acknowledgement sequences:

1. Dispatcher publishes a batch with a release store.
2. Worker observes it with an acquire load.
3. Worker submits every causally eligible action and publishes its waitable event.
4. Worker acknowledges with a release store.
5. Dispatcher observes the acknowledgement with an acquire load.

Both threads use `cpu_relax` while actively waiting. The steady hot path contains no condition variables, futexes, `nanosleep`, or `sched_yield`.

The worker remains continuously active while the runtime is open. Its readable hot loop is:

```text
handle newly published submissions
handle completed stream-frontier records
handle retirements and released capacity
```

Each call receives a short explanatory comment. Completed FIFO successors are drained immediately. An incomplete head is queried at most once per 1 us cadence while the worker continues checking other domains; this cadence does not sleep the worker.

### Remove population-dependent work

- Allocator frees append leases directly to the active task's retirement list.
- `after_task()` processes only that list.
- Storage handoffs use predecoded direct records.
- Repeated execution performs no object-ID hashing, task-ID parsing, alias deduplication, action decoding, or full active-lease scans.
- No backend operation occurs under any data-structure lock.

## Legacy Removal Inventory

| Area | Canonical replacement |
|---|---|
| Lifecycle `_legacy` implementations | One final lifecycle implementation |
| Monolithic backend | Pool, route, synchronization, and profiler interfaces |
| Runtime-owned execution/spill roles | Immutable plan-owned role binding |
| Raw and ID-based task APIs | One admitted handle API |
| Python task fallback branches | Handle-only frontend execution |
| Timed input-readiness waits | Worker submission acknowledgement invariant |
| Worker condition waits | Always-active polling loop |
| Global retirement scans | Task-local retirement lists |
| Handoff population scans | Direct admitted handoff records |
| Fake profiling/caller tasks | Dedicated allocation-scope and handoff APIs |
| Rebuilt initial-action arrays | Immutable admitted action batches |
| Python production simulator | Compiled simulator only |
| Python PressureFit fallbacks | Compiled PressureFit only |
| Installed Python reference algorithms | Move under `reference/python/` |
| Synthetic physical-admission traces | Fail closed when allocation evidence is required |
| Legacy Plan/diagnostic readers | Strict current schemas |
| Old NSYS task-label parser | Semantic execution labels only |
| Old benchmark recovery shapes | Current benchmark schema only |
| Historical public documentation | Current architecture only |

Hand-authored logical `Program` objects may still be simulated without physical admission. Publishing an executable callable requires complete physical-allocation evidence.

Missing or incompatible compiled planner/simulator libraries raise immediately; there is no silent Python fallback.

## Implementation Sequence

1. **Isolation and baseline**
   - Create the worktree, branch, cloned environment, and separate build/cache directories.
   - Record the baseline commit, test results, symbol inventory, and active-sweep isolation.
   - Materialize the accepted plan and append-only progress log.

2. **Compiled-only planning**
   - Move Python simulator and PressureFit implementations to the non-installed reference tree.
   - Make every production outcome, including detailed infeasibility, originate from C.
   - Add fail-fast library loading and differential reference tests.

3. **Generic runtime topology**
   - Split backend responsibilities.
   - Implement explicit pool and route registries.
   - Move execution/spill role selection into plan adoption.
   - Complete deterministic lifecycle teardown.

4. **Canonical task handles**
   - Admit immutable direct-reference task records.
   - Add dedicated profiling, initial-action, and caller-handoff APIs.
   - Convert training and forward execution to the shared handle path.

5. **Hot-path concurrency**
   - Add the atomic worker acknowledgement protocol.
   - Remove readiness condition waits and worker sleep.
   - Replace global scans with task-local retirements and direct handoffs.
   - Keep the worker loop short, documented, and nonblocking.

6. **Delete compatibility code**
   - Remove raw/ID APIs, Python fallbacks, legacy serialization readers, old trace parsing, and obsolete tests.
   - Do not add migration utilities.

7. **Diagnostics and documentation**
   - Preserve execution-ID-centered `PlanReport` and `StepDiagnostics` mappings.
   - Ensure frontend timings cover all rebinding, argument, output, and cleanup work.
   - Update separate Python and C references plus runtime/concurrency architecture pages.
   - Add source audits proving no legacy production path remains.

Each structural or behavioral milestone is a separate passing commit.

## Validation

### Focused correctness

- Two-pool and three-pool generic runtime construction.
- Sparse and unsupported route configurations.
- Partial initialization failure and repeated close.
- Immediate consumer after fetch submission.
- Artificially delayed and failed worker acknowledgement.
- Multiple queued transfers acknowledged before wire completion.
- Same-stream and cross-stream readiness.
- Task-local retirement, aliases, views, mutations, and caller handoff.
- Forward and training handle-only execution.
- Profiling and initial actions without fake tasks.
- Old JSON, diagnostics, and trace formats rejected clearly.
- ASan/UBSan and mock ThreadSanitizer lifecycle tests.

### Hot-path gates

- Ready/no-action warmed `before_task()` median: at most 10 us.
- Ready/no-action warmed `after_task()` median: at most 10 us.
- Corresponding p99: at most 25 us.
- Action-bearing acknowledgement excludes wire time and scales only with submitted actions.
- No global runtime mutex or population scan in either boundary.
- No worker `pthread_cond_*`, futex, sleep, or yield in NSYS.
- No backend call while a data-structure lock is held.
- No steady-state host allocation or event creation/destruction.

### End-to-end gates

- Complete Python and C test suites, Ruff, mypy, sanitizers, and `git diff --check`.
- Five approximately-1B correctness cells with numerical and checkpoint parity.
- mlops-Llama 8B at 16 GiB with repeated execution and strict budgets.
- Pure-Qwen 30 GiB selected-task span at most 312.4 ms.
- Simulator/runtime action, byte, and ordering equality.
- Simulator fidelity within 5% on retained full-model cases.
- No throughput or selected-span regression greater than 5%.
- Updated NSYS trace showing semantic task IDs, one task boundary per execution, an always-active worker, and no hidden inter-task work.

## Assumptions

- The active sweep remains pinned to `main`, commit `519b655`, and the existing `shadowspill` environment.
- The cleanup worktree never installs into or rebuilds libraries inside that environment.
- One runtime worker remains the default topology.
- Worker CPU consumption is intentionally traded for microsecond responsiveness.
- Cold shutdown may block while joining the worker; steady task execution may not sleep.
- No compatibility promise exists for C APIs, internal Python APIs, serialized artifacts, or historical traces.

## Accepted terminology and shared-residency additions

- The PyTorch state boundary is named `import_*` when storage enters runtime
  ownership and `export_*` when storage leaves it. All relocate/externalize
  names are deleted without compatibility aliases.
- Imported storage receives a stable runtime-object identity. Passing the same
  imported model to multiple planners binds each plan-local object key to that
  same retained runtime object; equal tensor values do not imply sharing.
- Runtime objects may hold one of two reference-counted shared-residency
  policies in a named pool:
  - `SHARED_READ_ONLY` guarantees residency and rejects every declared
    mutation during admission.
  - `SHARED_WRITABLE_UNORDERED` guarantees residency, permits only
    stable-address in-place mutation, and deliberately inserts no
    inter-callable read/write ordering. Concurrent readers may observe old,
    new, or partially updated data. A replacement mutation that would publish
    a different lease fails admission.
- PressureFit excludes either shared policy from movable-object decisions and
  emits no fetch, evict, or release directives for the shared lease.
- Shared bytes are hidden from per-plan movable-memory accounting by default,
  but remain visible and charged exactly once in the runtime's physical pool
  ledger. Admission reports the external shared footprint and the remaining
  plan capacity explicitly.
- The final shared reference owns release eligibility. Generations, leases,
  and allocator safety remain validated even when value ordering is
  intentionally disabled.
