# Qwen Runtime-Overhead Investigation

> **Historical, non-normative investigation.** This report preserves evidence
> from the implementation under investigation. Current behavior is defined by
> the [memory runtime architecture](../architecture/memory-runtime.md) and
> [diagnostics API](../python/api/diagnostics.md).

## Purpose

This note explains why one recurrent Qwen 3.5 training step took materially
longer under ShadowSpill than an equivalent whole-objective compiled PyTorch
step. It preserves the Phase-1 investigation record: established causes are
separated from residual time that required controlled measurement.

The workload is the pure-PyTorch Qwen 3.5 numerical configuration, two
microbatches, one optimizer update, and a 30 GiB physical device cap. At this
budget PressureFit selects no in-step H2D action. ShadowSpill v1 is non-cyclic,
however, so every recurrent invocation restores the schedule's initial device
set from host and writes most of that set back to host at the end.

## Clocks used

Every one of the 129 selected tasks has seven timestamps:

| Clock | Timestamp |
|---|---|
| host | `before_task` entry and exit |
| compute stream | `before_readiness_waits` |
| compute stream | `before_task_compute` |
| host | `after_task` entry and exit |
| compute stream | `after_task_compute` |

All tasks therefore have a measured readiness interval. A task inserts an
actual readiness dependency only when at least one input is still prefetching;
the runtime records one `readiness_wait` event for every corresponding
`cuStreamWaitEvent`. If every input is already device-ready, the two stream
markers are placed back-to-back and no readiness event is recorded.

The first diagnostic implementation used debug-only `cuLaunchHostFunc`
callbacks. That historical trace is useful for the root-cause evidence below,
but the callbacks proved invasive. Current tracing retains the four native
host timestamps and records the three stream boundaries with preallocated
CUDA events. A residency stall is established only when the neutral runtime
ledger also contains a `readiness_wait` event for that task.

## Process threads and CUDA streams

For each initialized device runtime, the application currently has these
relevant execution agents:

| Agent | Created by | Count per device | Responsibility |
|---|---|---:|---|
| PyTorch dispatcher | user/PyTorch | normally one calling thread | executes the returned callable, calls synchronous allocator and task-boundary APIs, and launches numerical CUDA work |
| ShadowSpill progress worker | ShadowSpill | one POSIX C thread | dispatches H2D/D2H actions, observes transfer/task/retirement completion, publishes residency, releases slab ranges, and wakes blocked allocators |
| H2D CUDA stream | ShadowSpill | one | serializes host-to-device copies and their completion events |
| D2H CUDA stream | ShadowSpill | one | waits on producer task fences, then serializes device-to-host copies and their completion events |
| compute CUDA stream(s) | PyTorch/user | one or more | executes numerical kernels and waits on input readiness events |

CUDA streams are GPU command queues, not additional CPU threads. CUDA may
have private driver threads, but ShadowSpill does not create or control them.
ShadowSpill currently spawns no separate allocator, H2D, D2H, binding, or
diagnostics thread.

One progress worker remains the sensible default. Its CPU work should be
small state transitions around asynchronous CUDA operations; adding workers
would not make either FIFO copy stream transfer two objects concurrently and
would make plan action order and failure propagation harder to reason about.
The measured failure is not insufficient worker parallelism. It is that this
one worker performs an unbounded full-state scan while holding a lock needed
by the dispatcher. After queue ownership and completion tracking are fixed,
we should add a second worker only if traces show CPU submission on one
direction delaying an independently ready opposite-direction transfer. No
current evidence shows that.

The progress worker sleeps on a condition variable when no completion can be
pending. With in-flight work it polls completion after a timed backoff. A new
action, logical free, close, failure, or allocation-pressure transition wakes
it. The PyTorch dispatcher must never poll for transfer readiness: it inserts
`cuStreamWaitEvent` into its compute stream and continues dispatching until a
real allocation-capacity dependency requires it to block.

## Original observation

| Measurement | Time |
|---|---:|
| whole-objective compiled PyTorch compute interval | 292.141 ms |
| ShadowSpill simulator compute completion | 275.975 ms |
| original real ShadowSpill compute interval | 493.626 ms |
| ShadowSpill simulator terminal D2H completion | 500.521 ms |
| original real public-call wall time | 639.724 ms |

Summed real CUDA-event durations for the compiled ShadowSpill tasks were about
286 ms. Thus most of the 201.485 ms standard-to-ShadowSpill compute-interval
difference was time between numerical task intervals, not slower kernels.

The original trace recorded 311 host-condition readiness waits but only two
compute-stream event waits. There were no allocator capacity waits. Two task
boundaries exposed the dominant error:

| Task | Semantic task | Host time inside `before_task` |
|---|---|---:|
| `task_000001` | microbatch-0 first forward stage | 64.824 ms |
| `task_000017` | microbatch-0 terminal loss-stage recomputation/backward | 81.252 ms |

For task 17 the original timestamps were:

| Boundary | Monotonic timestamp (ns) |
|---|---:|
| host enters `before_task` | 4,396,386,301,497,938 |
| stream reaches `before_readiness_waits` | 4,396,386,301,607,445 |
| host exits `before_task` | 4,396,386,382,749,869 |
| stream reaches `before_task_compute` | 4,396,386,382,929,801 |

The Python dispatch thread was therefore unable to enqueue later tasks for
81.252 ms. This was not allocation pressure: it was waiting for the transfer
service to make an input's CUDA event available.

## Root cause: a one-transfer dispatch window

At invocation entry the runtime reserved all 395 admitted device destinations
in 0.14 ms and queued 6,041,784,940 bytes of H2D restoration. The progress
service nevertheless admitted only one H2D action to the CUDA H2D stream at a
time. It did not dispatch the next copy or create its readiness event until the
previous copy had completed and been reaped.

This contradicted the intended causal model:

```text
incorrect
host:       wait for transfer 1 completion -> obtain event 2 -> enqueue task wait
H2D stream: [copy 1] [copy 2] [copy 3] ...

correct
host:       obtain event 1, event 2, event 3 -> enqueue task waits -> run ahead
H2D stream: [copy 1] [copy 2] [copy 3] ...
compute:          wait(event needed by task) -> task
```

CUDA streams already serialize the copies in FIFO order. The extra software
one-in-flight gate provided no ordering guarantee and prevented the Python
thread from expressing dependencies asynchronously.

## Behavior correction

The progress service now dispatches the complete admitted H2D and D2H windows
onto their respective streams. Every copy leases a pooled event. `before_task`
inserts a compute-stream wait on each unfinished input event and returns; it
does not wait for the copy on the host. Event-pool admission now includes the
complete initial and scheduled transfer windows, so steady state creates no
events.

An isolated delayed mock-backend regression proves that two H2Ds may both be
enqueued while the first remains in flight, and that acquiring the second
object returns in under 30 ms by inserting an event wait rather than waiting
for either transfer on the host.

After the correction:

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| median untraced recurrent interval | 642.580 ms | 542.809 ms | -99.771 ms |
| traced compute interval | 503.656 ms | 468.280 ms | -35.376 ms |
| sum of host `before_task` time | 165.209 ms | 15.571 ms | -149.638 ms |
| host-condition readiness waits | 311 | 0 | -311 |
| compute-stream readiness waits | 2 | 27 | +25 |
| summed task CUDA durations | 286.281 ms | 290.923 ms | +4.642 ms |
| allocation-pressure waits | 0 | 0 | unchanged |

The traced and untraced clocks are not directly interchangeable because the
three per-task stream callbacks perturb small tasks. The causal counters are
unambiguous: the dispatch thread no longer blocks for readiness.

## Remaining startup stalls

All 129 tasks recorded the two readiness timestamps. Only three tasks inserted
one or more actual readiness-event dependencies:

| Task | Semantic task | Objects waited on | Readiness-to-compute interval |
|---|---|---:|---:|
| `task_000001` | microbatch-0 first forward stage | 14 | 58.159 ms |
| `task_000017` | microbatch-0 terminal loss-stage backward | 1 | 44.910 ms |
| `task_000003` | microbatch-0 second forward stage | 12, already complete | 0.0029 ms |

The task-1 delay is the known non-cyclic startup cost. Task 17 reveals a
separate startup-ordering problem. Its only waited object is the 4-byte FP32
cotangent seed generated by AOTAutograd for the scalar cross-entropy objective,
conceptually `d(loss)/d(loss) = 1`. It is neither a model activation nor
workspace. ShadowSpill creates it from `torch.ones_like(loss)` and retains host
spill storage for reuse by both microbatches.

The scalar was the last of the 395 initial-device aliases in the startup queue:

| Event for objective cotangent | Time from trace start |
|---|---:|
| queued as startup prefetch | 4.571 ms |
| 4-byte copy dispatched to H2D stream | 5.560 ms |
| task 17 reaches `before_readiness_waits` | 117.227 ms |
| copy completes behind prior startup traffic | 162.056 ms |
| task 17 reaches `before_task_compute` | 162.137 ms |

The 4-byte transfer itself does not take 44.9 ms. It waits behind earlier
startup copies on the single H2D stream. The initial placement order currently
follows object-table order rather than first-use order. This is why a large
30 GiB cap can still expose a backward-stage startup stall even though the
PressureFit schedule contains no in-step H2D action.

The complete raw descending interval table is saved by qualification at
`qualification/results/phase1/qwen35_transfer_window_fix_readiness_gaps.csv`.
The next-largest raw intervals have no runtime readiness event and are at most
0.458 ms; they are debug callback overhead/jitter, not object stalls.

## Terminal cotangent correction

The scalar is a terminal differentiation seed, so it does not need to cross
the public backward ABI. Capture now specializes only a proven terminal scalar
unit cotangent into the backward graph using a device-relative `aten.new_ones`
operation. This is objective-agnostic: it applies to any differentiable scalar
objective, including native cross entropy, fused head losses, diffusion MSE,
composite primary/auxiliary losses, and custom autograd. Cotangents connecting
internal stages are not specialized because they contain real activation
gradients.

The same Qwen control after specialization recorded:

| Measurement | Before | After |
|---|---:|---:|
| startup H2D objects | 395 | 394 |
| startup H2D bytes | 6,041,784,940 | 6,041,784,936 |
| preceding-task completion to task-17 compute | 44.922 ms | 0.288 ms |
| task-17 actual readiness dependencies | 1 | 0 |

Exact post-correction boundaries were:

| Boundary | Monotonic timestamp (ns) |
|---|---:|
| preceding task `after_task_compute` | 4,399,056,440,412,880 |
| task 17 `before_readiness_waits` | 4,399,056,440,415,689 |
| task 17 `before_task_compute` | 4,399,056,440,700,728 |

The 0.285-ms readiness-marker interval contains no runtime readiness event and
is callback scheduling overhead, not an H2D stall.

## Corrected end-to-end ledger

After the transfer-window, terminal-cotangent, object-index, and progress-lock
fairness corrections, one traced recurrent invocation has the following
causal ledger. Times use the native monotonic trace, not NSYS timestamps.

| Interval | Time |
|---|---:|
| residual terminal D2H paid by the following invocation | 38.881 ms |
| first task's readiness marker to its compute marker | 59.538 ms |
| first task compute marker to final optimizer completion | 384.361 ms |
| sum of all 129 per-task CUDA-event intervals | 299.281 ms |
| idle between those task intervals | 85.079 ms |

The resulting non-cyclic steady cycle is approximately
`38.881 + 59.538 + 384.361 = 482.780 ms`. The independently measured untraced
median is 476.734 ms. The small difference is expected because the callback
trace perturbs the very stream it observes.

The compiled standard-allocator authority takes 292.141 ms. The sum of real
ShadowSpill task intervals is only 7.140 ms larger. The arithmetic kernels are
therefore not the source of the approximately 185-ms steady-cycle regression:
it is the deferred non-cyclic cooldown, next-step startup, and inter-task host
work.

The transfer ledger makes the boundaries concrete:

| Direction | Objects | Bytes | First dispatch/completion event | Last completion |
|---|---:|---:|---:|---:|
| startup H2D | 394 | 6,041,784,936 | 2.902 ms | 163.830 ms |
| terminal D2H | 388 | 6,041,733,704 | 365.004 ms | 485.856 ms |

Terminal D2H begins before the optimizer completes, so most writeback overlaps
the end of the numerical program; 38.881 ms remains after the final optimizer
compute marker. The next non-cyclic invocation waits for that remainder before
restoring its initial set.

## The Qwen kernel storm is not created by ShadowSpill

A warm NSYS control used the same pure-PyTorch Qwen model, data, optimizer, and
five prior warm steps under the standard allocator. Neither capture contained
JIT compilation.

| Warm NSYS capture | Standard allocator | ShadowSpill |
|---|---:|---:|
| kernels | 40,681 | 40,528 |
| summed kernel-active time | 76.420 ms | 77.220 ms |
| ordinary kernel-launch calls | 38,745 | 38,652 |
| device-to-device copies | 2,952 | 2,952 |

The pure reference's registered delta-rule operator intentionally contains an
explicit token recurrence. Each token performs many ordinary tensor
operations. A 64-token forward component launches
`24 + 64 * 11 = 728` kernels, while its 96-token counterpart launches 1,080.
The explicit backward recurrence launches approximately 28 kernels per token,
producing 1,864 or 2,761 launches for the two microbatch geometries. The outer
compiled graph treats the registered operation as opaque and therefore cannot
fuse across its Python implementation. Optimized `mlops` kernels avoid this
reference-model storm, but core runtime correctness cannot depend on them.

Most 12--19-ms backward dispatch intervals are consequently not compilation
or pure overhead: their host thread and GPU stream through thousands of small
operations together, and the corresponding task CUDA interval is also about
12--19 ms. This makes synchronous allocator and boundary costs
performance-critical, because there is little task-level run-ahead available
to hide them.

## Why identical numerical kernels take longer under ShadowSpill

The warm NSYS controls expose three independent costs.

### Transfer-induced CUDA launch backpressure

Standard PyTorch begins consuming kernels immediately. ShadowSpill first puts
the compute stream behind startup H2D dependencies while the Python thread
continues issuing the recurrent operation's hundreds of launches. The pending
CUDA queue eventually fills and an otherwise asynchronous launch call blocks
the dispatch thread until the device consumes prior work. Terminal D2H and the
fragmented optimizer create the same condition late in the step.

| Launch API observation | Standard | ShadowSpill |
|---|---:|---:|
| ordinary launch API total | 74.419 ms | 102.626 ms |
| largest ordinary launch | 0.076 ms | 25.245 ms |
| `cuLaunchKernelEx` total | 0.391 ms | 53.069 ms |
| largest `cuLaunchKernelEx` | 0.011 ms | 52.004 ms |

The exact outlier durations are profiler-perturbed, but their causal placement
is unambiguous: the early outlier overlaps startup H2D and the late outlier
overlaps terminal D2H. No corresponding launch block appears in the standard
control.

### One global runtime lock

All 37,563 allocator callbacks occur synchronously on the PyTorch dispatch
thread. Of these, 120 are zero-byte requests and 37,443 lease real ranges;
all 37,443 materialized allocations receive matching logical frees.

| Allocator callback statistic | Time |
|---|---:|
| median | 0.430 us |
| p90 | 1.215 us |
| p95 | 1.477 us |
| p99 | 6.622 us |
| mean | 1.429 us |
| maximum | 206.782 us |
| aggregate | 53.693 ms |

The aggregate is nested inside compiled dispatch and must not be added to it.
NSYS separately observes approximately 54 ms of slow `pthread_mutex_lock`
acquisitions on the dispatch thread. At the same time the progress thread
issues 223,208 `cuEventQuery` calls, consuming 70.643 ms of driver API time,
while walking transfers and retirements under that same global mutex. This is
the concrete concurrency reason that a cheap median allocator callback still
becomes a material critical-path cost.

The task visible in the accompanying NSYS investigation is a representative
example:

| Field | Value |
|---|---:|
| execution identity | `execution_000004` |
| semantic identity | `microbatch_0000.stage_0004.forward.recompute` |
| canonical IR identity | `task_000009` |
| host task range under NSYS | 16.733 ms |
| allocator callbacks | 794 |
| allocator NVTX aggregate | 4.850 ms |
| reportable mutex waits | 64 |
| mutex-wait aggregate | 7.067 ms |
| mean / maximum mutex wait | 110.4 / 139.4 us |
| kernel launches | 728 |
| host time in launch APIs | 1.603 ms |
| worker event queries overlapping this task | 17,557 |
| worker event-query API time | 5.715 ms |

The dispatch sequence is therefore repeatedly interrupted inside one
numerical task:

```text
launch recurrence kernels
    -> request next temporary allocation
    -> block on global runtime mutex
    -> worker scans and queries transfer/task events
    -> allocator obtains mutex and leases a range
    -> launch more recurrence kernels
    -> repeat
```

NSYS magnifies this small-kernel workload: the isolated profile predicts
4.781 ms, non-callback CUDA-event tracing observes 6.997 ms, and NSYS projects
15.107 ms. The absolute NSYS interval is therefore not a production timing,
but the 64 waits and 17,557 overlapping queries identify a real runtime defect.

#### Why completion is queried, and why the current query algorithm is wrong

The runtime must eventually learn that asynchronous work completed. A
completion has semantic consequences that a stream wait alone cannot publish:

- an H2D completion makes an object device-ready and allows its event lease to
  be recycled;
- a D2H completion makes the copied host version authoritative, releases the
  old device lease, and wakes capacity waiters;
- a task-fence completion permits an annotated release or offload to advance;
- a `record_stream` retirement completion makes a logically freed anonymous
  slab range safe to reuse.

These completions are already represented by CUDA events. `cuEventQuery` is
the driver's nonblocking test for such an event; it does not wait for the
device or synchronize a stream. The problem is neither the existence of
events nor one occasional query. The current worker repeatedly walks the
entire active-allocation list and the entire action list, queries events from
inside that traversal, and retains the global runtime mutex across the driver
calls. A single fence can also be encountered through many dependent actions.
This produced 223,208 queries in one step and made the unrelated synchronous
allocator wait for the scan.

Replacing every query with `cuEventSynchronize` would be worse: it would turn
a nonblocking progress check into a blocking wait, and with one worker it
could prevent dispatch/progress on the opposite transfer lane. Stream host
functions are also unsuitable: they serialize later commands on that stream,
cannot safely call CUDA APIs, and already produced a measured 50.961-ms trace
observer stall.

The corrected design still uses CUDA events, but organizes completion by the
stream that recorded each event:

1. Maintain a FIFO completion queue per H2D, D2H, and participating compute
   stream.
2. Snapshot and retain only the head record under that queue's lock.
3. Call `cuEventQuery` after releasing all ShadowSpill state locks.
4. If the head is incomplete, do not inspect successors: CUDA stream order
   proves they cannot be complete.
5. If it is complete, commit its transition and drain immediately completed
   successors. Query a shared task fence once even if several actions retain
   it.
6. Use condition-variable wakeups plus adaptive polling/backoff; temporarily
   increase polling frequency only when allocator pressure makes completion
   latency critical.

Thus queries remain asynchronous and cheap, while query count scales with
stream-frontier progress rather than `objects x polling passes`. Most
importantly, no driver call occurs while the slab allocator, object table, or
object record is locked.

### Optimizer task fragmentation and repeated global binding scans

The recurrent optimizer graph is dependency-partitioned into 97 components so
PressureFit can independently schedule each parameter, gradient, and state.
That planning ability is valid, but the executor performs an unnecessary
global operation at every component: `_current_optimizer_bindings()` walks the
complete model parameter and optimizer-state inventory, constructs a complete
name-to-tensor dictionary, and only then selects the few bindings needed by
that component.

| Traced host category | Forward | Backward | Optimizer | Total |
|---|---:|---:|---:|---:|
| native `before_task` | 1.507 ms | 0.798 ms | 3.787 ms | 6.093 ms |
| lookup, rebind, argument assembly | 1.168 ms | 3.753 ms | 56.765 ms | 61.686 ms |
| compiled dispatch | 133.323 ms | 198.718 ms | 13.122 ms | 345.163 ms |
| output/gradient postprocessing | 13.383 ms | 4.489 ms | 3.064 ms | 20.935 ms |
| native `after_task` | 0.625 ms | 1.421 ms | 2.768 ms | 4.814 ms |

The C++ storage-rebind ranges themselves total only 8.086 ms across 4,328
calls. Most of the 56.765-ms optimizer bucket is therefore repeated Python
inventory construction, not pointer replacement. Captured optimizer tensor
identities are stable between state transitions; the executor should maintain
one direct binding table and refresh it only after lazy-state creation or
checkpoint restore.

## Diagnostic observer effect

The original seven-timestamp implementation used three `cuLaunchHostFunc`
callbacks per task. In one trace, a callback submission itself blocked for
50.961 ms. The traced wall was 593.992 ms versus a 476.734-ms untraced median.
The callbacks are useful for the causal discovery above but fail the intended
less-than-one-percent tracing-overhead contract.

The replacement retains the four native host timestamps and uses three
preallocated CUDA events for `before_readiness_waits`,
`before_task_compute`, and `after_task_compute`. Event-relative stream times
can be resolved only when `DiagnosticsHandle.result()` explicitly
synchronizes. This preserves all seven logical timestamps without executing
host code on the compute stream. That replacement is now implemented; the
historical callback trace is retained only as root-cause evidence.

## Work outside the current measured task boundaries

The current names describe only calls into the neutral runtime. They do not
describe the complete logical task boundary. The following per-task work is
currently outside the corresponding host timing buckets:

- before native `before_task`: task/function lookup, current-stream lookup,
  object-to-alias translation, and alias deduplication;
- between native `before_task` and compiled dispatch: storage rebinding and
  graph-argument assembly;
- between compiled dispatch and native `after_task`: output flattening,
  output promotion/binding, gradient accumulation, raw-reference release, and
  storage dematerialization;
- after native `after_task`: released-object dictionary cleanup, terminal
  gradient cleanup, optimizer-state flags, and timing/NVTX finalization.

The logical diagnostic definitions will therefore be:

```text
before_task = static input selection + native acquire/waits
            + storage rebind + argument assembly + compute-start marker

task_compute = compiled numerical callable and nested allocator callbacks

after_task = compute-end marker + output/gradient processing
           + dematerialization + native action submission + object cleanup
```

Step-level work remains separate rather than being hidden in a task: public
input guards, prior-step cooldown, CPU/input staging, initial placement,
caller-output ownership transfer, objective/metric reconstruction, and result
construction. Background transfer dispatch, completion, retirement, and
allocator-pressure wakeups remain asynchronous component ledgers associated
with their trigger execution task.

## Required corrections

The evidence supports the following order without changing PressureFit
directives, recomputation, or task order:

1. Replace callback timestamps with preallocated CUDA events.
2. Make diagnostics measure complete logical `before_task` and `after_task`
   work while retaining their detailed subcomponents.
3. Precompute static task alias/action ABI records and cache optimizer tensor
   bindings across components.
4. Replace the global mutex with locks owned by semantically independent
   tables and queues. Define and test a strict lock order and object lifetime
   protocol.
5. Stop polling every transfer/fence under an allocator-visible lock. Exploit
   FIFO stream ordering and event pooling while retaining exact causal action
   semantics.

No PressureFit directive, recomputation choice, task order, or in-step
prefetch location is changed by these corrections.

## Proposed runtime data model and concurrency

The runtime should be organized around central ownership structures. Locks
belong to those structures rather than to an undifferentiated runtime object.

### `ShadowSpillObjectTable`

The object table owns the hash index from stable object ID to object record and
is responsible only for membership and lifetime.

```c
typedef struct ShadowSpillObjectTable {
    pthread_rwlock_t membership_lock;
    ShadowSpillObject **buckets;
    uint64_t bucket_count;
} ShadowSpillObjectTable;
```

Lookup takes the read lock only long enough to find the object and increment
its reference count. Registration/removal takes the write lock. A queued
action, admitted task, or diagnostic snapshot holds an object reference, so
removing an ID cannot free a record still used by the worker.

### `ShadowSpillObject`

One object represents an alias bundle, not an individual tensor view. Its lock
protects only that object's residency and version state.

```c
typedef struct ShadowSpillObject {
    uint64_t object_id;
    uint64_t size_bytes;
    _Atomic uint32_t references;
    _Atomic uint8_t detached;
    pthread_mutex_t lock;

    ShadowSpillResidency residency;
    uint64_t generation;
    uint64_t authoritative_version;
    uint64_t device_version;
    uint64_t host_version;

    ShadowSpillAllocationLease *device_lease;
    ShadowSpillHostLease *host_lease;
    ShadowSpillEventLease *readiness_event;
    uint64_t readiness_generation;
    ShadowSpillTransfer *inflight_transfer;
    uint64_t latest_writer_task;

    void *retired_device_pointer;
    uint64_t retired_generation;
    uint8_t retain_spill_copy;
} ShadowSpillObject;
```

The device pointer and slab offset live in the referenced allocation lease;
they are not duplicated as independently mutable object fields. A readiness
event is tagged with the generation it makes ready, preventing an old
completion from publishing state for an imported object.

`residency` is the explicit state machine (`host_only`, `prefetching`,
`device_ready`, `offloading`, `release_pending`, or `released`). The transfer
pointer is retained only while one state transition owns the object. Immutable
tensor view geometry is frontend metadata; the neutral object represents the
shared alias bundle/storage and therefore does not contain PyTorch tensor
objects.

The normal lookup protocol is `table read lock -> increment object reference
-> release table lock -> object lock`. Removal detaches the hash entry and
drops the table's reference; destruction occurs only when the final queued
action/task reference is released.

### `ShadowSpillMemoryPool`

Device and pinned-host memory are two configured instances of one generic
bounded-pool abstraction. A memory pool owns its arena base, range geometry,
alignment policy, physical accounting, and the condition on which a genuinely
capacity-blocked client waits. It does not own framework allocation records,
object residency, transfers, or stream retirement.

```c
typedef struct ShadowSpillMemoryPool {
    pthread_mutex_t lock;
    pthread_cond_t capacity_changed;
    ShadowSpillRangeAllocator ranges;
    void *base;
    uint64_t minimum_alignment;
    ShadowSpillMemoryKind kind;
} ShadowSpillMemoryPool;
```

The runtime instantiates exactly `device_pool` and `host_pool`. Both use the
same reserve/release/coalescing interface; only their arena provider,
minimum alignment, and placement policy differ. `malloc` and logical `free`
use the device pool through the allocation-record owner. Offload/prefetch use
both pools without changing the range algorithm. No CUDA query, event creation,
copy dispatch, or stream wait is performed while either pool lock is held.

Allocation-ID and pointer hashes, PyTorch logical allocation leases, stream-use
retirement, and requested-byte accounting remain a separate allocation-record
owner because those concepts apply to framework-visible device allocations,
not to every pinned-host subrange.

### `ShadowSpillExecutionTable`

The admitted execution plan is immutable and needs no hot-path lock. Each task
record stores direct retained object pointers, predecoded mutations/actions,
the contiguous execution index, and its semantic trace name.

```c
typedef struct ShadowSpillTask {
    uint64_t execution_ordinal;
    uint64_t canonical_task_id;
    const char *semantic_name;
    ShadowSpillObject **inputs;
    uint32_t input_count;
    ShadowSpillMutationTemplate *mutations;
    uint32_t mutation_count;
    ShadowSpillActionTemplate *actions;
    uint32_t action_count;
} ShadowSpillTask;
```

This removes repeated index parsing, object hashing, alias deduplication,
and ctypes array construction from each invocation. NSYS ranges use
`execution_XXXXXX` plus the semantic name; the canonical task ID remains
secondary correlation metadata.

### `ShadowSpillTransferLane`

Each H2D or D2H stream has one queue lock and two FIFOs: pending dispatch and
in-flight completion. Actions retain their object, destination lease, trigger
fence, and completion event.

```c
typedef struct ShadowSpillTransferLane {
    pthread_mutex_t lock;
    ShadowSpillBackendStream stream;
    ShadowSpillTransfer *pending_head;
    ShadowSpillTransfer *pending_tail;
    ShadowSpillTransfer *inflight_head;
    ShadowSpillTransfer *inflight_tail;
} ShadowSpillTransferLane;
```

The worker removes queue work under the lane lock, releases it, submits CUDA
operations, then reacquires only the affected lane/object lock to publish the
result. It never holds an allocator lock across a driver call.

### `ShadowSpillCompletionTracker` and event pool

Completion queues are FIFO per CUDA stream. The worker queries only each
stream's head event; an incomplete head proves every successor on that stream
is also incomplete. A completed head is removed and immediately completed
successors are drained. Shared task fences are tested once regardless of how
many actions or retirements reference them.

General progress uses nonblocking `cuEventQuery` outside all state locks, with
fixed, short FIFO-head polling. `before_task` does not poll a prefetch: it inserts
`cuStreamWaitEvent`. D2H dispatch does not host-wait for compute: the D2H stream
waits on the task fence. A targeted blocking `cuEventSynchronize` is permitted
only on a dedicated background waiter or allocator-pressure slow path; it
never executes on the PyTorch thread.

The fixed event pool owns creation and reuse behind its own short lock. No
steady-state event creation/destruction is required. Stream host callbacks are
not used for progress because they serialize subsequent stream work and may
not invoke CUDA.

### Central runtime ownership

The top-level runtime is deliberately a collection of central owners, not a
central lock:

```c
typedef struct ShadowSpillRuntime {
    ShadowSpillObjectTable objects;
    ShadowSpillMemoryPool device_pool;
    ShadowSpillMemoryPool host_pool;
    ShadowSpillAllocationTable allocations;
    ShadowSpillExecutionTable execution;
    ShadowSpillTransferLane h2d;
    ShadowSpillTransferLane d2h;
    ShadowSpillCompletionTracker completions;
    ShadowSpillEventPool events;
    ShadowSpillTraceBuffer trace;
    ShadowSpillFailureState failure;
    ShadowSpillLifecycle lifecycle;
} ShadowSpillRuntime;
```

This makes ownership auditable: object residency belongs to an object;
physical ranges and capacity belong to the two memory pools; queue order
belongs to a transfer lane; completion order belongs to the tracker; immutable
task topology belongs to the execution table. A mutex is attached to the
owner whose invariant it protects rather than to the runtime as a whole.

| Owner | Synchronization | Protects | Must never contain |
|---|---|---|---|
| object table | read/write membership lock | ID-to-object membership and table-owned references | CUDA calls, residency transitions, slab allocation |
| individual object | object mutex | residency, generations, versions, current leases and transfer | hash scans, unrelated objects, CUDA calls |
| execution or spill memory pool | one mutex + capacity condition per instance | bounded arena, free ranges, placement, physical byte accounting | object-table scans, event queries, copies, stream retirement |
| allocation table | mutex | framework allocation hashes, leases, logical free, stream retirement, requested bytes | host subranges, object-table scans, copies |
| H2D or D2H lane | one mutex per lane | pending/in-flight FIFO links and lane-local counters | allocator waits, object-table scans, CUDA completion waits |
| completion stream | short queue lock | ordered event frontier and retained completion records | `cuEventQuery` itself |
| event pool | short mutex | free/in-use event leases | event synchronization or action processing |
| trace buffer | atomic reservation index | append slots and overflow flag | any runtime-state lock |
| failure/lifecycle | atomic first status plus cold-path mutex | first-cause payload, open/closing/closed transition | ordinary allocation or task dispatch |

The main hot paths then have narrow behavior:

```text
malloc:
    allocation-table lock -> device-pool reserve/reuse -> unlock

before_task:
    immutable task record -> each object lock -> snapshot lease/event -> unlock
    -> cuStreamWaitEvent for unfinished snapshots -> batch storage rebind

progress:
    completion-frontier lock -> retain head -> unlock
    -> cuEventQuery -> object transition -> optional memory-pool release

free:
    allocation-table lock -> mark logical free / enqueue retirement -> unlock
    -> signal progress worker
```

The immutable execution table is important here: once a plan is admitted,
`before_task` receives a direct task record whose input array already contains
retained `ShadowSpillObject *` values. The hash table remains the authoritative
public ID lookup and lifecycle owner, but it is not on the repeated task path.

### Trace and failure state

The default-off trace uses preallocated arrays with an atomic slot counter, so
recording does not take an object, allocation, or transfer lock. The first
runtime failure uses an atomic compare-and-swap to latch status and a small
lock for its diagnostic payload; it then wakes capacity and worker conditions.
Lifecycle/bootstrap/close uses a separate cold-path lock.

### Hot-path locking rule

Steady-state code should normally hold at most one mutex. Cross-structure
transitions use snapshot-and-commit:

1. retain the relevant record and snapshot its generation under its owner;
2. release that lock;
3. perform range or CUDA work under no unrelated lock;
4. reacquire the record owner and commit only if identity/generation still
   match;
5. release retained references.

This is simpler and safer than relying on a large global lock order. The few
unavoidable nested cold-path operations have an asserted order documented in
the private header and exercised under ThreadSanitizer.

## Post-`ad2f4ef` NSYS accounting

The post-fix trace is
`qualification/results/phase1/qwen35_idle_wakeup_fixed.nsys-rep`. It preserves
the frozen 129-task/1,415-action plan. NSYS perturbs the selected-task span to
460.325 ms, so the unprofiled StepDiagnostics measurements
(304.287--304.859 ms) remain the performance authority. The trace is still the
right evidence for call nesting, lock ownership, and the causes of individual
outliers.

### Current allocator outliers

The allocator distribution is normally small even under NSYS:

| Metric | Value |
|---|---:|
| nonzero allocation callbacks | 37,563 |
| median | 0.349 us |
| p95 | 1.892 us |
| p99 | 2.805 us |
| aggregate allocator range time | 23.438 ms |
| callbacks above 50 us | 7 |
| callbacks above 100 us | 4 |

Every callback above 50 us has one of two concrete causes: five are dominated
by `device_pool.lock` contention and two are explicit NSYS profiler overhead.
There is no unexplained intrinsic allocator callback above 50 us.

The screenshot at 112.485 ms is a real lock-contention example:

| Event | NSYS interval |
|---|---:|
| progress worker's timed wait returns | 112.464495 ms |
| worker `cuEventQuery` | 112.478378--112.483440 ms |
| dispatcher allocator callback | 112.484972--112.508680 ms |
| nested dispatcher `pthread_mutex_lock` | 112.485306--112.507733 ms |

The worker unconditionally entered `progress_retirements_locked()` after the
query and walked approximately 766 active allocation records while holding
`device_pool.lock`. No record completed in that pass. The allocator did only
1.281 us of work after obtaining the lock; 22.427 of its 23.708 us was waiting
for an unrelated full-population scan.

The two largest real examples show the same ownership defect when work is
ready:

| Allocator callback | Mutex wait | Worker work under the pool lock |
|---:|---:|---|
| 178.482 us | 173.663 us | release/coalesce 197 records totaling 1,590,282,752 bytes |
| 116.210 us | 114.042 us | release/coalesce 30 records totaling 872,021,504 bytes |

The two other callbacks above 100 us are observer effects: one 125.806-us
range contains 124.493 us explicitly attributed to `PROFILER_OVERHEAD`, and
one 112.843-us range contains 112.412 us of profiler overhead.

FIFO completion tracking has already reduced worker queries from 223,208 to
1,317 in this trace. The remaining problem is downstream: after polling the
completion frontier, the worker still scans the complete active-allocation
linked list under the same mutex needed by synchronous PyTorch `malloc/free`.
The required correction is a retirement/completion queue of stable allocation
references. A completed frontier should directly identify records to retire;
the worker must not rediscover them by scanning every live allocation.

### Direct retirement queue and foreground priority

The correction creates a retirement record as soon as an allocation receives
its stream events or shared task fence. The record contains the allocation
generation and retained completion requirements. The worker detaches the queue,
tests already-published completion state without the pool lock, and enters the
pool only to validate the generation and release/coalesce the range. It never
walks `active_allocations` to rediscover retirement work.

The first direct-queue trace exposed a second ownership issue. POSIX mutexes do
not promise handoff fairness: while releasing a completed batch one record at a
time, the worker could unlock and immediately reacquire before an already
waiting PyTorch allocator callback ran. One callback still spent 104.222 us of
its 109.804 us range in the mutex while the worker processed roughly 30 ready
records.

Priority is now a generic `MemoryPool` property:

```text
foreground lock:
    trylock succeeds -> enter with no shared waiter update
    trylock fails    -> publish waiter -> blocking lock -> clear waiter

background lock:
    foreground waiter exists -> defer
    trylock fails             -> defer
    lock acquired but waiter arrived -> unlock and defer
    otherwise                 -> perform one range mutation
```

After each background mutation, the next acquisition repeats this policy. Thus
a foreground callback waits behind at most a mutation already in progress; the
worker cannot reacquire across an entire ready batch ahead of it. With no
foreground waiter, terminal retirement remains a fast batch. Completion
discovery, queue traversal, event operations, and CUDA calls never hold the
pool lock.

A physically separate workspace pool was rejected. An allocation created
inside a compiled task may later be promoted to a declared output, so its final
class is not always known at `malloc`. A fixed physical partition would also
strand capacity between workspace and Program objects under tight budgets. One
generic device `MemoryPool` retains exact sharing and the same physical cap.

The commit-specific resulting trace is
`qualification/results/phase1/qwen35_memory_pool_priority_18584ef.nsys-rep`:

| Allocator metric | Before queue | Direct queue | Pool priority |
|---|---:|---:|---:|
| Callback count | 37,563 | 37,563 | 37,563 |
| Median | 0.349 us | 0.453 us | 0.499 us |
| p99 | 2.805 us | 2.701 us | 2.991 us |
| Genuine callbacks above 50 us | 5 | 1 | 0 |
| Nested mutex wait | 0.895 ms | 0.338 ms | 0.286 ms |
| Aggregate allocator NVTX time | 23.438 ms | 27.105 ms | 28.335 ms |

The one pool-priority callback apparently above 50 us is 119.605 us and
contains 115.499 us of NSYS `Chunk Allocation` overhead on the same thread.
After subtracting that observer work, none exceeds 50 us. The increased
aggregate and median are the trace-visible cost of a `pthread_mutex_trylock` on
the foreground fast path. This is bounded rather than a tail stall.

NSYS StepDiagnostics selected span is 471.637 ms versus 460.325 ms before the
queue. Kernel union changes by +1.614 ms. Summed task-event intervals add 9.002
ms and between-task spacing adds 2.310 ms; the allocator-range aggregate itself
adds 4.896 ms under instrumentation and explains much of the within-task launch
spacing. The largest interval changes are recurrent backward tasks whose
kernels are generally unchanged within tens of microseconds while their
instrumented host calls add 0.4--1.6 ms. Production-like controls are
materially smaller: the first accepted run measured 307.783--309.521 ms
selected spans and 287.913--290.724 ms task sums; an abstraction-equivalent
repetition measured 311.384--314.125 ms and 292.806--293.228 ms. Both retain
exact plan identity and are recorded to expose run-to-run spread.

### Current inter-task gap

The Python executor now has real `_before_task()`, `_run_compiled_task()`, and
`_after_task()` orchestration functions. The refactor is nevertheless
incomplete in two important ways:

1. The NVTX ranges named `before_task` and `after_task` currently enclose only
   `RuntimeBridge.before_task()` and `RuntimeBridge.after_task()`, not their
   complete frontend orchestration functions.
2. Input and output storage operations remain individual Python calls to the
   C++ custom operator. Batch rebinding/dematerialization has not landed.

For the screenshot's transition from
`execution_000014` (`microbatch_0000.stage_0001.backward.recompute`) to
`execution_000015` (`microbatch_0000.stage_0000.backward.recompute`), the
859.586-us compiled-call gap is exactly accounted for:

| Host interval | Time |
|---|---:|
| current output/gradient processing and dematerialization | 326.335 us |
| current narrow native `after_task` range | 33.252 us |
| Python/NVTX transition before next boundary | 76.930 us |
| next narrow native `before_task` range | 73.726 us |
| transition into rebinding | 2.137 us |
| next storage rebind and argument assembly | 333.443 us |
| transition into compiled call | 13.763 us |
| **total** | **859.586 us** |

That next task performs 73 individual storage-rebind custom-op calls; the
individual C++ ranges total 153.927 us. The remainder is Python iteration,
dictionary access, validation, and graph-argument construction. The preceding
postprocessing walks and publishes backward outputs, promotes first gradient
allocations, updates frontend object maps, and dematerializes selected aliases.

Across all 128 task transitions in the NSYS capture:

| Component | Aggregate |
|---|---:|
| output/gradient postprocessing before narrow `after_task` | 16.649 ms |
| narrow native `after_task` | 1.962 ms |
| between-boundary Python work | 3.396 ms |
| narrow native `before_task` | 3.501 ms |
| pre-rebind transition | 0.198 ms |
| rebind and argument assembly | 9.607 ms |
| compiled-call transition | 0.967 ms |
| **compiled-call gap total** | **36.279 ms** |

NSYS expands these small Python/custom-op intervals. In the unprofiled trace,
the directly measured component totals are 12.678 ms of postprocessing,
1.751 ms of native after-task work, 0.808 ms of cleanup, 2.293 ms of native
before-task work, and 8.266 ms of rebind/argument assembly. Some host work runs
ahead of already queued GPU work, so only about 15.5 ms appears between the
summed task-event intervals and the 304.5-ms selected-task span.

A correlation check found all 40,528 dispatcher-launched kernels inside one of
the selected compiled-call ranges and zero kernels launched from the
inter-task windows. Thus this recurrent plan does not hide gradient
accumulation or other numerical CUDA work in `_after_task`; the gap components
above are host-side state, binding, validation, and dispatch work.

The next frontend milestone must make the outer `before_task`/`after_task`
ranges cover their complete functions while retaining nested component ranges,
then batch the storage operations and use predecoded argument slots. This is
separate from changing worker polling or condition-variable policy.

### Why the worker appears to call only `cuEventQuery`

It does not. The C worker's complete API ledger is:

| Driver API | Calls |
|---|---:|
| `cuEventQuery` | 1,317 |
| `cuStreamWaitEvent` | 782 |
| `cuEventRecord` | 782 |
| `cuMemcpyHtoDAsync_v2` | 394 |
| `cuMemcpyDtoHAsync_v2` | 388 |

The backend deliberately uses the CUDA driver API, hence `cuMemcpy...` rather
than runtime-API `cudaMemcpyAsync` names. All 394 startup H2D calls were
submitted between 29.440 and 31.222 ms. All 388 terminal D2H calls were
submitted between 524.786 and 542.685 ms. At the screenshots' 112-ms and
215-ms positions, those H2D operations were already enqueued and the D2Hs had
not yet been submitted, so completion queries are the only expected worker
CUDA calls visible in those local windows. GPU copies execute asynchronously
after their host enqueue call and remain visible on the H2D/D2H stream rows.

### Why the dispatcher calls `cudaStreamIsCapturing`

The trace contains exactly 390 `cudaStreamIsCapturing` calls followed one for
one by 390 `cudaEventRecordWithFlags` calls. They come from the optional
`runtime_trace=True` StepDiagnostics events: three stream timestamps for each of 129
tasks, plus the three step-level origin/start/end records. PyTorch checks stream
capture state before recording its event so that event semantics remain valid
inside CUDA graph capture. This is intended debug-only behavior, costs
0.110 ms in aggregate for the capture checks (0.953 ms for their paired event
records), and is absent from ordinary `runtime_trace=False` execution. Neutral-runtime
readiness, transfer, fence, and retirement events are separate precreated
timing-disabled driver events.

## Issue ledger

### Resolved and measured

- Removed the one-transfer-in-flight software window that host-blocked
  `before_task`; readiness is represented with stream-event waits.
- Specialized only terminal scalar unit cotangents, removing the misplaced
  four-byte startup transfer without changing arbitrary objective semantics.
- Added lifecycle-safe hash indices for objects and allocations.
- Prevented the progress thread from immediately retaining the global lock
  across consecutive completion passes.
- Replaced three per-task `cuLaunchHostFunc` timestamps with preallocated CUDA
  events. The stream trace no longer executes host code on the compute stream.
- Cached the complete optimizer binding inventory once per step rather than
  rebuilding it for every one of 97 components. On the unchanged Qwen plan:

  | Measurement | Before | After |
  |---|---:|---:|
  | first-to-last task compute span | 384.361 ms | 325.128 ms |
  | summed task CUDA intervals | 299.281 ms | 297.512 ms |
  | inter-task idle | 85.079 ms | 27.616 ms |
  | optimizer lookup/rebind bucket | 56.765 ms | 2.344 ms |
  | optimizer host task total | 80.581 ms | 21.192 ms |

  Program digest, schedule digest, 129 tasks, 1,415 actions, predicted peak,
  and predicted makespan are byte-for-byte unchanged. The steady non-cyclic
  wall remains near 480 ms because the faster optimizer exposes more terminal
  D2H tail; cross-step cyclic optimization is intentionally outside the
  current agenda.
- Added semantic execution labels and immutable admitted input/action records;
  the repeated admitted boundary no longer parses task IDs or hashes input
  object IDs.
- Added FIFO CUDA completion frontiers and event leases, reducing the Qwen
  worker from 223,208 event queries to 1,317. That earlier milestone did not
  yet remove the allocation-retirement population scan described above.
- Replaced that population scan with direct generation-tagged retirement
  records and centralized foreground/background acquisition policy in the
  generic `MemoryPool`. Updated NSYS contains no genuine allocator callback
  above 50 us.
- Split device/host memory pools and H2D/D2H transfer-lane ownership and named
  the pure-C worker `shadowspill.wkr`.
- Fixed the idle-barrier lost-notification race in `ad2f4ef` without changing
  worker progress cadence. The repeated unprofiled selected span is
  304.287--304.859 ms with exact plan identity.

### Current corrections

- Finish replacing residual global/cross-owner locking with the central
  ownership structures above.
- Make full frontend `_before_task()` and `_after_task()` ranges enclose all
  their work while retaining nested native/rebind/postprocess diagnostics.
- Batch storage binding/dematerialization calls while retaining one object
  state transition per alias bundle.
- Finish predecoded argument/output processing so repeated task paths do not
  rebuild Python dictionaries or perform avoidable validation.
- Re-run the identical Qwen control after each isolated change; acceptance is
  unchanged arithmetic/schedule plus a first-to-last span approaching the
  approximately 297.5-ms task sum.
