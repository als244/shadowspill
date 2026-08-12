# Qwen Runtime-Overhead Investigation

## Purpose

This note explains why one recurrent Qwen 3.5 training step took materially
longer under ShadowSpill than an equivalent whole-objective compiled PyTorch
step. It is a live Phase-1 investigation record: established causes are
separated from residual time that still requires controlled measurement.

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

The stream timestamps are implemented with debug-only `cuLaunchHostFunc`
callbacks. Consequently, a small positive interval can be callback scheduling
overhead. A residency stall is established only when the neutral runtime
ledger also contains a `readiness_wait` event for that task.

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
backing for reuse by both microbatches.

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

## Residual execution time

The transfer-window and cotangent corrections explain two independent
readiness stalls but do not yet explain the full standard-to-ShadowSpill
difference. The terminal-cotangent trace spans 439.745 ms from the first
`before_task_compute` to the final `after_task_compute`. Its noisy per-task
CUDA-event sum is 312.561 ms, leaving roughly 127.2 ms between numerical task
intervals. The trace attributes 73.885 ms of host time to storage
rebinding/validation, whose runtime object lookup is currently a linear scan.
Host categories overlap GPU execution and cannot simply be added to the gap;
the lookup implementation must be corrected and the identical diagnostic
repeated before attributing that time.

The next controls preserve identical numerical graphs and separately measure:

1. the selected staged program on the standard allocator;
2. the same program on a fully resident ShadowSpill allocator with memory
   actions disabled;
3. the complete ShadowSpill schedule without debug callbacks;
4. the complete schedule with internal tracing and an agreeing NSYS trace.

No PressureFit directive, recomputation choice, task order, or in-step
prefetch location is changed by this investigation.
