# Communication measurement


## Host and external transfers

Explicit latency is charged as a transfer phase. Concurrent fetch/evict rates
default to independent full-duplex lanes; callers can provide different rates.
The communication description is saved alongside the plan, independently of
compute measurements. Symmetric distributed simulations account for local host
shards, weight gathers and explicitly lowered gradient reductions.

`measure_communication()` optionally measures pinned-host copies on one GPU,
using ordinary PyTorch allocation and CUDA events. It returns CommunicationInfo;
save it independently and pass it to later planning runs. A size sweep fits
startup latency and bandwidth, and simultaneous copies measure overlapping lane
rates. Raw medians, sample counts, variability and fit errors are retained.
Neither planning nor execution triggers this probe. Copy buffer use is bounded
by twice the largest requested size on each of host and GPU.

`spill=SSDPool(...)` or `spill=RemotePool(...)` selects an external backend for
calibration and for Executor/Trainer construction. Measurement uses bounded native
probe pools and the real transfer lanes, including staging and completion; it does
not install the execution allocator. Its memory bound also includes backend staging
and alignment slack. Build, task profiling and planning have no spill setting.
Execution sizes the default pinned-host pool from the admitted peak with 5%
headroom, capped by the planned spill budget. Explicit pool sizes and backend
calibration must satisfy the [pool sizing rules](training.md#trainer). Initializers,
controls and checkpoint export use bounded byte-range access on every backend.
Compound weight/gradient-shard transfers use the selected spill backend;
non-addressable storage uses bounded staging. Each distributed worker owns its
own spill pool. Remote pools may share a server endpoint while owning separate
server allocations.

## Distributed collective probes

`measure_communication(program=program, rank_map=..., device=...)` derives
collective probes from saved transfer geometry. With multiple workers, initialize
distributed rendezvous first and call on every worker with the same complete
worker-to-logical-rank map. Each worker passes its own Program; the workers'
Programs must agree on what the ranks exchange, as at execution. No compiled
task is executed. Host copies are sampled
concurrently across workers; the shared result uses the minimum observed rates
and maximum fitted latency, preserving every worker's observations. Each collective
group is then probed separately, using coordinated repetition counts and completion
timing. Logical member order is retained after worker remapping. Topology is
derived from physical hostname membership. Every worker receives the same result.
Calibration records NCCL version, group options, relevant environment settings
and configuration-file digests separately from compute setup. Workers must agree
on this backend setup, except for the settings that belong to one node: the
interfaces and devices it reaches the network through, its logging, and the
variables naming where something is installed or written, which are recorded
per worker. Changing the setup invalidates matching communication timings and
memory records without invalidating single-GPU compute profiles.

## Compute and transfer overlap

`measure_overlap(profiled_program, communication_info, ...)` then measures saved
compute tasks alone, transfers alone, and both together on all participants.
It returns an additive Profiles overlay for merge/apply before planning. The
distributed quickstart and homogeneous text recipes include this step in their
communication phase. Solo task profiles remain reusable; overlap evidence is
specific to the task implementation and collective configuration. Calibration
uses actual NCCL work events and bounded actual message sizes, excludes arrival
skew and post-compute tails, and never fits observed full-training-step times.
Cleanup and transfer-buffer admission precede task warmup. Timed routes share
one buffer and preserve cached task allocations. Isolated baselines are reused
within a communicator session; optional `OverlapProfileCache` entries share
supported paired evidence across candidates after all-rank agreement. Cache
identity includes task/input lineage, route/backend/rank geometry and sampling
policy. It never substitutes for candidate-specific memory evidence. Expensive
probes use one conditioned trial by default; probes whose three trials are
predicted to fit 100 ms use three. Host phase diagnostics keep GPU waiting,
event processing and artifact I/O distinguishable.
See [profile artifacts](artifacts.md#profiles-and-taskprofile) for
coverage, missing-sample reporting and the current model-owned/heterogeneous
communication exclusions.

```python
# All workers have initialized rendezvous and loaded the same Program.
communication = measure_communication(
    program=program,
    rank_map={worker: worker for worker in range(8)},
    device=device,
)
if worker_rank == 0:
    communication.save("communication.ssp")
```

## Probe specification

An explicit `collectives=[...]` replaces derived probes; `collectives=[]` selects
host copies only. A probe is a plain dictionary, for example
`dict(operation="reduce_scatter", ranks=[0, 1], tensor_bytes=64 << 20,
dtype="torch.bfloat16")`. Supported operations are all-gather, reduce-scatter and
all-reduce. Bytes describe the full padded gather output/scatter input, or the
full tensor on each all-reduce participant. Sizes must contain whole elements in
each shard. Duplicate probes are removed. Calls without Program, overrides or
rank_map measure only local host copies.

Gather probes accept `inplace=True` when the input shard occupies its ordered
slot inside the full output buffer. Program-derived weight gathers select this
form. Calibration preserves that layout, and cost lookup distinguishes it from
a gather with separate buffers. All-reduce is in-place. Native gradient evictions
use in-place reduce-scatter; optimizer tasks use a separate output so the input
accumulator remains available to an overlapping spill.

Derived requests cover lowered weight fetches, gradient completion and update
publication tasks. Model-owned communication requires its own lowering before
automatic derivation is supported. Exact observed collective timings do not infer
a general scale-up/out rate or overlapping-group contention.

## Memory evidence

CommunicationInfo also owns communication memory evidence. Each collective row
can contain a `memory` mapping with `workspace_bytes`, `memory_kind` and per-rank
allocation traces/contracts. Calibration observes the same allocating-output
callable used for execution, separately from event timing. Returned collective
outputs are accounted as Program objects, not counted again as workspace.
`fixed_device_bytes` covers retained communicator resources; its metadata records
the setup, groups, topology and per-rank native/NVML observations. This remains
separate from compute/provider overhead in Profiles.

NCCL-owned allocations use ordinary external headroom. Executor warms the saved
collective forms before admitting its arena, adds measured communicator memory
to the reservation, and checks physical usage at admission and completed calls.
An overrun of external or whole-process memory is reported and execution
continues unless `reject_overbudget=True`, which refuses it; pool bounds and
device allocation failures stay enforced either way.
These external resources are distinct from task-local PyTorch allocations, which
still require allocation contracts. Estimated task memory remains simulation-only.

The IR also supports explicit framework collective tasks with declared input,
output and completion dependencies. Their durations and workspace are resolved from
CommunicationInfo during planning, so applying compute profiles does not require
a process group or dummy collective profiles. The saved ProfiledProgram can leave
both fields unspecified for those tasks. Planning requires their missing memory
information, including an explicit zero when appropriate; unknown is never zero.
Manually supplied numeric memory values default to estimates and permit simulated
timelines only. A memory-only observation can omit `duration_seconds`, allowing
timing to come from another compatible observation or the supplied rates.
Model/provider operations that communicate internally retain their ordinary
measured task duration and memory in Profiles.

## Scheduling and execution

Generated training plans place optimizer reductions, local updates and weight
publication on the same ordered compute resource as forward/backward. Separate
collective cost records preserve single-GPU compute profiling; they do not put
communication on an independent task stream. Each reduction/update/publication
sequence is placed together, either interleaved after the corresponding backward
work or at the tail. Reports retain phase labels and ring-equivalent traffic.

Parameter gathers remain inside compound fetches. Gradient reduce-scatter and
optional cross-replica all-reduce remain inside compound evictions. These transfers
can overlap unrelated compute using the existing readiness/completion boundaries.
The generic IR can describe independent task lanes, but concurrent shared-reader
retirement is not qualified; generated DP plans deliberately retain ordered task
execution.

The runtime's existing task-boundary handshake acknowledges submission of queued
collective transfers before the next explicit task is dispatched. Within a batch,
collective evictions precede collective fetches, independent of event polling.
This guarantees host submission order without waiting for GPU completion.

Worker rendezvous, rank mapping, compatible Plan sequences and sharded checkpoint
loading are covered by the [distributed execution reference](distributed.md#fresh-execution-workers).

The GPU allowance includes fixed compute/provider and communication overhead.
Planning subtracts both before scheduling objects/workspace, and reporting adds
them back to the total peak. Background weight-fetch and gradient-eviction callbacks have no task
allocation scope, so their observed scratch is conservatively reserved per
sharded object and collective operation rather than assumed to retire at the next asynchronous submission.
Explicit communication tasks account for scratch over their task lifetimes.
Spill capacity is separate. No physical memory is allocated for these calculations.

Task resource lanes bind to stable CUDA streams in the worker. Native task and
physical-reuse dependencies insert stream waits; independent lanes can overlap.
Each invocation joins its lanes before output acquisition and result completion,
including when successive calls originate on different caller streams. Retained
BLAS workspaces are initialized before the fixed execution arena is admitted.
Profiling includes per-resource library capacity in fixed memory, and the saved
setup records the BLAS backend and workspace settings. The cuBLAS/cuBLASLt bound
is conservative when the libraries share a workspace.
