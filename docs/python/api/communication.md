# Communication measurement

Measure transfer and overlap costs separately from task profiles, then supply
both to planning. These probes use GPUs and communication participants; planning
only reads their saved results. The [phase guide](../phases.md) shows the order.

## API options

`measure_communication` returns `CommunicationInfo`, saved independently and
passed to `plan_program`. With no arguments it measures local pinned-host copies.

<!-- api-options: packages/engine/src/shadowspill/communication/measure.py:measure_communication -->

| Argument | Default | Meaning |
| --- | --- | --- |
| `program` | `None` | Derive collective probes from this Program's lowered transfers. |
| `rank_map` | `None` | Complete worker → logical-rank mapping for distributed probes. |
| `collectives` | `None` | Explicit probes replacing derived ones; `[]` means host copies only. |
| `device` | `"cuda"` | Device used by this worker. |
| `sizes_bytes` | `(1024, 1 << 20, 16 << 20, 64 << 20)` | Host-copy size sweep used to fit latency/bandwidth. |
| `options` | `None` | Default `ProfilingOptions()`, or explicit timing bounds. |
| `warmup_seconds` | `None` | Conditioning-time shortcut, mutually exclusive with `options`. |
| `measure_seconds` | `None` | Measurement-time shortcut, mutually exclusive with `options`. |
| `spill` | `None` | Pinned-host copies, or an explicit external spill-backend configuration. |
| `numa_binding` | `True` | Take a runtime's host placement before measuring: thread affinity narrowed to the device's host NUMA node and its memory preferred, for the life of the process. `False` keeps the caller's placement, to match execution that runs with `numa_binding=False`. |

Rates describe execution only when they are measured where execution runs. A
pinned buffer on another NUMA node copies at a fraction of the device-local
rate, and a distributed calibration prices every worker at its slowest
worker's rate, so one misplaced worker lowers them all. The measuring process
therefore takes its placement the way an execution worker does, from the
initialization of a runtime, here a probe runtime with minimal pools that is
released at once. The result's `host_placement` records the choice, the CPUs
and NUMA nodes measured on, and `unplaced` with the reason when the process
could not be placed; a distributed result keeps each worker's record beside its
own rates under `per_rank`.

After applying compute profiles, `measure_overlap(profiled_program,
communication_info, ...)` returns a `Profiles` overlay. Merge it with the base
profiles and apply the result before planning:

```python
from shadowspill import apply_profiles, measure_overlap

overlap = measure_overlap(profiled, communication)
profiled = apply_profiles(profiled.program, profiled.profiles.merge(overlap))
```

<!-- api-options: packages/engine/src/shadowspill/profiling/overlap.py:measure_overlap -->

| Argument | Default | Meaning |
| --- | --- | --- |
| `profiled_program` | Required | Program with isolated compute measurements applied. |
| `communication_info` | Required | Compatible collective calibration. |
| `device` | `"cuda"` | Device used by this worker. |
| `rank_map` | `None` | Complete worker → logical-rank mapping. |
| `stages` | `None` | All stages, or an explicit subset of stage indices. |
| `inputs` | `None` | Explicit task inputs by profile key; disables overlap-cache reuse. |
| `trials` | `None` | Three trials when predicted work fits 100 ms, otherwise one; a positive integer fixes the count. |
| `cache` | `None` | Shared `OverlapProfileCache` for compatible paired measurements. |

Call distributed probes on every participant after rendezvous. Overlap evidence
currently covers supported sharded collective transfers; model-owned opaque
communication is excluded and recorded as missing coverage. No relevant routes
means an empty overlay. See the [distributed workflow](distributed.md).

## Host and external transfers

A size sweep fits copy latency and bandwidth; simultaneous copies measure
fetch/evict overlap. Raw timings, sample counts, variability and fit errors stay
in `CommunicationInfo`. Local probes use ordinary PyTorch allocation and CUDA
events, bounded by twice the largest requested size on each of host and GPU.
Planning and execution never launch these calibration probes implicitly.

`spill=SSDPool(...)` or `spill=RemotePool(...)` selects an external backend.
Calibration uses bounded native probes with its actual staging/completion paths
and alignment overhead; it does not install the execution allocator. Use the same
backend for execution. Default pinned-host execution capacity is the admitted
peak plus 5%, capped by the planning budget. Explicit capacities must satisfy the
[pool sizing rules](executor.md#sharing-state-across-plans). Every distributed worker
owns its spill pool, including when workers share a remote server endpoint.

Supplied transfer rates can support simulation without measurement. Concurrent
fetch/evict rates default to independent full-duplex lanes; callers can provide
different rates. Compatible measured costs take precedence over supplied rates.

## Distributed calibration

Initialize rendezvous, then call on every worker with the same complete
worker-to-logical-rank map. Each worker supplies its own Program; their saved
transfer descriptions must agree on what ranks exchange. The probes do not
execute compiled model tasks.

```python
from shadowspill import measure_communication

communication = measure_communication(
    program=program,
    rank_map={worker: worker for worker in range(8)},
    device=device,
)
if worker_rank == 0:
    communication.save("communication.ssp")
```

Host copies are measured concurrently across workers. The shared result uses
minimum observed rates and maximum fitted latency, retaining all observations.
Each collective group is then measured separately, with coordinated repetitions.
Every worker receives the same result; logical member order follows `rank_map`,
and physical host membership determines topology.

Calibration records NCCL version, group settings and relevant environment/config
identity. Workers must agree on shared settings; node-local device/interface and
installation paths are recorded separately. Changing the backend setup
invalidates matching communication evidence without invalidating isolated
compute profiles. See [distributed execution](distributed.md).

## Explicit collective probes

`collectives=[...]` replaces Program-derived probes; `collectives=[]` selects
host copies only. For example:

```python
probe = dict(
    operation="reduce_scatter", ranks=[0, 1],
    tensor_bytes=64 << 20, dtype="torch.bfloat16",
)
```

Supported operations are all-gather, reduce-scatter and all-reduce. Bytes describe
the full gathered output/scattered input, or the tensor per all-reduce
participant. Every shard must contain whole elements. Duplicate probes are
removed. Gather `inplace=True` places the input shard within its output slot;
Program-derived weight gathers use this form. All-reduce is in-place. Native
gradient evictions use in-place reduce-scatter; optimizer tasks use a separate
output so overlapping spill can still read the accumulator.

Automatic probes cover lowered weight fetches, gradient completion and update
publication. Opaque model-owned communication is measured within its task;
standalone probes require an explicit lowering. Exact collective samples do not
infer general scale-up/out rates or contention between arbitrary groups.

## Compute and transfer overlap

Overlap calibration times the saved task alone, the transfer alone and both
together, using actual NCCL work events and bounded message sizes. It excludes
arrival skew and post-compute transfer tails, and does not fit a full training
step. Isolated baselines and buffer allocations are reused within the session;
`OverlapProfileCache` can also reuse compatible paired evidence across candidates.
Its identity includes task/input lineage, routes, backend/rank geometry and
sampling policy, and reuse requires all-rank agreement.

Expensive probes default to one conditioned trial; cheap probes use three when
their predicted work fits 100 ms. Override with `trials` for repeatability studies.
A single trial estimates contention, not statistical stability. Insufficient
memory or unsupported samples remain explicit missing evidence. See
[profile artifacts](artifacts.md#profiles-and-taskprofile) for coverage fields
and model-owned/heterogeneous communication exclusions.

## Memory evidence and planning

`CommunicationInfo` includes retained communicator memory and per-collective
allocation evidence, separately from compute/provider overhead. Returned
collective outputs are Program objects, not duplicate workspace charges.
Explicit framework-collective tasks can leave duration/memory unresolved in
compute profiles; planning obtains them from CommunicationInfo. Unknown memory
is never zero, and numeric estimates alone do not qualify execution.

The total device budget includes compute and communication overhead. Executor
warms saved collective forms before admitting its arena, then checks actual
memory use. External/whole-process overruns are reported by default;
`reject_overbudget=True` refuses them. Pool bounds remain enforced in either mode.
See [memory reservation](../../architecture/memory-runtime.md#external-memory-reservation-and-enforcement).

Generated training Plans keep optimizer reductions, updates and publication on
the ordered compute resource. Weight-fetch and gradient-eviction collectives
can overlap unrelated compute on transfer lanes; the overlap profiles model
that contention. These are different mechanisms. See
[distributed scheduling](distributed.md) and [runtime transfers](../../architecture/memory-runtime.md#transfers)
for ownership, dependencies and supported execution scope.
