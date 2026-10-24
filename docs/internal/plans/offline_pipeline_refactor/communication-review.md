# Communication and resource inputs: review proposal

Status: third of the six reviews. The user selected flat, explicit-unit
rate/latency fields alongside topology and measured timings. Topology inside
CommunicationInfo, optional public calibration, measured-first precedence, and
per-rank memory budgets were already selected. This is a design, not an
implementation. Detailed measurement/link schemas and cost-model conventions
below still need to be finalized; selecting flat fields does not settle them.

## Common constructor

Use flat numeric fields on CommunicationInfo for common effective rates and
startup latencies, alongside topology and optional measured timing records.
The user selected this organization over nested per-transfer records. Keep
numeric units explicit in field names. For a single-node eight-rank example:

```python
communication_info = CommunicationInfo(
    topology=Topology(nodes={"node0": tuple(range(8))}),
    fetch_bytes_per_second=25e9,
    evict_bytes_per_second=25e9,
    scale_up_bytes_per_second=300e9,
    fetch_latency_seconds=5e-6,
    evict_latency_seconds=5e-6,
    scale_up_latency_seconds=2e-6,
    measurements=[],
)
```

These numbers illustrate units and syntax; they are not Della measurements or
recommended hardware constants. The scale-out counterparts are
`scale_out_bytes_per_second` and `scale_out_latency_seconds`. They are only
required when the requested transfers need between-node estimation. Compatible
measurements can cover operations without supplied rates. Omitted required costs
remain missing; this proposal does not silently substitute zero latency.

| Fields | Proposed meaning |
| --- | --- |
| `fetch_*` | Host/spill-to-GPU copy bandwidth and startup latency for one local transfer leg. |
| `evict_*` | GPU-to-host/spill copy bandwidth and startup latency for one local transfer leg. |
| `scale_up_*` | Effective per-participant, per-direction bandwidth and per-step startup latency for within-node communication in the selected cost model. Not the sum of all GPU links in a node. |
| `scale_out_*` | The corresponding between-node quantities. A shared NIC/host bottleneck must also be described when individual participants cannot each sustain this rate concurrently. |
| `measurements` | Optional observations for compatible copy/collective operations; durations, payload geometry/dtype, participants/placement, backend/setup, and provenance. Exact row type/fields remain to finalize with calibration. |

Rates are payload bandwidths in bytes/second. A full-duplex transfer must not be
charged against an aggregate send-plus-receive rate as if that were a one-way
rate. Host copies use their actual transferred bytes, including any padding
required by the implementation. Inter-GPU cost models derive traffic from the
operation and its full input/output geometry, not just the CPU shard size.

Topology's common node map supplies logical placement. Named collective groups
still come from Parallelism/Program. Advanced explicit link/shared-resource
details can override homogeneous assumptions for NUMA, PCIe, NVLink, or shared
network resources. Exact advanced fields and supported contention equations need
implementation validation; an isolated rate is not itself proof of aggregate
node throughput. Reports must state the homogeneous/default sharing assumptions
used when those details are absent.

## How a sharded fetch is charged

A fetch consists of its actual host-copy leg and any required weight-gather leg.
For example, with an eight-way shard of a 1 GiB parameter, each rank copies its
128 MiB local shard and participates in an all-gather producing the 1 GiB tensor.
A simple ring estimate moves 896 MiB per rank in each direction over seven steps;
that is an illustrative algorithm's traffic, not a claim that a backend uses a
ring. The estimated latency count comes from the chosen algorithm's phases.
The host-copy volume is not reused as the collective volume.

If a compatible measured duration covers that all-gather, use it for the
collective leg instead of adding the rate-based estimate to it. Continue charging
the separate host copy. If a measurement covers an entire composite transfer,
its scope must be explicit so component costs are not added again. Likewise,
communication already inside a profiled compute task is not charged again as a
standalone collective or retimed just by changing these link rates.

Reduce-scatter and eviction receive the same separation of reduction/collective
work and actual local device-to-host bytes. Gradient dtype, participant count,
padding, and any required replica reduction determine those operations; an
operator's name alone is not enough to match a measurement.

## Optional calibration

`measure_communication(...)` remains separate from `measure_program` and returns
the same CommunicationInfo object. Callers can save/reuse or manually construct
equivalent input; planning does not trigger calibration. Host-copy calibration
can run on a representative GPU. Collective probes require real participants and
bindings for the requested groups. Exact operation/size/dtype selection and
worker-binding arguments remain to finalize with the binding interface; this
review does not require measuring every possible group/size to use the planner.

The selected measured-first policy uses compatible observations and fills gaps
using supplied rates/topology and the chosen model. Interpolated or modeled
values must be labeled estimated, not reported as directly measured timings.
Curve fitting/interpolation defaults are not selected by choosing these fields.

## Memory input remains separate

```python
plan = plan_program(
    profiled_program,
    memory_budgets=MemoryBudgets(
        execution_bytes=60 * GiB,
        spill_bytes=96 * GiB,
    ),
    communication_info=communication_info,
)
```

The pair is per rank. Execution bytes include runtime/provider overhead as
already selected; planning subtracts accounted reservations to obtain usable
pool capacity. Provider communication buffers and collective scratch require
memory evidence independently of their bandwidth and belong in the Program /
Profiles resource information, not in a bandwidth field. Shared resources are
counted once. Estimated memory remains simulation-only.

MemoryBudgets does not allocate pools or cap all host-process memory. Node-wide
host capacity, exact overhead evidence, and advanced per-resource capacities must
remain explicit in the finished design; the simple rate constructor does not
silently settle those policies.

## Scope of this review

The selected common shape is Topology plus flat rate/latency fields and optional
measurements, with advanced topology details available where needed. Grouping
each transfer category's rate and latency into a nested record was not selected.
No new measured/estimated value wrappers are proposed.
Exact calibration/binding types and advanced links must implement the selected
meanings. This review does not add communication aggregation or a new asynchronous
collective scheduler; those remain deferred.
