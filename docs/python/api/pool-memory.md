# Pool memory evidence

Large pinned spill pools can consume device memory that the driver does not
attribute to the process. Task profiling and transfer timing do not measure this
capacity-dependent cost. `PoolMemoryProfile` records the actual pool and both
transfer lanes, separately from compute profiles and `CommunicationInfo`.

Measure once in a disposable process on an otherwise idle GPU:

```python
from shadowspill import measure_pool_memory
from shadowspill.memory import PinnedHostPool

pool_memory = measure_pool_memory(spill=PinnedHostPool(128 << 30), device=0)
pool_memory.save("pool-memory.ssp")
```

The probe acquires and releases the full spill capacity twice. It uses a separate
4 MiB device control pool, regardless of spill capacity. Acquiring a large pinned
pool can take tens of seconds. Reuse the saved measurement for all sequence,
chunk-size and retention candidates using that configuration on compatible GPUs.
The probe does not install the framework allocator or run model operations.

CPU planning consumes the artifact without PyTorch, device initialization or
pool acquisition:

```python
from shadowspill import PoolMemoryProfile, sweep_programs

pool_memory = PoolMemoryProfile.load("pool-memory.ssp")
result = sweep_programs(
    profiled_candidates,
    memory_budgets=budgets,
    communication_info=communication,
    pool_memory=pool_memory,
)
```

`plan_program` accepts the same optional argument. The selected `Plan` saves the
evidence, including raw observations. Quickstart accepts `pool_memory=...`, or
`--pool-memory pool-memory.ssp` on the CLI, and uses it for every chunk, policy,
ordering and budget candidate. Calibration is explicit; search never reacquires
a large pool for each candidate. Existing callers without this argument retain
their current behavior and do not acquire fabricated measurement evidence.

## Two different constraints

`MemoryBudgets.execution_bytes` remains the declared process-attributable budget.
With evidence, `Plan.effective_execution_bytes` is the smaller of that budget and
the observed physical-device allowance. The latter is `total - used + own_process`
while the full spill pool is live, so it already includes device-global spill
costs. Those costs are **not subtracted from this allowance again**.

The process increment beyond the 4 MiB device control is an explicit conservative
reservation added to compute/provider overhead. It may overlap mature provider
state; we do not claim it is disjoint. Sequential named Plans and gradient
accumulation share this reservation once. They must use identical pool-memory
evidence; mixing calibrated and uncalibrated Plans is refused.

Before creating the execution arena, cold startup computes:

```text
current allowance = max(0, current total - current used + current own process
                           - measured global spill increment)
effective allowance = min(declared budget, saved allowance, current allowance)
arena bytes = floor_to_2MiB(effective allowance
                           - max(cold process baseline, profiled external floor)
                           - reserved headroom)
```

Reserved headroom includes the pool process increment once, alongside the existing
provider margin and communicator reservation. In-arena provider state stays
inside the arena. The default provider margin remains 512 MiB and
`reject_overbudget=False` remains unchanged. Physical allocation failures and
saved-layout admission are enforced independently of that reporting policy.

The native admission record reports the declared, saved, current and effective
allowances separately. Reduced current availability can refuse a previously
planned layout. Execution does not recalibrate, resize a saved layout or replan.
The adapter's configuration and admission fields require **adapter ABI3**;
rebuild Python and native libraries together. Runtime ABI3 and backend ABI1 are
unchanged.

## Identity, distributed use and limits

Evidence records exact capacity, pool kind/configuration, normal NUMA binding,
GPU model, driver and core/adapter/backend/extension library hashes. Cold
execution checks these before installing the framework allocator. GPU UUID
and hostname are retained as measurement provenance, not compatibility
requirements: a saved Plan can run on another equivalent GPU or host without
recalibrating, rebuilding or profiling again. `validate_worker` checks this
hardware/software contract for the selected logical rank.

Saved capacity is an observed planning ceiling, not a promise of free memory on
the new device. Cold admission still takes the minimum of the declared budget,
saved physical allowance and current availability after the measured global
spill reservation. It then checks that the fixed layout fits the actual arena;
pool acquisition and device-allocation failures remain enforced. Less current
capacity can therefore refuse a portable Plan. Recalibration and CPU replanning
can use a different measured allowance; execution does not do either implicitly.

Changing a checked resource requires new evidence. Compiling another model or
changing its chunk size does not by itself invalidate pool measurements.

Measure each logical DP rank using `logical_rank=...`, then combine records with
`PoolMemoryProfile.combine(*rank_profiles)`. Planning requires exactly the
Program's logical ranks and uses the least physical allowance and largest process
reservation across ranks. Execution validates the actual rank's device and uses
its global reservation. Distinct physical GPUs are required. Shared-GPU/MIG
deployment and jointly certifying aggregate host capacity are not supported by
this calibration. Actual host/remote allocation remains authoritative.

Configured extension pools implement `SpillPool.calibration_identity()` with
stable JSON values. Never serialize ctypes pointer bytes as configuration.
Scratch SSD identities include directory, staging size, chunk size and queue
depth; remote identities include endpoint and selector. Persistent SSD images
are excluded from disposable measurement.

Synchronous before/live/warmed/released observations establish **retained costs**,
not a transient startup upper bound. `startup_peak_bytes` and physical pinned
host-page counts remain `None`. Host RSS, page-table bytes and native requested
registration counters are preserved as distinct observations. An empty sampled
process inventory is not an exclusive-device lease. Detected concurrent processes,
changing global baselines or inconsistent counters refuse calibration rather than
turning missing evidence into zero. `PoolMemoryMeasurementError.observations`
preserves partial raw evidence after acquisition/observation failures. Even a
successful measurement cannot guarantee a future startup against all driver
transients or unrelated activity.
