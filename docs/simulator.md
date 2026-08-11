# Standalone simulator

`shadowspill.simulator` replays an explicit `Program` and `MemorySchedule`. It
does not import or invoke the planner, PyTorch, CUDA, or model code.

```python
from shadowspill.simulator import SimulationConfig, simulate

config = SimulationConfig.single_device(
    "cuda_0",
    device_capacity_bytes=24 << 30,
    host_capacity_bytes=64 << 30,
    h2d_bandwidth_bytes_per_second=24_000_000_000,
    d2h_bandwidth_bytes_per_second=24_000_000_000,
)

result = simulate(program, schedule, selections=selections, config=config)
print(result.makespan_ns)
print(result.device_peak("cuda_0").total_bytes)
```

All capacities and timestamps are integers. Transfer duration is latency plus
`ceil(bytes * 1_000_000_000 / bytes_per_second)`, computed without floating
point.

## Timing contract

- Tasks become eligible after their dependencies complete and their declared
  resource lane is available.
- Distinct resource lanes may overlap. Declaration order is the deterministic
  tie-break and prevents tasks on one lane from overtaking one another.
- A task starts only when every input alias is device-ready and its output
  allocations plus task-local workspace fit the device capacity.
- Output storage and workspace are charged over the active task interval.
  Workspace is released at completion; outputs become ready at completion.
- Memory actions are submitted after their trigger task. Their global plan
  order is preserved even if tasks on different lanes finish out of order.
- Each device has one FIFO H2D lane and one FIFO D2H lane. The directions may
  overlap each other and computation.
- D2H reserves host capacity at transfer start and retains device capacity until
  completion. H2D reserves device capacity at transfer start.
- A prefetch submitted while the same alias is offloading waits for D2H
  completion. It does not overlap the opposite transfer of that alias.
- Simulation drains submitted transfers and then verifies required final
  residency.

Host backing retained by an alias is physical from simulation start. A
non-retained host extent is released after H2D completes.

## Results and failures

`SimulationResult` contains task intervals, transfer intervals, per-device
object/workspace/total peaks, host peak, and makespan. Every interval exposes
its earliest readiness time, actual start, and stable stall-reason labels.
Optional memory snapshots are available through `record_timeline=True`.

Infeasible schedules raise `SimulationInfeasibleError`. Callers should consume
its structured `kind`, time, task, alias, location, capacity, used, and requested
fields rather than parse the message. Queued transfer constraints are diagnosed
before downstream task-input waits so the error identifies the physical root
cause.

## Implementations

Installed wheels use `libshadowspill_simulator.so`. The source-tree Python
implementation is a readable differential oracle and also supplies optional
memory snapshots. Both implementations use the same public records and are
required to return identical successful results and capacity diagnostics.
