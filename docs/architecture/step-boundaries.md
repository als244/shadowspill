# Step boundaries

A plan describes one invocation from its declared initial residency through
its required final residency. Search, admission, simulation, execution, and
tracing consume the same memory actions. Repeating a callable starts another
invocation after the preceding invocation has reached its final state.

## Fresh and explicitly resident inputs

**Fresh** means no plan-owned value is initially on the execution device.
The PyTorch frontend emits fresh plans: imported model state and external
inputs begin in spill, and every needed device copy is a scheduled fetch.
Fresh is a description of the state, not a placement policy or CLI option.

The neutral planner also accepts explicitly GPU-resident inputs through
`initial_residency`. Those inputs need no fetch. The declaration must match
execution: a missing or stale device value is an error, and an unexpected
live allocation is not silently overwritten. Shared objects retain their
separate ownership and residency contracts.

Neither a reserved device range nor capacity in a slab means that an object's
bytes are present. Physical admission distinguishes an initial device object
from a destination reserved for a future fetch.

## One schedule from invocation start

PyTorch lowering emits a zero-duration `ResourceKind.CONTROL` task before
model computation. It has no callable, inputs, outputs, or workspace. Its
ordinary `after_task` boundary can trigger ordinary `FETCH` actions.

For example, with a spill-resident input and a required final writeback:

```text
start control → fetch input → compute/update → write back result → complete
```

The simulator charges both copies. Candidate search sees them before it
chooses a plan, physical admission reserves their destinations, and runtime
uses the same task/action identities. Each consumer waits only for its own
inputs and allocation reuse dependencies. Later-input fetches may overlap
computation normally.

There is no separately reconstructed entry transfer batch. Background action
batches remain available for lifecycle operations such as materializing or
reconciling model state; they do not implement a planned invocation's fetches.

## Repeated execution

At an invocation boundary the executor:

1. Waits for the preceding invocation's required terminal work to finish.
2. Records this invocation's origin and, when requested, enables its trace.
3. Refreshes external inputs and validates actual initial residency.
4. Runs the start control task and the selected computational tasks through
   their ordinary before/after boundaries.
5. Returns caller-owned results. Required terminal transfers may still run.

The next invocation's initial wait completes the previous invocation, not
this one's entry work. `mark_cycle_end()` performs the same wait before
recording the final cycle marker when no next invocation follows.

## Matching time boundaries

Simulation starts at invocation entry, time zero. The trace records its origin
at the corresponding execution boundary. Neither timeline is shifted to the
first compute start, and no measured startup constant is added to simulation.
CPU staging or dispatch delay before the first transfer remains visible.

Both invocations split into:

```text
entry delay + task window + terminal transfers = invocation duration
```

Entry delay ends at the first computational task. The task window contains
task execution and gaps between tasks. Terminal duration ends when all
required computation and transfers have completed. Overlapping copies are
counted by their completion times, not added together. Control tasks retain
trace identities for their actions but are excluded from model-compute totals.
A lane that supplies no device timestamps leaves the measured invocation
completion unavailable rather than making its copies appear instantaneous.

**Whole-cycle time** is a separate throughput measurement: one invocation's
origin to the next origin, or the final cycle marker. It includes caller and
instrumentation work between invocations as well as required terminal work.
Three always-on events partition it into `entry_delay_seconds`,
`selected_span_seconds`, and `exposed_tail_seconds`.

Detailed tracing is optional. Call `prepare_runtime_trace()` before a measured
loop to allocate its buffers in advance; quickstart does this automatically.
Calling it repeatedly is harmless. Lazy preparation remains available when a
caller first requests `runtime_trace=True`.

See [timelines](timelines.md), [the timing API](../python/api/timing.md), and
[step diagnostics](../python/step-diagnostics.md) for the recorded fields.
