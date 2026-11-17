# Installed process orchestration

Training recipes, benchmarks and developer tools share this small package for
process infrastructure and communication-request composition. It is installed as
`shadowspill.orchestration` in the engine wheel. It does not select
a model, construct a search space,
choose a training or measurement protocol, or interpret a saved recipe directory.

| Module | Responsibility |
| --- | --- |
| `launch/processes` | Fresh Python / single-node torchrun command construction |
| `launch/rank_device` | Shared ordinary/profiled script/module wrapper that isolates the assigned GPU while preserving both global and local rank identities |
| `launch/workers` | Host Gloo rendezvous, launcher-to-logical rank mapping and small result gathers |
| `launch/staging` | Quota-bound node-local artifact delivery, complete size/checksum checks and consumer ownership |
| `launch/supervision` | Fresh owned process groups, non-reaping exit observation, bounded logs and orderly/forced shutdown |
| `launch/connected` | One inherited private control socket per owned child, startup I/O timeout and channel-before-process shutdown |
| `launch/admission` | Atomic assigned-device and host-memory claims retained through owned child cleanup |
| `monitoring/lifecycle` | Append-only host phase journals, including context entry/cleanup and failures |
| `candidates` | Record a local candidate's device exhaustion after releasing its traceback; other failures propagate |
| `communication` | Union exact collective requests across compatible caller-selected Programs; measurement remains in the public phase APIs |
| `capture/nsys` | CPU-only Nsight command configuration and runtime capture boundaries |
| `capture/nsys_worker` | Pre-CUDA per-rank GPU masking, collector launch and report validation; global rank paths support external multi-node launchers |

These modules import no model, recipe, benchmark or qualification implementation.
Imports are CPU-only; `launch/workers` imports PyTorch when a distributed group is
entered, and candidate recovery imports it when an operation is attempted.
Communication request composition imports the engine only when called and
consumes Programs one at a time. It requires the same build setup and declared
parallelism, preserving collective sizes, dtypes, ordered ranks and aliasing.
Recipes and quickstart measure this union once per phase invocation; each
candidate still receives its own task/communication overlap measurement. This
does not reuse calibration across machines or separate phase invocations.
`capture/_nsys_runtime.UpdateCapture` controls cudaProfilerApi boundaries and rank barriers.
Executor supplies task annotations; capture contains no model-specific logic.
The collector keeps its session open after the single capture range so Nsight
converts the report after worker exit, when training pools have been released.
Worker coordination takes an explicit participant count and optional rank map,
independent of any client's JSON schema:

```python
from shadowspill.orchestration.launch.workers import workers

with workers(world_size=8, rank_map=None) as group:
    result = run_my_phase(logical_rank=group.logical_rank)
    results = group.gather(result)
```

Group lifetime belongs to the calling phase. GPU/NCCL resources, compilation,
planning and execution remain in ShadowSpill's phase APIs. A collective
candidate failure must end its worker group; local OOM recovery is for
independent single-worker attempts only.

The clients keep their own policies:

- `packages/workloads/src/shadowspill_workloads/recipes` owns workload selection, data, initialization and saved
  recipe configuration, and composes the public training phases.
- `benchmarks/quickstart` owns its candidate factory and budget sweep protocol.
- `benchmarks/plan_runner` owns saved-Plan benchmarking, warmup exclusion,
  per-rank timing aggregation, GPU sampling and offline reports. Its `recipe`
  module is an explicit consumer of saved workload recipes; its generic `run`
  API accepts caller hooks and has no recipe dependency.

This package imports no repository client or model catalog. Repository tests
enforce those dependency boundaries, including CPU-only import checks.

Node-local `FileDelivery` takes opaque portable artifact references, an explicit
cache root and shared disk/memory quotas. It reserves complete file volume and
metadata/copy scratch before creating files or opening authenticated source
streams. Sources supply bounded reads and transport deadlines. Every file must
match its declared length and checksum before the delivery becomes ready. It
copies in 64 KiB chunks and releases scratch before exposing ready files.

Consumers retain a `use()` context until workers finish or their agents confirm
termination. Cleanup refuses to delete files still borrowed or owned by an active
producer. Cancellation takes effect between reads and retains reservations while
a read is blocked. A failed deletion retains its quotas until cleanup succeeds.
Files are ephemeral and never establish durable completion or execution replay.
Agent startup, supervision and allocation admission still belong to the caller.

`OwnedProcess` starts a trusted installed command in a fresh POSIX session with
an explicit environment and working directory. Private worker control has its own
stream; stdout/stderr are drained independently into capped log files. Excess log
bytes or log-write errors are reported while draining continues, so logging cannot
block a child. This is neither a scheduler allocation request nor an idle timeout.

Exit observation uses `waitid(..., WNOWAIT)` to preserve the leader's PID until
owned-descendant cleanup finishes. Closing the private control streams first
permits orderly worker/report finalization. Explicit process close waits for that
exit, then escalates TERM/KILL only for its owned process group if needed. Any
remaining group descendants are removed before reaping the leader, preventing
PID reuse from targeting unrelated work. Non-reaping observation is required on
the worker host. Startup handshakes, assigned-device admission and agent/service
authentication are separate integration responsibilities.

`ConnectedProcess` composes that supervisor with a private inherited socket pair.
Its installed command factory receives only the child descriptor number; no
public listener or request-selected executable is introduced. The parent drops
its copy of the child end immediately after spawn, so child death produces EOF.
Startup reads/writes have a configurable I/O timeout. After the caller verifies
the startup barrier, `ready()` removes that timeout; live workers have no default
idle timeout. Close shuts down the socket before waiting on the owned process,
waking blocked readers and permitting orderly worker/report finalization. Actual
child tests cover startup, commands, log pressure, lost replies and timeout cleanup.

`NodeAdmission` accepts trusted `NodeCapacity` facts and complete `ResourceRequest`
demands. Device slots are exclusive; host bytes and claim count are
reserved atomically before launch. Host demand includes decoded artifacts,
framework/process overhead, spill, initialization, input/output staging and agent
delivery. Disk is admitted by file/artifact owners and remains reserved until
file cleanup or storage handoff; stopping workers alone cannot release it. These
amounts must be computed by the caller, not copied from an untrusted capacity
claim. Hardware/cgroup discovery and complete training-demand calculation remain
agent integration work.

Impossible requests fail immediately even when waiting is requested. Temporary
shortages fail by default or wait explicitly, with an optional resource timeout.
A `NodeLease` launches once per reserved device and sets its assigned visibility
before the child imports code. Close disconnects all peers before waiting on any,
then returns resources only after every owned process finishes cleanup. Failed
cleanup retains the entire claim for a later retry. Closing admission wakes
waiting callers but leaves existing leases responsible for their live workers;
it does not release a scheduler allocation.
