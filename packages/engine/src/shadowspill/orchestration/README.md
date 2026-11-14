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
| `monitoring/lifecycle` | Append-only host phase journals, including context entry/cleanup and failures |
| `candidates` | Record a local candidate's device exhaustion after releasing its traceback; other failures propagate |
| `communication` | Union exact collective requests across compatible caller-selected Programs; measurement remains in the public phase APIs |
| `capture/nsys` | CPU-only Nsight command configuration and a lazy Trainer adapter for shared capture boundaries |
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
The private `capture/_nsys_trainer` worker module is imported only when capture is enabled;
it uses cudaProfilerApi boundaries, rank barriers and the existing Executor NVTX
annotation flag. It contains no model or recipe-specific logic.
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
