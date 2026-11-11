# Repository process orchestration

Training recipes, benchmarks and developer tools share this small package for
process infrastructure. It is available from the checkout, outside the installed
`shadowspill` wheel. It does not select a model, construct a search space,
choose a training or measurement protocol, or interpret a saved recipe directory.

| Module | Responsibility |
| --- | --- |
| `processes` | Fresh Python / single-node torchrun command construction |
| `rank_device` | Shared ordinary/profiled script/module wrapper that isolates the assigned GPU while preserving both global and local rank identities |
| `workers` | Host Gloo rendezvous, launcher-to-logical rank mapping and small result gathers |
| `lifecycle` | Append-only host phase journals, including context entry/cleanup and failures |
| `candidates` | Record a local candidate's device exhaustion after releasing its traceback; other failures propagate |
| `nsys` | CPU-only Nsight command configuration and a lazy Trainer adapter for shared capture boundaries |
| `nsys_worker` | Pre-CUDA per-rank GPU masking, collector launch and report validation; global rank paths support external multi-node launchers |

These modules import no model, recipe, benchmark or qualification implementation.
Imports are CPU-only; `workers` imports PyTorch when a distributed group is
entered, and candidate recovery imports it when an operation is attempted.
The private `_nsys_trainer` worker module is imported only when capture is enabled;
it uses cudaProfilerApi boundaries, rank barriers and the existing Executor NVTX
annotation flag. It contains no model or recipe-specific logic.
Worker coordination takes an explicit participant count and optional rank map,
independent of any client's JSON schema:

```python
from orchestration.workers import workers

with workers(world_size=8, rank_map=None) as group:
    result = run_my_phase(logical_rank=group.logical_rank)
    results = group.gather(result)
```

Group lifetime belongs to the calling phase. GPU/NCCL resources, compilation,
planning and execution remain in ShadowSpill's phase APIs. A collective
candidate failure must end its worker group; local OOM recovery is for
independent single-worker attempts only.

The clients keep their own policies:

- `workloads/recipes` owns workload selection, data, initialization and saved
  recipe configuration, and composes the public training phases.
- `benchmarking/quickstart` owns its candidate factory and budget sweep protocol.
- `benchmarking/plan_runner` owns saved-Plan benchmarking, warmup exclusion,
  per-rank timing aggregation, GPU sampling and offline reports. Its `recipe`
  module is an explicit consumer of saved workload recipes; its generic `run`
  API accepts caller hooks and has no recipe dependency.

The installed engine never imports this repository package. Neither recipes nor
this package import benchmark implementation utilities. Repository tests enforce
those dependency boundaries.
