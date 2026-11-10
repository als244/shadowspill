# Training from Python

Use a model source and ordinary Python configuration. The shared runner owns
build, profiling, communication calibration, candidate planning and execution.
The checkpoint's configuration selects its workload adapter internally.

```python
from workloads.recipes import LoRA, Model, train

model = Model.from_checkpoint(
    "/checkpoints/Qwen3.5-35B-A3B", lora=LoRA(rank=32, alpha=32),
)
train(model, {
    "run_dir": "runs/finetune",
    "data": "/data/tokenized-fineweb",
    "frozen_state_id": "exact-base-checkpoint-revision",
    "sequence_length": 65536,
    "sequences_per_step": 4,
    "microbatch_sizes": [1, 2, 4],
    "chunk_sizes": [8192],
    "moe_chunk_sizes": [8192, 32768],
    "execution_gib": 77,
    "spill_gib": 200,
})
```

Run the script once, without an outer `torchrun`. A new build needs a fresh run
directory. `Model` describes the source; it does not load model tensors or
initialize CUDA. `train` launches fresh phase workers and returns the run path
after completion. Imports and configuration do not start training.

The checkpoint adapters currently support Qwen3.5 MoE and GLM-5.3-Flash LoRA.
For full training from random weights, use
`Model.from_config("qwen3moe-30b-a3b")`. That catalog adapter supports sharded DP
and EP. Arbitrary model factories are not yet exposed by this convenience API;
the generic [build APIs](../../docs/python/api/frontend.md) and
[quickstart factories](../../benchmarking/quickstart.md) remain available.

See the commented scripts for [Qwen35 LoRA](text/qwen35_lora/example.py),
[GLM LoRA](text/glm53_flash_lora/example.py) and
[Qwen3 full training](text/qwen3/example.py). Each adapter documents its dtype,
data and parallelism requirements. The example budgets must fit the actual
machine; pinned spill also needs host RAM for the process and checkpoint I/O.

## Search and saved phases

`sequences_per_step` is the number per rank and optimizer update.
`microbatch_sizes` lists the exact positive divisors to build. Changing its
choice does not change the update's global token count.

Text adapters cross those sizes with `chunk_sizes`, `moe_chunk_sizes` and
`graph_pair_policies`. Zero projection size means unchunked; zero MoE size
inherits the projection size. Each distinct implementation is built and
profiled. Planning tries all factor orderings and PressureFit choices, then
selects the lowest predicted training time among admitted candidates. When
evaluation is enabled, training and evaluation must also fit their shared
runtime allocation. Adding candidates expands the search; it cannot remove
an existing admitted choice.

| Graph-pair choice | SAVE variant | RECOMPUTE variant |
|---|---|---|
| `cost-input` | Cost-based retention | Inputs and required routing choices |
| `cost-attention` | Cost-based retention | Inputs, routing choices and selected attention backward state |
| `attention-input` | Selected attention backward state and routing choices | Inputs and routing choices |

Attention selection is adapter-owned: Qwen's policy covers dense attention;
GLM's covers its KDA and sparse attention operations. Chunk-local expert scratch
must be regenerated within chunks. These choices do not change checkpoint cuts
or require GPU residency; PressureFit controls saving, recomputation and spill.
`resolution_options` controls which SAVE/RECOMPUTE mixtures planning considers.

Local build/profile GPU exhaustion is recorded in `candidate-failures.json`
and `search.json`, and other candidates remain eligible. Other errors fail the
phase. Collective preparation failures terminate the worker group; a rank must
not skip a collective independently. EP currently supports unchunked execution
only because its provider does not declare an execution-axis contract.

```python
saved = {"run_dir": "runs/finetune"}
train(model, saved, phase="profile")
train(model, {**saved, "execution_gib": 40}, phase="plan")
train(model, saved, phase="execute")
```

To compare several budgets on CPU before requesting GPUs, reuse one completed
preparation in separate run directories:

```python
for budget in (20, 40, 77):
    train(model, {
        "run_dir": f"runs/budget-{budget}",
        "prepared_from": "runs/prepared",
        "execution_gib": budget,
    }, phase="plan")
# Later, on the matching hardware:
train(model, {"run_dir": "runs/budget-20"}, phase="execute")
```

`prepared_from` requires completed task and communication measurements and
inherits frozen build choices. The new run links the prepared artifacts and
owns its Plans, reports, logs and checkpoints. Keep the source available for
replanning. Such a run may only plan or execute, so it cannot overwrite its
source measurements. Its directory must be outside the source run. Training
starts fresh unless `resume` is explicit.

Phases are `build`, `profile`, `communicate`, `plan` and `execute`. With one rank,
profiling also measures host transfers, so omit `communicate`. Multi-rank runs
calibrate communication and computation overlap on the participating devices.
DP build/profile uses one representative GPU; EP preparation needs all resource
participants. Planning reads saved artifacts on CPU without importing Torch.

Repeat `train(model, saved, phase="build")` to continue an interrupted build.
Completed training and evaluation Programs are checked independently against
the frozen configuration, adapter metadata, current target and numerical settings.
Missing Programs are built; invalid completed artifacts stop with an error.
Collective workers agree on reuse. Recorded exhausted candidates remain excluded.
This preserves completed work across allocations; it does not shorten a fresh build.

Repeating `phase="profile"` reuses completed per-Program measurements after
validating their implementation, memory evidence, headroom and current target.
An interrupted candidate is measured again; completed ones are preserved. All
collective workers agree on reuse before invoking tasks. Set `refresh_profiles=True`
in that phase's configuration to request fresh measurements. Run `communicate`
(multiple ranks) and `plan` before executing refreshed measurements. Single-rank
profiling also recalibrates host transfers.

Build settings are frozen. Changing shape, precision, candidates or profiling
headroom requires a fresh build. Planning may change budgets, search settings
or the optional `pool_memory` path to a measured `PoolMemoryProfile`. That
artifact must match the exact spill capacity and all logical ranks; its physical
allowance already includes the observed global pool-registration charge.
See [pool calibration](../../docs/python/api/pool-memory.md).

Execution loads the selected Plans, initializes their buffers using bounded
adapter callbacks, and calls the ordinary `Trainer`. `checkpoint_every`,
`keep_last` and `resume` control checkpoints. A distributed resume path may
contain `{rank}`. `steps` is the total desired update count, including restored
updates; keep `schedule_steps` fixed across a continuation.

## Adapter boundary

The runner in [training/](training/) has no architecture, operator, checkpoint
format or provider-cache knowledge. Internal adapters supply:

| Hook | Contract |
|---|---|
| `defaults`, `validate(options)` | Workload configuration and required preparation participants |
| `candidates(options)` | JSON descriptions with unique relative `path` and positive `num_examples` |
| `inputs(options, rank, split)` | Stateful input source with `example()` and checkpoint cursor support |
| `setup(options, rank)` | Context yielding `BuildSetup(model, optimizer, objective, build_options, metadata)` |
| `initializer(options, metadata, rank)` | Context yielding a bounded tensor-filling callback |
| `trainer_options(options, metadata)` | Schedules, metrics and frozen-state identity |
| `evaluation_metric(value)` | Workload metric reduction when evaluation is enabled |

Optional `trainer_type`, `finish` and `runtime_options` hooks serve specialized
instrumentation and receipts. Setup runs separately for every candidate so
token-dependent collective resources retain the correct capacity. Model and
provider code do not implement execution chunk loops; adapters choose explicit
compiler contracts. Saved adapter module names are worker metadata, not public
call arguments.
