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
`tail_save_counts` is a list of additional planning choices, automatically crossed
with every resolution option:

```python
"tail_save_counts": [0, 1, 2, 4, 8, 16],
"resolution_options": ["0", "1/4", "1/2", "3/4", "1"],
```

For each chosen count, save the last K eligible forward invocations in **each**
forward/backward pass, repeated at every depth. Apply the resolution's recompute
fraction to the remaining flexible invocations, evenly distributed in their
stable group order. Thus one choice can retain both a tail and earlier values.
Required saves do not count toward K. Counts larger than a pass saturate, and
identical complete selections are planned once. The default is `[0]`, preserving
the original search space; include zero to keep it when adding tail choices.
Small inventories are still exhaustive. These controls only require replanning:
they add no build or profile candidates and do not change GPU-versus-host placement.
Nonzero tails require non-interleaved forward/backward passes, as emitted by the
training orderings. They use generic task phases, without recognizing model types.

For example, if an update has two passes with eight eligible invocations each,
K=2 forces four tail saves across the update. A recompute share of `"1/2"` then
recomputes six of the twelve remaining invocations and saves the other six.
The two controls therefore produce ten flexible saves, plus any required saves.
They choose a graph-pair variant per invocation; they do not force those saved
tensors to remain on GPU. The graph-pair policy decides what each variant retains,
and PressureFit chooses residency and transfers within the device/spill budgets.

The saved config and Plan search key record both lists. Change them using
`train(model, {"run_dir": saved_run, "tail_save_counts": [0, 1, 2, 4, 8, 16]},
phase="plan")`; the runner rejects changing them only at execution time.
Planning takes the minimum admitted predicted duration over the crossed search.
It is a finite heuristic search, not an exhaustive optimization of all schedules.

CPU planning has two independent concurrency controls:

| Option | Default | Work assigned to each worker |
|---|---:|---|
| `program_workers` | `8` | One built/profiled Program, searching its factor orderings sequentially in a separate process |
| `search_workers` | `8` | Native PressureFit worker threads within each Program process, searching the resolutions and placement policies for the current ordering |

For example, three graph-pair policies, two chunk choices and three microbatch
sizes produce 18 Programs. Their factor orderings may produce more than 18
planning cases. A Program worker loads its measurements once, processes all its
orderings, and saves admitted Plans directly; it does not send large Plans
between processes. All cases remain eligible, and selection uses the minimum
predicted duration with ties resolved in configured candidate/ordering order.
Known capacity failures remain in `search.json`; other worker errors fail the
phase and leave no runnable selection.

The two counts are not divided into a shared CPU quota: eight Program workers
with eight search workers permit up to 64 native search threads, plus their
Python processes. This allows other Programs to progress while one is preparing
search inputs or saving a Plan. Concurrent Programs also use more host RAM.
Set `program_workers=1` for serial Program processing; `search_workers=0` selects
the native library's automatic thread count in each process. Both controls may
change without rebuilding. They apply only to CPU planning. `planning-work.json`
records actual process concurrency, per-Program elapsed time and case counts,
and total planning wall time; summed worker time is not wall time.

For performance sweeps, generate weights in host memory and omit evaluation and
checkpoint I/O:

```python
model = Model.from_checkpoint(model_path, lora=LoRA(rank=32, alpha=32))
train(model, {
    **config,
    "initialization": "random",  # Bounded host generation; no weight payload reads.
    "seed": 0,
    "eval_every": 0,
    "checkpoint_every": 0,
})
```

Qwen3 `Model.from_config(...)` already uses random host initialization. Checkpoint
recipes default to `"checkpoint"`; choose `"random"` explicitly for a benchmark.
Qwen35 needs only `config.json` for this mode. GLM also reads tensor headers to
preserve its mixed BF16/FP8 storage and scale geometry, plus tokenizer metadata;
it does not load pretrained tensor values. Existing bounded family initializers
generate correct norm, gate, scale and LoRA values, with the seed independent of
load chunks and rank shards. No full extra model is materialized on GPU or host.
Generated frozen bases get a separate seed/configuration identity. Initialization
is recorded with the build; benchmark losses are not pretrained-model quality
measurements. Real training retains its checkpoint initialization default.

Set evaluation and checkpoint cadences to zero before preparation for a benchmark.
Checkpoint cadence and retention count can also change on a prepared run without
rebuilding or replanning. Checkpoint reload tests and reference/state audits are
separate opt-in validation work, not part of a throughput sweep.

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
adapter callbacks, and calls the ordinary `Trainer`. The default pinned spill
pool reserves the selected worker's required peak plus 5%, capped by `spill_gib`;
the configured spill budget remains the search ceiling. Joint training/evaluation
and accumulation storage is included. The actual per-rank `spill_capacity_bytes`
is printed at initialization and recorded in `completed.json`. See
[execution memory](../../docs/python/api/training.md) for explicit pool sizing.
`checkpoint_every`,
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
