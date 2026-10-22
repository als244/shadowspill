# ShadowSpill over the catalog models

ShadowSpill through its public training API, `Trainer` over the
`ShadowSpill` backend, over any catalog model, full or LoRA, with the shared
objective, geometry, precision contract and result row of
`baselines/common`: the same model, the same update and the same
measurement as every other engine, with the device and host budgets as the
configuration. Every case starts through the launcher
(`baselines.common.launch`), run from the repository root in the `baselines`
or `shadowspill` environment; the trainer runs in the `shadowspill`
environment (the runtime library built for this checkout; the `baselines`
environment carries the same install).

One case:

```bash
python -m baselines.common.launch --engine shadowspill --gpus 1 \
  --output <dir> \
  --model llama3-8b --tokens-per-step 65536 --tokens-per-microbatch 8192 \
  --execution-gib 16 --spill-gib 128
```

The tiny qualification matrix runs through the shared sweep, `python -m
baselines.sweep --engine shadowspill --config
baselines/engines/shadowspill/configs/qualification.json --output <dir> --gpus 1`; a
throughput matrix names only ShadowSpill's own knobs and runs inside an
experiment, which gives every engine the same model, geometry and
precision contract (`../../README.md`).

## What a case names

The common flags are the FSDP2 engine's (`../fsdp2/README.md`): the model,
the geometry, the precision contract, `--data`, `--checkpoint` and the run.
Two of them are not settings here and a value is refused: `--recompute`,
since the planner decides what to keep, recompute or spill (a row records
none), and `--allocator-cap-gib`, since the execution budget is the device
cap and PyTorch's allocator cap does not exist under ShadowSpill's
allocator. This engine adds:

| Flags | Meaning |
|---|---|
| `--execution-gib G` | The device budget: the execution pool's physical capacity, with `--external-headroom-gib` (0.5) reserved inside it for allocations outside the pool |
| `--spill-gib H` | The spill pool in pinned host memory, held whole from the first second |
| `--host-headroom-gib` | Host memory reserved per process beside the pinned pool (2) |
| `--artifact-store` | Where builds and plans persist, read where they hit and written where they miss: `default`, the library's own store (`~/.cache/shadowspill`), so every case and every experiment builds each model once; `none`, a store of the case's own that ends with it; or a directory |
| `--min-tokens-per-microbatch`, `--max-tokens-per-microbatch` | The smallest and largest microbatch the search plans, in tokens; by default one sequence and a replica's whole update |
| `--orderings` | The microbatch orderings the search tries per geometry: `factors` (every depth x breadth pair, the default) or `depth-first` |
| `--tokens-per-microbatch` | Optional: one microbatch size, which skips the search over sizes |
| `--resolution-options` | The shares of the flexible groups the planner may recompute, each planned and the best kept: `quarters` (0, 1/4, 1/2, 3/4 and 1, the planner's default), `eighths`, `halves`, or comma-separated exact fractions such as `0,1/2,1` |

The trainer is configured as the quickstart configures ShadowSpill. One
case searches the microbatch size itself: every split of a replica's update
into whole sequences within `--min-tokens-per-microbatch` and
`--max-tokens-per-microbatch` (both optional) is a candidate, each tried
under the `--orderings` (`factors`, every depth x breadth pair, by default;
or `depth-first`), and the backend's search keeps the fastest that fits the
budgets, with what to keep, recompute or spill. Each microbatch carries its
sequence lengths, so every candidate is its own step to plan.
`--tokens-per-microbatch` pins one size instead. The row names the size and
ordering the search chose. The
objective is the shared one, each microbatch's share of the update at scale
one, divided by the world size since the backend sums the ranks'
contributions. The optimizer is mlops's AdamW with the contract's dtypes.
The model is built on meta and materialized by the backend in its pools,
where its initialization and the checkpoint's values are written.

## What each offload option moves

Where each option puts the weights, gradients, optimizer state and activations,
and where the optimizer steps; an empty cell is the default's.

| Option | Weights the compute uses | Weights the optimizer steps | Gradients | Optimizer state and step | Activations |
|---|---|---|---|---|---|
| `--execution-gib G`, `--spill-gib H` | the planner's: each object is kept in the execution pool, or spilled to the pinned spill pool and fetched back before a task needs it, as its plan for the two budgets says | the planner's | the planner's | the planner's; the step on the device | the planner's, or recomputed |

## Precision

[`../../PRECISION.md`](../../PRECISION.md) sets this engine beside the others.
ShadowSpill honors the whole contract: the model dtype, masters at any
dtype or none, the dtype gradients are summed at, the optimizer state's
dtype (`parameter` for the dtype of what it steps), both roundings and
fused narrow-gradient accumulation. A rounding left unnamed is rounding to
nearest, as every other engine's AdamW rounds; ShadowSpill's own training
default, stochastic rounding of bfloat16 moments, applies when a case names
it. The mlops operations that sum a weight's gradient (norms, embeddings,
heads, experts, LoRA factors) create it at the gradient dtype inside the
captured graphs, and the optimizer reads the accumulated gradient at that
dtype, so float32 gradients there are float32 from the sum on.

The tiny matrix creates gradients at the model dtype. PyTorch's autograd
gives a weight's gradient the weight's own dtype, so the reference copy the
check compares to rounds each microbatch's gradient to bfloat16, as FSDP2
and DeepSpeed do; at the model dtype both sum the same rounded values, where
float32 gradients from the captured graphs would be finer than the copy on
every element whose microbatches cancel. The matrix measures bfloat16
compute and gradients with float32 masters and moments, bfloat16 weights
stepped in place with moments at their dtype, bfloat16 gradients the
multiplies add into themselves (fused narrow-gradient accumulation), a
float32 model and a float16 model (ShadowSpill takes float16 for
devices without bfloat16, with no loss scaling, as its quickstart does).

## What a run writes

As the FSDP2 engine: `case.json`, `rank-NNNNN/request.json` (with the
plan's record and the pools' statistics after planning), `result.json` or
`failure.json`, `results.json` and `row.json` from the first rank; also
`plan.json`, the admitted plan in the neutral schema, and `search.json` when
the backend searched (an accumulation above one). A budget the planner cannot fit is a
row on the frontier's boundary: its failure says so, under the schema's
name for the boundary (`cuda_out_of_memory`), and the sweep continues.

The tiny qualification compares every update's loss, masters and moments to
an unsharded copy compiled whole (`torch.compile(fullgraph=True)`, as the
numerical gate compiles its reference). The values come through the public
API: the stepped values and the moments from a checkpoint the trainer writes
out of the pool after each update, the buffers from the in-place state
reader. Gradients are consumed by the step inside the pool and are not
exposed; the moments the first update makes are the gradient's.

## Measurement

A case checked against the reference (`--verify-reference`) is never a
throughput result, and its plan times tasks as the numerical gate does: the
exact task's warmups, the allocation probes and the minimum samples,
without the sustained-clock conditioning and duration floors of
ShadowSpill's default profiling, which make a tiny model's search take
minutes; the request records the profiling options. Steps are timed around
`Trainer.step` with device synchronization, as every engine times its
update; the step's own host collection of its losses and
metrics is inside the time. Planning (capture, profiling, search and
admission) is timed apart as `planning_seconds` and excluded. The row's
`peak_allocated_bytes` is the execution pool's measured peak occupancy over
the process (the runtime's statistics), `peak_reserved_bytes` the execution
budget, the pool's physical capacity, and `host_peak_rss_bytes` the process
high-water mark, which holds the whole pinned spill pool;
`memory.shadowspill` records both pools' statistics, the
plan's predicted peaks and the device's used bytes beside them. PyTorch's
own allocator counters are not read: ShadowSpill installs its allocator in
PyTorch's place, and the pools' statistics are the account of the device.
