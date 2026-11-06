# Actual FP16 regions and distributed training

This regression matrix combines actual FP16 parameters and gradients with
checkpoint partitions, per-region arbitrary-axis chunks, nested composition,
selective retention, lifetime policies and two-rank DP. The model has tied
projections, nonzero LoRA factors, odd shard sizes, two unequal rank-local
microbatches and a globally normalized objective. Cases cover full SAVE training,
sharded LoRA with retained sequence operations and custom checkpoint cuts, and
sharded batch-axis LoRA with fresh/continuing/fresh gradient accumulation.

Run the CPU readiness checks without a GPU:

```bash
CUDA_VISIBLE_DEVICES= python -m pytest tests/shadowspill/build/test_narrow_region_fixture.py
```

Run the saved-phase integration on two distinct GPUs:

```bash
python -m pytest tests/shadowspill/execution/narrow_regions -m cuda -v
```

Every integration phase has its own process and timeout. Build and profiling
produce saved artifacts; planning imports no PyTorch and has devices hidden.
Execution loads only the selected Plan with preparation, compilation and tuning
forbidden. It uses a 4 GiB process budget, a 64 MiB pinned-host pool, permuted
logical ranks, bounded initialization/exports, and physical admission checks.
Full and LoRA optimizer cases retain FP32 masters and Adam state. Continuing
gradients, complete collective order/geometry, checkpoint restoration and
fresh-process resume are checked explicitly. These tiny cases do not force
host spills or qualify attention/MoE provider kernels.

The numerical authority is ordinary CUDA autograd compiled with
`torch.compile(fullgraph=True, dynamic=False)` under normal precision
(`emulate_precision_casts=False`). It reads only the original input/weight
fixture, never a Program, Plan, captured VJP or execution activation. Each
logical rank accumulates its microbatches in FP16 before the two rank
contributions are added; only then are gradients promoted for FP32 masters.
Two complete reference trajectories must repeat bitwise. Full tensors are
reconstructed from the executed shards and compared using the unchanged
`qualification.numerical.tolerances` criteria. Frozen state and resumed state
must match bitwise.

The CPU control validates reference bookkeeping and region VJPs; its rounding
is not used to accept GPU training. FP16 cancellation can change a gradient sign
between CPU, CUDA eager and normal compiled execution. Selecting an explicit
independent reference contract avoids treating those modes as identical.
