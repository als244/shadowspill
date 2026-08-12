# Numerical qualification

Run the five-step, two-microbatch gate in fresh eager and planned processes:

```bash
python -m qualification.numerical.run run llama3 qualification/results/numerical
```

Pure PyTorch is the default and formal numerical authority. The optional
external implementation is selected explicitly:

```bash
python -m qualification.numerical.run run llama3 qualification/results/numerical \
  --model-implementation mlops
```

Both modes use `mlops.optim.AdamW`; the model implementation changes only the
forward/backward operation provider. Results and large eager checkpoints are
written beneath the gitignored `qualification/results/` tree with a
`pytorch_` or `mlops_` filename prefix.

The command verifies five optimizer updates, two heterogeneous accumulated
microbatches per update, a step-three checkpoint followed by bitwise replay,
real EVICT/FETCH traffic, selected recomputation, numerical tolerances, and the
physical device cap. The JSON artifact records all tolerances and planning
phase timings used for that run. Physical qualification checks the sealed cap
after planning, after each of the five steps, and after both replay steps. It
also requires the observed process high-water to remain within the public cap,
one initial CUDA slab allocation, no steady-state device or pinned allocation,
bounded slab/host peaks, and no allocator callback or pointer-lookup failure.
Each planned case also writes canonical initial/recurrent PressureFit fixtures.
They contain exactly the framework-free arguments passed to `pressurefit()`—the
Program, residency inputs, simulator configuration, and planner options—and
exactly its expected selections, schedule, full simulator result, and candidate
diagnostics. Frontend capture, profiling, physical admission, ExecutionPlan
construction, and outer `plan_step()` timing are intentionally excluded.

For repeatable matrices and configurable/custom model cases, use
`python -m verification.run_model_correctness`; its README documents model
configuration, data geometry, factory, and case-option arguments.
