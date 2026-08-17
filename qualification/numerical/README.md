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
forward/backward operation provider. The default reference root is
`qualification/results/references/approximately_1b/`, with one
identity-checked final state plus an exact-input `inputs.pt` sidecar under each
model/provider directory.
Pass `--reference-dir` to place that canonical set elsewhere, or
`--regenerate-reference` to replace it deliberately.

Optimizer updates are grouped by captured training stage and placed immediately
after that stage's final-microbatch backward task by default. Pass
`--optimizer-ordering tail` only for an explicit ordering comparison.

The command verifies five optimizer updates, two heterogeneous accumulated
microbatches per update, a step-three checkpoint followed by bitwise replay,
real EVICT/FETCH traffic, numerical tolerances, and the physical device cap.
Recomputation availability and selection are reported diagnostically but are
not independently required for these tiny geometries. The JSON artifact
records all tolerances and planning phase timings used for that run. Physical
qualification checks the sealed cap
after planning, after each of the five steps, and after both replay steps. It
also requires the observed process high-water to remain within the public cap,
one initial execution-pool arena allocation, no steady-state pool-arena
allocation, bounded execution/spill peaks, and no allocator callback or
pointer-lookup failure.
By default, the JSON contains compact correctness, physical-budget, planning,
and step-summary evidence and planning artifacts are not retained. Add
`--detailed-artifacts` to write the complete PlanReport, per-task traces, and
canonical initial/recurrent PressureFit fixtures. Those fixtures contain the
framework-free arguments passed to `pressurefit()` and its complete result.

The supported matrix uses a 10 GiB execution cap for Llama and Qwen and an
8 GiB cap for OLMoE, which is required to exercise real transfer pressure for
the small qualification geometry.

For repeatable matrices and configurable/custom model cases, use
`python -m qualification.numerical.matrix`; the parent
[`qualification/README.md`](../README.md) documents the launcher.
