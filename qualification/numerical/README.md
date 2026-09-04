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
Pass `--reference-dir` to read that canonical set from elsewhere, or
`--regenerate-reference` to record it instead of reusing it. A reference is
specific to what produced it -- the machine, and the kernels that machine
selects -- so a set recorded elsewhere belongs in a directory named for what
recorded it, and the gate is pointed at the one that matches. Through the
gates wrapper these go in the `numerical` section of its config rather than on
its own command line; see [../README.md](../README.md).

Optimizer updates are grouped by captured training stage and placed immediately
after that stage's final-microbatch backward task by default. Pass
`--optimizer-ordering tail` only for an explicit ordering comparison.

The command verifies five optimizer updates, two heterogeneous accumulated
microbatches per update, a step-three checkpoint whose replay agrees with the
uninterrupted run within the same per-tensor tolerance the reference
comparison uses, real EVICT/FETCH traffic, numerical tolerances, and the
physical device cap.

The replay is answered two ways, and both are reported.
`checkpoint_replay_bitwise` says whether it agreed exactly, and
`checkpoint_replay_within_tolerance` whether it agreed within the per-tensor
tolerance; the second is what the verdict keys off. They are separated because
a step can only replay bitwise if every kernel under it does, which is a
property of the kernels rather than of the replay: a cell that used a kernel
summing with atomics would fail an exact check while computing nothing wrong.
Qualification asks the operations that offer the choice for their ordered
variant, so in practice both answers hold, and the bitwise one is the earlier
warning if a kernel stops being reproducible.
Recomputation availability and selection are reported diagnostically but are
not independently required for these tiny geometries. The JSON artifact
records all tolerances and planning phase timings used for that run, and, when
a cell fails, the category of each failure -- a reference disagreement, a
replay that did not reproduce, a budget exceeded -- with the failing tensor
counts split into model state and optimizer state. Physical
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
