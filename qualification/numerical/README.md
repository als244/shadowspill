# Numerical qualification

Run the five-step, two-microbatch gate in fresh reference and planned
processes:

```bash
python -m qualification.numerical.run run llama3 qualification/results/numerical
```

The reference is the same model and optimizer run by PyTorch alone, compiled
with `torch.compile` in Inductor fullgraph mode and with no part of ShadowSpill
involved: no planner, no runtime, no pools. It is not an eager run. What the
gate asserts is that planning a step does not change what that step computes.

Pure PyTorch is the default and formal numerical authority. The optional
external implementation is selected explicitly:

```bash
python -m qualification.numerical.run run llama3 qualification/results/numerical \
  --model-implementation mlops
```

Both modes build `mlops.optim.AdamW`, naming the learning rate in `hyperparams`
at planning and supplying it on every step. On SM80+ they retain BF16 weights,
gradients and moments with no masters; below SM80 they use FP16 weights and
gradients with FP32 moments. Independent `--model-dtype`, `--master-dtype`,
`--grad-dtype`, and `--opt-state-dtype` flags override that policy and travel to
both arms. The case banner and artifacts show their resolved values. See
[the precision options](../README.md#hardware-defaults-and-dtype-overrides).
Custom factories configure their model and optimizer with `--case-option`.
The original default reference root remains
`qualification/results/references/approximately_1b/`; below SM80 the default
is `qualification/results/references/pre_sm80/`. Each contains one
identity-checked final state plus an exact-input `inputs.pt` sidecar under each
`<model>/<implementation>` directory.
Pass `--reference-dir` to read that canonical set from elsewhere, or
`--regenerate-reference` to record it instead of reusing it. A reference is
specific to what produced it -- the machine, and the kernels that machine
selects. Through the gates wrapper these go in the `numerical` section of its
config rather than on its own command line, and
[`../README.md`](../README.md) covers how a per-machine reference set is
named and pointed at.

Optimizer updates are grouped by captured training stage and placed immediately
after that stage's final-microbatch backward task by default. Pass
`--optimizer-ordering tail` only for an explicit ordering comparison.

The store flags are the ones every surface uses: `--artifact-store` roots
both trees, `--build-store` and `--plan-store` override either, and
`--build-store-mode`/`--plan-store-mode` say what the run does with each; the
four modes are defined in [the artifact store
guide](../../docs/python/artifact-store.md#store-modes), and `contribute` is the
default. Run mode defaults the store below the result directory.

The command verifies five optimizer updates, two heterogeneous accumulated
microbatches per update, a step-three checkpoint whose replay agrees with the
uninterrupted run within the same per-tensor tolerance the reference
comparison uses, real EVICT/FETCH traffic, numerical tolerances, and the
physical device cap.

A disagreement is reported as one of two kinds, and the structural kind is
reported first because it decides whether the other means anything.
`structure_failures` names states that do not have the same shape as the
reference -- a tensor of a different geometry, a mapping with different keys, a
sequence of a different length. When those appear, whatever values still line
up are being compared across tensors that do not correspond, so the cosine and
relative-L2 figures describe nothing and the run is not a numerical
disagreement at all. The usual cause is that the reference was recorded against
a different configuration: changing the optimizer's parameter groups, for
instance, permutes the index a state entry is stored under, and every entry
then mismatches on shape while nothing computes wrong. That is a reference to
regenerate, not a fault to debug.

`exact_failures` is the second kind: same shape, different value, where the
value is one that must agree exactly -- an integral tensor, a scalar option.
`metric_failures` is the third: same shape, floating-point value, outside the
per-tensor tolerance. Only those two are evidence about arithmetic.

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
qualification measures process memory after planning, after each of the five
steps, and after both replay steps. With `--reject-overbudget`, the observed
process high-water must remain within the declared cap; by default an overrun
is reported without failing the gate. In both modes the gate requires one
initial execution-pool arena allocation, no steady-state pool-arena allocation,
bounded execution/spill peaks, and no allocator callback or pointer-lookup
failure.
By default, the JSON contains compact correctness, physical-budget, planning,
and step-summary evidence and planning artifacts are not retained. Add
`--detailed-artifacts` to write the complete PlanReport, per-task traces, and
the step's plan record. A plan record is the framework-free
request the search was given -- program, initial and final residency,
simulation config, search options, admission and placement facts -- beside the
answer it returned, with a digest over each, so one run's plan can be compared
against another's without either being repeated.

The SM80+ matrix retains the original roughly billion-parameter models and
execution budgets: Llama 1,179,699,200 parameters / 10 GiB; Qwen 1,006,955,408 /
10 GiB; OLMoE 975,766,528 / 8 GiB. Their default BF16 reference identities and
paths are unchanged.

Below SM80 only, the matrix defaults to four layers and vocabulary size 8,192:
Llama 251,676,672 parameters / 3 GiB; Qwen 147,224,776 / 3 GiB; OLMoE
234,963,968 / 3 GiB. Qwen keeps a complete three-linear/one-full-attention cycle.
Attention and expert widths are retained. Explicit `--model-config` fields
and per-family `--budget` values take precedence on every device.

For repeatable matrices and configurable/custom model cases, use
`python -m qualification.numerical.matrix`; the parent
[`qualification/README.md`](../README.md) documents the launcher.

The numerical and performance launchers accept `--external-headroom-mib`
(default **512 MiB on every GPU**). This allowance is subtracted, together with
the initial process baseline, when sizing the execution pool. Zero reserves no
external allowance. It does not change model or reference identity.

The independent `--reject-overbudget` flag defaults to **off**. Without it,
external and whole-process memory overruns are measured and reported. With it,
they fail the run. `--no-reject-overbudget` explicitly restores reporting mode.
Neither setting resizes the pool or permits pool overflow; actual device OOMs
still fail. Banners and artifacts record both controls, including
`external_headroom_bytes`, `reject_overbudget`, `physical_budget_enforced`, and
`physical_budget_within_limit`.
