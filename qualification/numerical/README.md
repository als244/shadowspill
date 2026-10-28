# Numerical qualification

Run the five-step, two-microbatch gate in fresh reference and planned
processes:

```bash
python -m qualification.numerical.run run llama3 qualification/results/numerical
```

The reference is the same model and optimizer run by PyTorch alone, compiled
with `torch.compile` in Inductor fullgraph mode and with no part of ShadowSpill
involved: no planner, no runtime, no pools. It is not an eager run. Before
either arm, a warm-up process compiles and runs every microbatch once and
exits: kernels that tune on first use (Quack's autotuner under the SonicMoE
implementations) are tuned there, because a process that tunes a kernel
computes slightly differently from one that reads the tuned choice, for the
rest of its life, and both arms must be readers of the same cached choices.
What the gate asserts is that planning a step does not change what that step
computes.

The default matrix uses pinned-host spill. SSD/remote settings flow through the
same saved phases to communication calibration and Executor; build/task profiling
remain independent of the backend. See [SSD qualification](ssd.md).

Pure PyTorch is the default and formal numerical authority. The optional
external implementation is selected explicitly:

```bash
python -m qualification.numerical.run run llama3 qualification/results/numerical \
  --model-implementation mlops
```

Both modes use `mlops.optim.AdamW`. Build records its host hyperparameters;
execution supplies the learning rate on each invocation. On SM80+ they retain BF16 weights,
gradients and moments with no masters; below SM80 they use FP16 weights and
gradients with FP32 moments. Independent `--model-dtype`, `--master-dtype`,
`--grad-dtype`, and `--opt-state-dtype` flags override that policy and travel to
both arms. The case banner and artifacts show their resolved values. See
[the precision options](../hardware.md).
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
[`../README.md`](../README.md) covers recording and replacing the set.

Optimizer updates are grouped by captured training stage and placed immediately
after that stage's final-microbatch backward task by default. Pass
`--optimizer-ordering tail` only for an explicit ordering comparison.

The planned arm uses four separate processes, shared with the performance gate:

1. `prepare.py`: construct the case, call `build_step`, save `program.ssp` and
   exact CPU initialization/input values in `inputs.pt`.
2. `qualification.pipeline.profile`: load the Program, measure task timing/memory
   with the ordinary allocator, save `profiles.ssp`, `profiled.ssp` and
   `communication.ssp`.
3. `qualification.pipeline.plan`: load the saved evidence and plan on CPU, with
   devices hidden and no PyTorch import. Save `plan.ssp` and `summary.json`.
4. `planned.py`: start a fresh execution worker, load Plan into Executor,
   initialize through bounded buffers, run and compare.

Each result has a sibling `<case>.phases/` directory containing these artifacts,
settings, and `phases.json` timings. The archives are retained independently of
compiler caches, except `inputs.pt` and `checkpoint.ssp`: each is the size of
the model, only execution reads them, and they are removed once the verdict is
written unless `--retain-execution-files` asks for an execution-only rerun. A
failed case keeps them. To replan a case, edit its `settings.json` and run:

```bash
python -m qualification.pipeline.plan /path/to/case.phases
```

`run.py` only orchestrates the reference and these phases. Shared process,
initialization and observation code lives in `qualification/pipeline/`; there is
no second model construction or profiling path inside execution.

The command verifies five optimizer updates, two heterogeneous accumulated
microbatches per update, a step-three checkpoint whose replay agrees with the
uninterrupted run within the same per-tensor tolerance the reference
comparison uses, real EVICT/FETCH traffic, numerical tolerances, and the
physical device cap.

A disagreement is reported in three categories, and the structural kind is
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
qualification checks setup, every update, and checkpoint replay. Executor always
enforces the total device allowance, including external memory; bounded pools,
sealed event inventories, zero steady-state allocation/event growth, and clean
allocator callbacks remain required.

The result JSON records numerical comparisons, physical evidence, phase timings,
per-step device/host elapsed times, and paths to the saved archives. Device elapsed
time includes transfers and dispatch gaps; it is not labeled compute-only time.
`--detailed-artifacts` additionally retains bounded runtime traces for each step.
Checkpoint writing and restoration stream Executor-owned tensors in bounded
chunks instead of creating a second full-state device image.

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
The `matrix/` package separates selection/arguments, worker commands, execution
and reporting. Remote qualification calls the same `matrix.run.main` entry.

The numerical and performance launchers accept `--external-headroom-mib`
(default **512 MiB on every GPU**). This allowance is subtracted, together with
the initial process baseline, when sizing the execution pool. Zero reserves no
external allowance. It does not change model or reference identity.

Whole-process and external memory overruns fail qualification. The obsolete
`--reject-overbudget`/`--no-reject-overbudget` switches are removed; the execution
API always enforces its admitted budget.

## Determinism and comparison criteria

The numerical gate runs every case with mlops's `deterministic_kernels`
in effect, which asks each operation that offers the choice for the kernel
whose accumulation order is fixed. Without it a kernel that sums with atomics
returns a slightly different answer each run, across most of a step's gradient
tensors, and no comparison against a reference or against a replay can mean
anything. The ordered kernels cost
throughput, so they are not the default outside this gate. The request covers
reference generation as well as the planned run, so a regenerated reference is
itself reproducible.

The request is baked into a compiled graph without a guard. The integrated gates
pass `--empty-caches`, giving each worker fresh Inductor/Triton caches removed
afterwards. A standalone matrix invocation uses the existing caches unless that
flag is set; do not reuse caches across deterministic/nondeterministic settings.

The planned arm also accumulates as the reference arm's PyTorch step does: a
later microbatch's weight gradient narrower than float32 is written at its
dtype by its multiply and added after it
(`fuse_narrow_gradient_accumulation=False`), where ShadowSpill by default adds
it inside the multiply. Float32 gradients are added inside the multiply either
way.

Against the reference, every weight must agree within a relative L2 of
2.5 % (cosine at least 0.999, sign agreement at least 99 %); an optimizer
moment gets 5 %, because it is an accumulator whose reduction order follows the
plan, so the same arithmetic in two orders moves it further than it moves a
weight. A weight that is still nothing but its optimizer steps -- every
element of the reference within `steps` learning rates of zero, as a bias
started at zero is after a few steps -- has no scale of its own for a relative
bound to measure, so it is held to an absolute one instead: no element may be
further from the reference than two learning rates, which is one step whose
gradient sign the two runs' roundings disagreed on. The gate also requires a checkpoint replay to agree with the
uninterrupted run within tolerance, and records whether it agreed bit for bit
besides.

Use the [nondeterminism probe](../nondeterminism/README.md) to locate a divergence.
