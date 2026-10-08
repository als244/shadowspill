# DP4 small-model bias corruption

## Reproduction and cause

The four-rank test uses two linear layers, sharded `torch.optim.AdamW`, FP32
parameters/gradients/moments, and a nine-element first bias. Padding makes twelve
elements, with three elements owned by each rank. The first update corrupted
every bias element by up to **3.3567963**, identically across replicas.

The direct task compiler passed a graph containing intermediate mutations to
Inductor without first functionalizing it. Inductor's post-gradient passes
explicitly assume normalized, functionalized IR. Two optimizations then became
unsafe: clone elimination could expose an input to an internal mutation, and
buffer reuse could overwrite a still-live alias returned by a fallback mutation.

In the saved generated optimizer code, `buf26` is the result of
`addcdiv_(view(buf2), ...)`. It aliases `buf2`. The code then assigns
`buf34 = buf2; del buf2` and writes the next parameter's update denominator into
that storage, before the later `all_gather_into_tensor(buf26, ...)` reads the
bias shard. Rank 0's corrupt bias values match that scratch calculation.

The generated file retained on Della is:

```
evidence/fatnode/dp4/auto/rank-00000/artifacts/v1/build/inductor/default-37a8eec1ce19/si/csizfblir6twkusj4l75jqvsyeaoqmgxhfzdd5zjj4rat7zfxf4b.py
```

Relevant statements are around lines 1131, 1150, and 1209. The padding and shard
sizes made scratch-buffer reuse expose this issue in DP4. Earlier successful
DP2 or large-model checks did not establish correctness of this compiler path.

## Evidence

- `logs/fatnode/dp4/bias-probe.log`: exact failing run with CPU snapshots of
  parameters and optimizer state. All four ranks reproduce the same bias error.
- Snapshots under `evidence/fatnode/dp4/bias-20261008T054017/` on fatnode:
  moments match the CPU oracle within 4.7e-10; the other parameters match within
  3e-8. Gradient accumulation and Adam's moment calculations therefore completed
  correctly in this run.
- `bias-no-metrics.log`: disabling parameter metrics reproduces the same error.
- `alias-repro-valid.log`: a small ordinary tensor operation reproduces the
  incorrect result through direct task compilation, with no runtime pool,
  training loop, or collectives. It also reveals an unintended input mutation.
  The initial ordinary-compile comparison reused an input that the faulty arm
  had modified; later runs use separate input copies and log their mutations.
- `alias-normalized.log`: an intermediate-alias rewrite alone fixed the output
  but still mutated a cloned input. That prototype was discarded.
- `alias-functionalized.log`: functionalization fixes the output and preserves
  all original inputs exactly; ordinary `torch.compile` agrees.
- Public `torch.func.functionalize` rejects opaque mutable operators, including
  `shadowspill_training::sum_gradient_`. This prototype was replaced with the
  compiler's Python functionalization mode, which supports those operators.
- `bias-python-functionalized.log`: all four ranks pass three updates,
  checkpoint policies/replay, fresh restore, and evaluation. Maximum parameter
  error is **2.9802322e-8**.

All GPU checks run in fatnode `codex:0.0`, in the container exposing only healthy
devices. The one-off scripts are `scripts/fatnode/bias_probe.py` and
`scripts/fatnode/alias_repro.py`. Binary snapshots remain out of Git.

## Generic compiler correction

`compilation/inductor/functional.py` normalizes intermediate mutations through
PyTorch's `FunctionalTensorMode` before decomposition and Inductor passes.
The task retains its calling convention: terminal copies publish actual input
updates. Opaque mutable operators go through PyTorch's automatic functionalization.
No optimizer, model, parameter name, tensor size, or process count is special-cased.

Aliased inputs use one shared base when functionalized, so a write through one
view is visible through another. Its extent covers the declared input views;
it does not include unused allocation capacity. The normalization occurs during
compilation and adds no Python normalization to the execution hot path.

Regression tests exercise temporary-buffer reuse, absence of unintended input
mutation, overlapping input views, and opaque mutable operations. All 34 tests
in the compiler directory pass on the healthy GPU container. Ruff and the
production module's strict mypy check pass.

## Distributed validation

Each row passed on all four ranks, including three optimizer updates against a
CPU combined-batch oracle, both checkpoint policies, replay, fresh restoration,
and forward evaluation. The independent-planning control uses unequal local
batch sizes; each run has its own matching CPU oracle.

| Configuration | Maximum parameter error | Fatnode evidence directory |
|---|---:|---|
| Save, verified symmetric planning | 2.9802322e-8 | `evidence/fatnode/dp4/bias-20261008T055612/` |
| Recompute, verified symmetric planning | 2.9802322e-8 | `evidence/fatnode/dp4/bias-20261008T060558/` |
| Save, independent planning | 3.7252903e-8 | `evidence/fatnode/dp4/bias-20261008T060642/` |

Compact per-rank results are in `evidence/bias_validation.json`. Full stores and
snapshots remain on fatnode; console logs were also copied to Della.

The first full-suite attempt could not collect repository tests because the
minimal container lacked `git`. The test launcher now mounts host Git tools
read-only, plus `nvidia-smi` for gate telemetry. This required no production
changes. The complete suite rerun passed in fatnode `codex:0.0`: **1,167 passed,
1 skipped, 38 deselected**, with all 20 CPU and 51 GPU CTest canaries passing.
The deselected fresh-process tests are run separately by CTest. The gate took
22 minutes on this machine. `evidence/bias_suite.json` summarizes the checks;
`logs/fatnode/suite-functionalized-rerun.log` retains the complete console output.
