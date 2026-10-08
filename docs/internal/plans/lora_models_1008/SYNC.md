# LoRA/model release — 2026-10-08

The user approved committing and pushing this work and updating the existing master checkouts on all four development machines.

## Source commits

### MLOps

```text
1b8cec6 Add configurable LoRA modules and omit frozen head and expert gradients
791a44a Respect precision and device limits in grouped expert GEMMs
27b9c95 Keep MoE auxiliary routing counts in FP32 for every default dtype
```

### ShadowSpill

```text
f49f6f3f Organize example architectures and expose full-model LoRA recipes
371dee83 Account for empty tensor views of nonempty allocations
0c7ee75d Materialize shared activation cotangents at task boundaries
```

The following documentation commit stores this report, compact validation results and reproduction scripts. Large compiler/build stores, allocation timelines and tensor checkpoints remain local.

## Checkouts

| Machine | MLOps | ShadowSpill |
|---|---|---|
| Chicago | ~/Documents/grad_school/research/mlops | ~/Documents/grad_school/research/shadowspill |
| Della | ~/mlops | ~/shadowspill |
| Tübingen | ~/Documents/mlops | ~/Documents/shadowspill |
| fatnode | ~/mlops | ~/shadowspill |

Synchronization uses ordinary fast-forward pulls of master, followed by import and CPU LoRA loss/gradient checks in each machine's shadowspill environment. This release changes Python source and documentation; it has no compiled-extension or dependency changes. Existing editable installations point to these checkouts. No new branch or worktree is required. Existing unrelated gate configs, report symlinks and old plan drafts stay local.

## Validation before release

- ShadowSpill suite: 1,238 passed, one skipped; both CTest groups passed.
- Numerical gate: five cases passed against existing references.
- Whole-model LoRA: 32 GPU/reference cases across eight implementations, FP32/BF16 and save/recompute.
- Benchmarking: 28 earlier full-model performance cases and six matched default-model full/LoRA comparisons, all passing optimizer allocation audits.

See [RESULTS.md](RESULTS.md) and [PERF_SCALE.md](PERF_SCALE.md) for scope, model dimensions, timings and memory measurements. Multi-GPU validation of the consolidated optional-EP constructors and full-model FP8 LoRA remain separate follow-up work.

## Completed synchronization

Both repositories were pushed to origin/master and fast-forwarded on all four machines. MLOps is at 1b8cec6; ShadowSpill source and evidence are at 415a551a. This final synchronization note is a documentation-only follow-up.

| Machine | Editable imports resolve to the intended checkouts | CPU LoRA checks |
|---|---|---|
| Chicago | Yes | 16 passed |
| Della | Yes | 16 passed |
| Tübingen | Yes | 16 passed |
| fatnode | Yes | 16 passed |

Each check covers losses, gradients, three parameter updates, frozen-state preservation and state reload across eight implementations, with frozen or LoRA output heads. All ran on CPU in the existing shadowspill environment. Full GPU gates were not repeated during synchronization; their pre-release results are above. Each checkout remains on master with one worktree.

Per-machine logs and machine-readable summaries are stored locally under logs/post-pull-cpu-20261008.{log,json}.
