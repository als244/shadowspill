# LoRA output-head loss

## Implemented on Chicago

The MLOps checkout at `~/Documents/grad_school/research/mlops` now exports
`mlops.lora_head_loss` beside `mlops.head_loss`. Changes are uncommitted.
The existing ordinary head-loss implementation is unchanged.

```python
loss = mlops.lora_head_loss(
    hidden, head_weight, lora_a, lora_b, targets,
    scale=alpha / rank, chunk_size=chunk_size, reduction="sum",
)
```

The operation computes `X W.T + scale * (X A.T) B.T` one row chunk at a time.
Set `head_weight.requires_grad_(False)` to omit its gradient. Base weights,
factors and hidden remain explicit tensor inputs; no model-specific lowering
rules, optimizer ownership, or dense `B @ A` update is introduced.

The chunked forward computes requested unit-cotangent VJPs. Backward scales
these seeds without modifying them. Factor storage dtype may differ from the
hidden compute dtype, and the existing weight-gradient precision setting applies.
There is no dropout argument or higher-order derivative support.

The low-rank contribution uses an in-place addmm into chunk-local logits,
avoiding a second logits-sized low-rank product. The shared matrix accumulation
helper gains optional `alpha=1.0` scaling for factor gradients.

## Validation

All checks ran after both baseline qualification runs completed.

- 74 CPU tests passed on the installed checkout: new-head math/trainability,
  ordinary-head regressions, fake/autograd contracts, logical costs, dispatch,
  and documentation. Six hardware-dependent checks skipped in this CPU run.
- Six GPU tests passed: FP32/FP16/BF16, eager and fullgraph Inductor. Numerical
  checks include relative L2 gradient error, not only absolute tolerances.
- Seven GPU contract checks passed, including ordinary head regression cases.
- ShadowSpill save and recompute each completed three SGD updates, using a
  frozen head, nonzero A/B, and a trainable input projection. Losses and updated
  parameters matched an independent PyTorch composition; frozen head bytes were
  unchanged. Both variants were explicitly selected and executed.
- No ShadowSpill compiler/planner changes were needed.

In the isolated CPU save-state test, seeds contain only hidden/A/B:
`(70 + 21 + 57) * 4 = 592` bytes. The `19 * 7 * 4 = 532` byte base gradient
is absent. No full-batch logits are retained, and backward does not repeat the
projection in the save variant.

## Graphpair evidence

The training probe uses 67 rows, width 128, vocabulary 1024, rank 16, FP32.
It includes an extra trainable 128-by-128 input projection to test propagation
through the head. These are correctness-test timings with conditioning off,
not representative model benchmarks.

See:

- `evidence/head-shadowspill-save/graphpairs.md` and `graphpairs.json`
- `evidence/head-shadowspill-recompute/graphpairs.md` and `graphpairs.json`
- Each directory's `result.json` and fresh `artifacts/` store
- `logs/lora-head-gpu.log`, `logs/lora-head-contracts-gpu.log`
- `logs/installed-head-cpu.log`
- `evidence/installed-lora-head-files.json` for installed-source SHA-256 hashes

Saved forward output bytes are 108,036: hidden-gradient seed 34,304,
A-gradient seed 8,192, B-gradient seed 65,536, and scalar loss 4.
The 524,288-byte base-head gradient is absent. Backward output is 139,264 bytes:
the input-projection gradient 65,536 plus A/B gradients 73,728.

## Remaining whole-model work

Automatic target selection/head-module conversion and dedicated expert LoRA
integration remain on the agenda. Existing MLOps workload logits/loss paths
read `lm_head.weight`; wrapping only its forward would miss that loss path.
The new operation is ready for the head module to call. The model catalog labels
whole-model conversion as in progress; it does not advertise a working
`--lora` flag.

In LoRA mode the head will default to frozen, with optional LoRA or full
training by name. Embedding full training remains optional. The normal
full-training model behavior is unchanged.
