# Full-model performance qualification

Each worker runs one large model entirely through ShadowSpill. It does not
construct a standard-allocator model, which would exceed the target physical
device budget.

The worker initializes, registers, and calibrates the runtime pools before it
constructs the model. It prints the exact solo and bidirectional-concurrent
route bandwidths before planning and persists that same runtime snapshot in
the result artifact. Model state is then constructed, imported into the spill
pool, and released from anonymous CPU storage.

The retained geometries all use sequence length 1,024 and 65,536 tokens per
optimizer step:

| Family | Tokens/microbatch | Accumulation |
|---|---:|---:|
| Llama 3 8B | 8,192 | 8 |
| Qwen 3.5 9B | 16,384 | 4 |
| OLMoE 7B | 32,768 | 2 |

Run one cell with fresh planning artifacts:

```bash
python -m qualification.performance.run llama3 mlops \
  qualification/results/full_model/mlops_llama3.json \
  --force-fresh
```

For a throughput-only runtime probe on a host that cannot hold both the full
pinned spill arena and an anonymous checkpoint copy, add `--skip-checkpoint`.
The resulting artifact records that checkpoint qualification was skipped; the
default release gate still checkpoints and restores.

`--spill-budget-gib` changes the configured runtime spill-pool capacity.
`--planning-spill-budget-gib` may set a smaller budget for PressureFit without
shrinking that physical pool. The planning budget is rejected immediately if
it exceeds the configured capacity.

Run the complete five-cell matrix with:

```bash
python -m qualification.performance.matrix \
  --output-directory qualification/results/full_model \
  --force-fresh \
  --keep-going
```

The default protocol checkpoints the planned callable, performs and diagnoses
one warm step, restores the checkpoint, then measures three groups of four
steps. Planning, compilation, warmup, and restore are outside timed execution.
