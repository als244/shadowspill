# Full-model performance qualification

Each worker runs one large model entirely through ShadowSpill. It does not
construct a standard-allocator model, which would exceed the target physical
device budget.

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
