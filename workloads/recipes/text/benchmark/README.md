# Catalog benchmark factory

This package supplies ordinary text models to the generic phase quickstart.
Model choices stay here; the runner builds, profiles, calibrates communication,
plans on CPU and executes saved Plans in fresh workers.

```bash
python -m benchmarking.quickstart \
  --factory workloads.recipes.text.benchmark.factory:experiment \
  --factory-args '{"model":"llama3-8b","sequence_length":1024,"sequences_per_step":64,"min_tokens_per_example":8192,"max_tokens_per_example":16384}' \
  --search-budget-gib 20 30 --run-budget-gib 20 30 --spill-gib 112 \
  --output-dir runs/llama3
```

The ordinary Llama3, Qwen3, Qwen3.5, OLMoE, Qwen3MoE and Qwen3.5MoE catalog
entries share this factory. `model_config` overrides the family's configuration;
`implementation` selects its supported `mlops` or `pytorch` implementation.
GLM/checkpoint and model-owned EP factories remain separate migration work.
Provider tuning portability still needs qualification beyond the tested paths.

Factory options:

| Option | Meaning |
| --- | --- |
| `sequence_length`, `sequences_per_step` | Rank-local update geometry; defaults come from the catalog. |
| `sequences_per_example` | Fix one candidate batch size; otherwise enumerate divisors of the step total. |
| `min_tokens_per_example`, `max_tokens_per_example` | Optional candidate bounds. |
| `precision` | Dictionary of `Precision` fields, including model/master/gradient/moment dtypes and rounding. Defaults preserve the catalog's BF16 training policy. |
| `lora` | Omit or use `null` for full training; `{}` enables the family's predefined targets with rank/alpha 32. Other keys match `configure_lora`, including explicit targets/head/shared-expert controls. |
| `ranks`, `num_replicas` | Ranks in one complete-model replica and number of replicas. |
| `weights`, `gradients`, `optimizer` | Storage modes, defaulting to `"sharded"` within each replica. |
| `seed` | Parameter/input seed. Each logical rank has distinct data, reused across candidates and budgets. |

For example, add `"ranks":8` for eight-way sharded DP, or
`"precision":{"master_dtype":"float32","grad_dtype":"float32","opt_state_dtype":"float32"}`
for FP32 training state. Metadata records the resolved settings and global token
count. Batch orderings are enumerated during CPU planning using the saved Program.

`initialize.py` fills Executor-provided CPU storage in deterministic 256K-element
RNG blocks. Scratch stays bounded, shard/chunk boundaries do not change values,
and no complete anonymous model is allocated. It preserves family initialization
distributions and derived RoPE/norm/gate values, not the original module traversal's
RNG stream. Optimizer state and master copies use the Program's initialization.

Tiny Llama full training and LoRA are tested through every phase with FP32
masters/gradients/moments. Initializer coverage, shape/stride consistency and
global loss normalization are also checked on CPU for all six families. These
checks do not establish large-model performance or every provider's portability.

See the [phase quickstart guide](../../../../benchmarking/quickstart.md) for
independent phase commands, custom communication costs and result locations.
