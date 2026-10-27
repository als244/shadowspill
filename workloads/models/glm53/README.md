# GLM-5.3

This workload composes the normal GLM-5.3 text architecture: 78 decoder layers,
three initial dense MLPs, then 256 routed experts/top-8 plus one shared expert.
It uses ordinary residual connections and positional sparse MLA. Full indexers
select causal keys in layers 0, 1, 2, 6, 10, and so on; intervening layers reuse
those indices. Packed sequences reset RoPE positions and never attend across
sequence boundaries. This architecture is distinct from GLM-5.3-Flash's KDA,
NoPE attention and mHC streams.

The workload uses the general MLOps `sparse_latent_attention` operation with
separate 64-dimensional positional query/key tails. Its indexer uses
`pooled_topk` with pool size one. SonicMoE handles the local experts; no expert
parallel group is created.

## Training and precision

The [shared GLM recipe](../../recipes/text/glm/factory.py) defaults to rank-32 BF16 LoRA factors,
alpha 32, FP32 accumulated gradients, and BF16 AdamW moments with stochastic
rounding. The importer preserves the checkpoint's block-FP8 expert data and
scales; expert GEMMs run in FP8. Compressed dense projections keep their stored
weights and use BF16 GEMMs (`gemm_precision="bf16"`). The router, shared experts,
indexer, normalization and base weights stay frozen. The output head is frozen
unless `lora_head=true`.

The shared checkpoint reader supports `copy_chunk(name, shape, offset, destination)`
after meta-model declaration, matching the bounded loading interface used by
other families. See [checkpoint loading](../glm53_flash/README.md#checkpoints-and-execution)
for byte preservation and the scope of this direct-copy path.

The normal model's quickstart preserves intermediate BF16 casts during
compilation. Removing those casts can change discrete top-k attention choices.
Build finalizes task code under this setting. Independent profiling and execution
load that saved code without rebuilding the model.

For eight ranks, 4K sequences, 2M global tokens per step, and an 8K-token
microbatch per rank:

```bash
python -m benchmarking.quickstart \
  --factory workloads.recipes.text.glm.quickstart:glm53 \
  --factory-args '{"checkpoint":"/local/models/GLM-5.3","expert_storage":"fp8_block","ranks":8,"sequence_length":4096,"sequences_per_step":64,"sequences_per_example":[2]}' \
  --search-budget-gib 70 --run-budget-gib 70 --spill-gib "$SPILL_GIB" \
  --steps 7 --output-dir /local/results/glm53-dp-sharded
```

Set `SPILL_GIB` from the checkpoint inventory plus training/activation storage;
the Flash model's capacity requirement does not apply to this larger model.
Host memory must cover all eight spill pools plus initialization and compiler
scratch. Set `sequences_per_example` to `[2,4,8,16]` to search the requested
8K/16K/32K/64K geometries together, or run each separately for independent
throughput measurements. `sequences_per_step` is per rank. The objective divides
by the global token count once; data-parallel reductions sum contributions.
The shared runner builds and profiles compute on one GPU, then launches all
ranks for communication calibration and execution.

Weights, gradients and optimizer data use the common logical parallelism
specification. See the [phase API](../../../docs/architecture/offline-pipeline.md) and the
[shared recipe arguments](../glm53_flash/README.md#quickstart-configuration).
`tiny=true` selects a small random fixture explicitly; `layer_limit` selects a
checkpoint prefix. Omit both for all 78 layers.

## Qualification

Tests compare eager and compiled packed attention outputs, selected keys and
input gradients against the Transformers GLM reference. Separate tests cover
positional sparse-attention gradients, shared-index layers, the complete training
composition, and exact synthetic checkpoint import. Full-checkpoint distributed
training throughput and numerical qualification are still in progress; the
example above is a configuration, not a completed benchmark result.
