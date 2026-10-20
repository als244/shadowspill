# NeMo AutoModel baseline

This uses the user's [pinned NVIDIA example](https://github.com/NVIDIA-NeMo/Automodel/blob/b203f3f9f9f3140c660942a088ff36bbc3a5e856/examples/llm_benchmark/qwen/qwen3_moe_30b_te_deepep.yaml):
FSDP2 + EP8, Transformer Engine attention/linear/norm, DeepEP dispatch and
`torch_mm` experts. It has no ShadowSpill dependency. GPU validation and
performance measurements are pending.

The original YAML and its license remain unchanged in `../references/`.
Prepare an isolated derived configuration:

```bash
python -m baselines.automodel.prepare \
  --output runs/baselines/automodel/16K-real \
  --microbatch-tokens 16384 --routing real --recompute \
  --global-tokens 2097152 --sequence-length 4096 \
  --warmup-steps 3 --steps 5
```

Then, in a separate AutoModel environment installed from the pinned revision:

```bash
automodel runs/baselines/automodel/16K-real/config.yaml --nproc-per-node 8
```

Use `--routing balanced` to reproduce artificial balanced routing; it must be
reported separately from real routing. `--no-recompute` disables checkpointing.
`--cpu-offload` sets FSDP2's `CPUOffloadPolicy` as the recipe's
`distributed.offload_policy`, which AutoModel's parallelizer hands to every
`fully_shard` call: parameter shards, gradients and optimizer state on the
host, pinned. AutoModel offloads no activations.
`--microbatch-tokens` is per rank; accumulation is derived and checked against
the fixed global update. Every generated case records its source and changes.

The derived recipe requests FP32 resident parameter shards, gradients and AdamW
moments, with BF16 compute. It uses the matched optimizer settings from the
MLOps comparison. The upstream example otherwise leaves storage dtype implicit
and uses Adam with different hyperparameters; the original remains available
for an unmodified upstream reproduction. Weight loading is from configuration,
not a pretrained checkpoint; model configuration metadata may be downloaded.

Use an isolated environment for AutoModel's pinned dependencies. Do not upgrade
the environment used to collect the existing ShadowSpill reports. Preserve the
resolved config, environment, stdout, all-rank timings, memory and precision
evidence before accepting a result. Preparation alone is not GPU qualification.
