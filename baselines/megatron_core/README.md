# Megatron-Core baseline

NVIDIA provides an official Qwen3-30B-A3B model and an eight-H100 pretraining
recipe through **Megatron Bridge**, backed by Megatron-Core. This is a separate
baseline from NeMo AutoModel's FSDP2 implementation.

The initial source pins are:

- Megatron Bridge: `6dea15d7fd499975ebf91fbeaf8c2b740a73d679`
- Its Megatron-Core submodule: `ce852bfb93f364f64a556602fc71a5d9ed2c377e`
- Recipe: `qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config`
- Model config revision: `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`

The [official recipe source](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/6dea15d7fd499975ebf91fbeaf8c2b740a73d679/src/megatron/bridge/recipes/qwen/h100/qwen3_moe.py)
constructs the model with `load_weights=False`; it does not require downloading
the model checkpoint. Its initial parallelism is TP4 / PP2 / EP4, expert TP1.
It uses BF16 compute, FP32 main parameters/gradients/Adam moments, full-layer
recomputation, Transformer Engine and DeepEP.

The comparison will retain 4,096-token sequences and 2,097,152 global tokens
per update. Here DP size depends on TP/PP/CP: do not apply the native FSDP2
accumulation formula using all eight ranks when TP or PP is active. Record the
resolved topology, gradient accumulation and recomputation with each result.

Prepare an explicit command against the pinned checkout:

```bash
python -m baselines.megatron_core.prepare \
  --bridge-root /path/to/pinned/Megatron-Bridge \
  --python /path/to/megatron/environment/bin/python \
  --output runs/baselines/megatron-core/official-8gpu \
  --microbatch-tokens 4096 --sequence-length 4096 --global-tokens 2097152 \
  --warmup-steps 3 --steps 5
```

The output contains `run.sh`, the exact command, and a request manifest. It
invokes NVIDIA's own training recipe launcher, with mock data and checkpoint
loading/saving disabled. LR/betas/epsilon/weight decay are matched to the MLOps
baseline; LR warmup and clipping are disabled. This is distinct from measurement
warmup: exclude the first `warmup_steps` complete updates in throughput analysis.

`--tensor-parallel`, `--pipeline-parallel`, `--expert-parallel`, and
`--expert-tensor-parallel` expose the topology. The generator validates group
divisibility and computes accumulation from the actual attention DP size.
`--no-recompute` disables the recipe's full-block recomputation. The resolved
upstream configuration must still be validated before accepting GPU results.

Source support and command generation have been verified. Environment
installation and GPU qualification are pending; there are no performance
results for this baseline yet. External checkouts/environments/results belong
under `runs/`, outside tracked source.
