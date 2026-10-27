# Expert-parallel text phases

This recipe reuses the common text inputs, objective, optimizer and bounded
initialization. Qwen3 MoE, Qwen3.5 MoE and OLMoE supply portable Quack expert
leaves; `Resource` descriptions recreate their shared MoonEP buffer in build,
profiling and execution. Models contain no ShadowSpill imports or live groups.

```bash
python -m benchmarking.quickstart \
  --factory workloads.recipes.text.expert_parallel:experiment \
  --factory-args '{"model":"qwen3moe-30b-a3b","ranks":8,"sequence_length":4096,"sequences_per_step":64,"sequences_per_example":16}' \
  --search-budget-gib 30 50 77 --spill-gib 96 --steps 3 \
  --output-dir runs/qwen-ep8
```

Launch once: quickstart creates the distributed workers. Each candidate has one
exact token geometry because MoonEP buffers have fixed capacity. Build/profile
need all participants for this model; ordinary DP preparation remains local.
CPU planning consumes the resulting archives without opening resources.

Expert parameters are contiguous local expert subsets. Other parameters are
replicas with sharded optimizer data; their gradients are summed once before
updates. Input data differs across logical ranks, while matching dense tensors
start identically. Loss normalization and throughput use the global token count.
Masters, gradients and moments default to FP32; compute weights are BF16.

The provider currently requires SM90 and fixed GEMM policies. Use the MLOps
portable-resource implementation (`codex/offline-pipeline-resources`) until that
dependency merges. Local metadata/construction tests pass; saved EP transport
and numerical qualification on two H100s remain pending. Run:

```bash
python -m pytest tests/shadowspill/execution/expert_parallel/test_training.py -o addopts= -q
```

That test covers both save/recompute choices, four examples per invocation,
reversed logical ranks, three updates, master/moment comparisons and checkpoint
restore. Its eager reference shares expert kernels but uses ordinary PyTorch
accumulation and FP32 AdamW. Logs and artifacts are separate for each phase.
