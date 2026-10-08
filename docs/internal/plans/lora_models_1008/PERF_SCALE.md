# LoRA at the default performance-gate model sizes

Completed normal-policy cases: **6/6**; additional controlled cases: **1**. Each case runs in a fresh process on Chicago's RTX 5090.

## Configuration

Unmodified MLOps Llama3, dense Qwen3.5 and OLMoE throughput presets. All updates contain 65,536 tokens at sequence length 1,024. Microbatches: Llama 8,192 tokens ×8; Qwen 16,384 ×4; OLMoE 32,768 ×2.

Both modes use a 16 GiB execution budget, 112 GiB pinned spill capacity, BF16 base weights/gradients/AdamW moments and no masters. LoRA uses rank/alpha 32, BF16 compute with FP32 factor storage; embedding, head, router, normalization and shared-expert base weights are frozen. Base weights and input batches are initialized identically before LoRA factors are added.

Each run uses two exact accumulated-step warmups at LR=0, then three groups of four measured updates. The table uses the median group throughput; timings include required terminal transfers using the performance gate's cycle measurements. The primary comparison lets the ordinary planner choose save/recompute per task. Any explicitly labeled save/recompute rows are separate controlled experiments.

## Step throughput

| Model | Variant | Full step s | LoRA step s | Full tok/s | LoRA tok/s | Speedup |
|---|---|---:|---:|---:|---:|---:|
| llama3 | auto | 18.429 | 13.489 | 3,556 | 4,859 | 1.37× |
| qwen35 | auto | 19.052 | 15.013 | 3,440 | 4,365 | 1.27× |
| olmoe | auto | 4.760 | 3.877 | 13,767 | 16,903 | 1.23× |

## Why the first 22.2-second Llama result differed from the gate

The initial controlled run forced all 272 task choices to save and measured 22.225 s/update. The recent gate result used 144 save and 128 recompute choices and measured 17.980 s (18.507 s simulated). Forcing save increased planned fetch traffic from 375.69 to 523.69 GiB and eviction traffic from 181.11 to 330.95 GiB per update. The completed forced-save result is retained as evidence, but is not used as the normal-policy baseline.

For a large linear projection, full training computes forward, input gradient and weight gradient; a frozen base weight omits the weight-gradient GEMM. LoRA therefore approaches two-thirds of the projection FLOPs plus the low-rank products, before recomputation. Attention, activation kernels, optimizer work and transfers also contribute to total step time.

## Memory, model sizes and measurement variation

Both modes reserve the same 112 GiB pinned pool. Consequently RSS is not a measure of their differing live tensor requirements here; the actual peak spill allocation is recorded separately. Host RSS is the process high-water mark through preparation and execution. Per-case host-memory.json additionally samples process-tree RSS/PSS including compiler children.

| Model | Mode | Variant | Total params B | Trainable M | Host RSS GiB | Spill peak GiB | Group step range s | Graphpairs |
|---|---|---|---:|---:|---:|---:|---|---|
| llama3 | full | auto | 8.030 | 8030.26 | 128.46 | 59.03 | 18.329–18.475 | [table](evidence/performance-gate-scale/llama3-full-auto/graphpairs.md) |
| llama3 | full | save | 8.030 | 8030.26 | 128.46 | 84.29 | 22.202–22.238 | [table](evidence/performance-gate-scale/llama3-full-save/graphpairs.md) |
| llama3 | lora | auto | 8.114 | 83.89 | 128.78 | 19.62 | 13.411–13.518 | [table](evidence/performance-gate-scale/llama3-lora-auto/graphpairs.md) |
| olmoe | full | auto | 6.919 | 6919.16 | 126.39 | 51.42 | 4.759–4.775 | [table](evidence/performance-gate-scale/olmoe-full-auto/graphpairs.md) |
| olmoe | lora | auto | 7.162 | 243.27 | 127.30 | 26.95 | 3.876–3.879 | [table](evidence/performance-gate-scale/olmoe-lora-auto/graphpairs.md) |
| qwen35 | full | auto | 8.954 | 8953.80 | 130.12 | 91.41 | 19.030–19.086 | [table](evidence/performance-gate-scale/qwen35-full-auto/graphpairs.md) |
| qwen35 | lora | auto | 9.034 | 80.27 | 130.42 | 39.92 | 14.959–15.033 | [table](evidence/performance-gate-scale/qwen35-lora-auto/graphpairs.md) |

All per-task input, mutation, output and workspace sizes and runtimes are also available in [perf-scale-graphpairs.csv](perf-scale-graphpairs.csv). These are per-task allocation totals, not simultaneous whole-model peaks.

## Transfer traffic and selected recomputation

| Model | Mode | Recompute choices | Fetch GiB/step | Evict GiB/step | New device allocations during measurement |
|---|---|---:|---:|---:|---:|
| llama3 | full | 192/272 | 330.50 | 133.43 | 0 |
| llama3 | lora | 128/264 | 194.14 | 31.91 | 0 |
| olmoe | full | 32/36 | 80.40 | 51.29 | 0 |
| olmoe | lora | 24/34 | 73.08 | 25.80 | 0 |
| qwen35 | full | 64/136 | 287.22 | 187.00 | 0 |
| qwen35 | lora | 64/132 | 195.34 | 87.66 | 0 |

## Reproduction

```bash
bash docs/internal/plans/lora_models_1008/scripts/run_perf_scale.sh
```

The runner resumes completed compatible cases and saves progress after each case. `scripts/perf_scale_lora.py` accepts CLI settings or a JSON configuration file. Each case stores the full manifest, parameter inventory, optimizer audit, graphpairs, build artifacts, detailed timings and console log. These are synthetic-token throughput measurements, not fine-tuning-quality experiments.

[Final validation](perf-scale-validation.json) records all six optimizer/physical audits and the absence of new device allocations or event-pool growth during measurement. No production code was changed for this follow-up, and no further LoRA optimization is included, as requested.

Primary evidence and full build stores live on Chicago under `/home/shein/Documents/grad_school/research/shadowspill/docs/internal/plans/lora_models_1008/`. Reports, logs and small case evidence are mirrored on Della under the same relative plan directory; full build stores remain on Chicago.
