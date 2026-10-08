# Longer 1.18B Llama timing checks

Same BF16 base, rank-32 FP32 factors, FP32 gradients/moments and 2048-token batches as the scale comparison. At least 10 exact-step warmups and 2 seconds at LR=0, then 40 measured updates. GC stays enabled; a callback records its timestamps. No artificial GC disabling or cache release is used.

| Case | Median ms | Mean ms | p95 ms | Range ms | Aggregate tok/s | Max GC during steps ms | Max warmup GC ms | Host RSS GiB |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| lora-recompute | 122.18 | 122.27 | 122.83 | 121.77–123.72 | 16,750 | 0.081 | 0.06 | 10.19 |
| lora-save | 99.60 | 99.62 | 99.92 | 99.03–100.78 | 20,558 | 0.082 | 231.85 | 10.18 |
| lora_head-recompute | 125.33 | 125.41 | 125.79 | 124.87–127.66 | 16,330 | 0.064 | 0.07 | 10.32 |
| lora_head-save | 101.39 | 101.42 | 101.67 | 100.80–103.48 | 20,194 | 0.084 | 0.07 | 10.33 |

The earlier 10-step runs remain in `evidence/llama-1b`; these follow-ups do not overwrite them. A long generation-2 collection during warmup, followed by stable measured steps, supports compilation-related garbage collection as the likely source of the earlier isolated stalls. The original runs did not record GC timestamps, so that attribution is an inference.

`lora` freezes the output head. `lora_head` also trains its low-rank factors; it still freezes the large original head matrix. Each case retains complete graphpairs and optimizer-allocation audits.
