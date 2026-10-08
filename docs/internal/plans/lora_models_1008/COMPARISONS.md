# Full-model LoRA comparisons

BF16 base/compute; rank 32, alpha 32, FP32 factors/gradients/AdamW moments; no masters.
4-layer reduced-dimension models, width 768, vocabulary 32768, 2048 tokens, sequence length 512.
MoE: 32 experts, top-4, expert hidden width 512. These are architecture-preserving benchmarks, not published 30B/35B model sizes.
At least 5 exact-task warmups and 1 second at lr=0; 10 measured training updates. Fresh per-case artifact stores.

| Architecture | Mode | Variant | Trainable M | Step ms | tok/s | Peak process RSS GiB | Peak tree PSS GiB | Spill reserved / peak allocated GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llama3 | full | recompute | 77.864 | 33.91 | 60,388 | 6.22 | 6.44 | 4 / 0.83 |
| llama3 | full | save | 77.864 | 36.08 | 56,762 | 6.22 | 6.44 | 4 / 0.83 |
| llama3 | lora | recompute | 1.835 | 14.58 | 140,447 | 5.02 | 5.30 | 3 / 0.50 |
| llama3 | lora | save | 1.835 | 13.55 | 151,195 | 5.01 | 5.30 | 3 / 0.50 |
| olmoe | full | recompute | 207.727 | 87.33 | 23,453 | 8.78 | 9.02 | 6 / 2.03 |
| olmoe | full | save | 207.727 | 90.85 | 22,542 | 8.62 | 8.77 | 6 / 2.03 |
| olmoe | lora | recompute | 13.238 | 26.29 | 77,896 | 5.53 | 5.85 | 3 / 0.60 |
| olmoe | lora | save | 13.238 | 23.07 | 88,769 | 5.53 | 5.85 | 3 / 0.60 |
| qwen35 | full | recompute | 78.503 | 35.36 | 57,923 | 6.57 | 6.93 | 4 / 0.83 |
| qwen35 | full | save | 78.503 | 36.15 | 56,646 | 6.47 | 6.79 | 4 / 0.83 |
| qwen35 | lora | recompute | 1.787 | 21.27 | 96,267 | 5.33 | 5.66 | 3 / 0.50 |
| qwen35 | lora | save | 1.787 | 19.52 | 104,937 | 5.32 | 5.66 | 3 / 0.50 |
| qwen35moe | full | recompute | 213.084 | 90.63 | 22,598 | 9.24 | 9.58 | 6 / 2.08 |
| qwen35moe | full | save | 213.084 | 91.45 | 22,394 | 9.25 | 9.59 | 6 / 2.08 |
| qwen35moe | lora | recompute | 13.191 | 29.83 | 68,663 | 6.08 | 6.47 | 3 / 0.62 |
| qwen35moe | lora | save | 13.191 | 25.84 | 79,243 | 6.08 | 6.44 | 3 / 0.62 |
| qwen3moe | full | recompute | 207.724 | 87.92 | 23,293 | 8.78 | 9.03 | 6 / 2.03 |
| qwen3moe | full | save | 207.724 | 91.14 | 22,471 | 8.78 | 9.04 | 6 / 2.03 |
| qwen3moe | lora | recompute | 13.238 | 26.54 | 77,170 | 5.61 | 5.94 | 3 / 0.60 |
| qwen3moe | lora | save | 13.238 | 23.42 | 87,462 | 5.63 | 5.97 | 3 / 0.60 |

RSS is the case process peak, including its pinned pool; PSS is a 250 ms sampled process-tree peak that apportions shared pages and includes compiler children. These are different scopes.
Spill capacity follows the same tensor-count formula for each mode and is reserved up front. Report it separately from live allocation evidence; RSS reductions include smaller reservations.

## Individual graph pairs

All sizes are MiB. Total is input + mutated + output + workspace; totals across separate tasks are not a simultaneous model peak.
Occurrences indicate how often a structurally identical stage appears. Stage IDs are local to the case.

| Architecture | Mode | Variant | Stage × occurrences | Pass | Input | Mutated | Output | Workspace | Total | ms |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| llama3 | full | recompute | unique_stage_0000 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.05 |
| llama3 | full | recompute | unique_stage_0000 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| llama3 | full | recompute | unique_stage_0001 × 1 | forward | 51.02 | 0.00 | 0.00 | 361.00 | 412.02 | 1.81 |
| llama3 | full | recompute | unique_stage_0001 × 1 | backward | 51.02 | 0.00 | 99.00 | 265.01 | 415.03 | 1.98 |
| llama3 | full | recompute | unique_stage_0002 × 4 | forward | 16.39 | 0.00 | 3.00 | 27.00 | 46.39 | 0.35 |
| llama3 | full | recompute | unique_stage_0002 × 4 | backward | 19.39 | 0.00 | 29.26 | 62.11 | 110.76 | 0.87 |
| llama3 | full | save | unique_stage_0000 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.05 |
| llama3 | full | save | unique_stage_0000 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| llama3 | full | save | unique_stage_0001 × 1 | forward | 51.02 | 0.00 | 99.00 | 262.00 | 412.02 | 1.81 |
| llama3 | full | save | unique_stage_0001 × 1 | backward | 102.00 | 0.00 | 99.00 | 3.10 | 204.11 | 0.24 |
| llama3 | full | save | unique_stage_0002 × 4 | forward | 16.39 | 0.00 | 32.09 | 9.00 | 57.49 | 0.35 |
| llama3 | full | save | unique_stage_0002 × 4 | backward | 48.49 | 0.00 | 29.26 | 42.02 | 119.76 | 0.72 |
| llama3 | lora | recompute | unique_stage_0000 × 3 | forward | 18.14 | 0.00 | 3.00 | 30.00 | 51.14 | 0.48 |
| llama3 | lora | recompute | unique_stage_0000 × 3 | backward | 21.14 | 0.00 | 4.75 | 63.45 | 89.35 | 1.02 |
| llama3 | lora | recompute | unique_stage_0001 × 1 | forward | 63.14 | 0.00 | 3.02 | 30.00 | 96.16 | 0.51 |
| llama3 | lora | recompute | unique_stage_0001 × 1 | backward | 66.14 | 0.00 | 1.75 | 63.34 | 131.23 | 0.94 |
| llama3 | lora | recompute | unique_stage_0002 × 1 | forward | 51.02 | 0.00 | 0.00 | 265.00 | 316.02 | 1.26 |
| llama3 | lora | recompute | unique_stage_0002 × 1 | backward | 51.02 | 0.00 | 3.00 | 265.01 | 319.02 | 1.33 |
| llama3 | lora | save | unique_stage_0000 × 3 | forward | 18.14 | 0.00 | 33.84 | 9.00 | 60.99 | 0.48 |
| llama3 | lora | save | unique_stage_0000 × 3 | backward | 50.24 | 0.00 | 4.75 | 36.02 | 91.00 | 0.72 |
| llama3 | lora | save | unique_stage_0001 × 1 | forward | 63.14 | 0.00 | 36.72 | 9.00 | 108.86 | 0.52 |
| llama3 | lora | save | unique_stage_0001 × 1 | backward | 48.20 | 0.00 | 1.75 | 33.04 | 82.99 | 0.56 |
| llama3 | lora | save | unique_stage_0002 × 1 | forward | 51.02 | 0.00 | 3.00 | 262.00 | 316.02 | 1.26 |
| llama3 | lora | save | unique_stage_0002 × 1 | backward | 6.00 | 0.00 | 3.00 | 3.10 | 12.11 | 0.13 |
| olmoe | full | recompute | unique_stage_0000 × 3 | forward | 78.32 | 0.00 | 3.00 | 39.10 | 120.41 | 0.81 |
| olmoe | full | recompute | unique_stage_0000 × 3 | backward | 81.32 | 0.00 | 153.10 | 84.47 | 318.89 | 1.76 |
| olmoe | full | recompute | unique_stage_0001 × 1 | forward | 78.32 | 0.00 | 3.00 | 39.10 | 120.41 | 0.81 |
| olmoe | full | recompute | unique_stage_0001 × 1 | backward | 81.32 | 0.00 | 153.10 | 84.47 | 318.89 | 1.78 |
| olmoe | full | recompute | unique_stage_0002 × 1 | forward | 51.02 | 0.00 | 0.00 | 361.00 | 412.02 | 1.81 |
| olmoe | full | recompute | unique_stage_0002 × 1 | backward | 51.02 | 0.00 | 99.00 | 265.01 | 415.03 | 1.98 |
| olmoe | full | recompute | unique_stage_0003 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.06 |
| olmoe | full | recompute | unique_stage_0003 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| olmoe | full | save | unique_stage_0000 × 3 | forward | 78.32 | 0.00 | 30.34 | 20.00 | 128.66 | 0.80 |
| olmoe | full | save | unique_stage_0000 × 3 | backward | 108.66 | 0.00 | 153.10 | 50.29 | 312.05 | 1.27 |
| olmoe | full | save | unique_stage_0001 × 1 | forward | 78.32 | 0.00 | 30.34 | 20.00 | 128.66 | 0.80 |
| olmoe | full | save | unique_stage_0001 × 1 | backward | 108.66 | 0.00 | 153.10 | 50.29 | 312.05 | 1.25 |
| olmoe | full | save | unique_stage_0002 × 1 | forward | 51.02 | 0.00 | 99.00 | 262.00 | 412.02 | 1.80 |
| olmoe | full | save | unique_stage_0002 × 1 | backward | 102.00 | 0.00 | 99.00 | 3.10 | 204.11 | 0.24 |
| olmoe | full | save | unique_stage_0003 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.05 |
| olmoe | full | save | unique_stage_0003 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| olmoe | lora | recompute | unique_stage_0000 × 1 | forward | 135.94 | 0.00 | 3.02 | 43.22 | 182.18 | 1.18 |
| olmoe | lora | recompute | unique_stage_0000 × 1 | backward | 138.94 | 0.00 | 12.62 | 114.88 | 266.45 | 2.34 |
| olmoe | lora | recompute | unique_stage_0001 × 3 | forward | 90.94 | 0.00 | 3.00 | 43.22 | 137.16 | 1.14 |
| olmoe | lora | recompute | unique_stage_0001 × 3 | backward | 93.94 | 0.00 | 15.63 | 115.00 | 224.57 | 2.41 |
| olmoe | lora | recompute | unique_stage_0002 × 1 | forward | 51.02 | 0.00 | 0.00 | 265.00 | 316.02 | 1.26 |
| olmoe | lora | recompute | unique_stage_0002 × 1 | backward | 51.02 | 0.00 | 3.00 | 265.01 | 319.02 | 1.33 |
| olmoe | lora | save | unique_stage_0000 × 1 | forward | 135.94 | 0.00 | 52.91 | 16.10 | 204.94 | 1.17 |
| olmoe | lora | save | unique_stage_0000 × 1 | backward | 126.32 | 0.00 | 12.62 | 48.09 | 187.03 | 1.61 |
| olmoe | lora | save | unique_stage_0001 × 3 | forward | 90.94 | 0.00 | 50.03 | 16.10 | 157.07 | 1.13 |
| olmoe | lora | save | unique_stage_0001 × 3 | backward | 128.35 | 0.00 | 15.63 | 48.09 | 192.07 | 1.71 |
| olmoe | lora | save | unique_stage_0002 × 1 | forward | 51.02 | 0.00 | 3.00 | 262.00 | 316.02 | 1.26 |
| olmoe | lora | save | unique_stage_0002 × 1 | backward | 6.00 | 0.00 | 3.00 | 3.10 | 12.11 | 0.14 |
| qwen35 | full | recompute | unique_stage_0000 × 1 | forward | 17.39 | 0.00 | 3.00 | 27.00 | 47.39 | 0.47 |
| qwen35 | full | recompute | unique_stage_0000 × 1 | backward | 20.39 | 0.00 | 31.51 | 72.23 | 124.13 | 1.15 |
| qwen35 | full | recompute | unique_stage_0001 × 1 | forward | 51.02 | 0.00 | 0.00 | 361.00 | 412.02 | 1.81 |
| qwen35 | full | recompute | unique_stage_0001 × 1 | backward | 51.02 | 0.00 | 99.00 | 265.01 | 415.03 | 1.98 |
| qwen35 | full | recompute | unique_stage_0002 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.11 |
| qwen35 | full | recompute | unique_stage_0002 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| qwen35 | full | recompute | unique_stage_0003 × 3 | forward | 16.16 | 0.00 | 3.00 | 27.00 | 46.16 | 0.89 |
| qwen35 | full | recompute | unique_stage_0003 × 3 | backward | 19.16 | 0.00 | 29.32 | 72.30 | 120.78 | 2.04 |
| qwen35 | full | save | unique_stage_0000 × 1 | forward | 17.39 | 0.00 | 35.09 | 9.00 | 61.49 | 0.47 |
| qwen35 | full | save | unique_stage_0000 × 1 | backward | 52.49 | 0.00 | 31.51 | 46.14 | 130.13 | 0.95 |
| qwen35 | full | save | unique_stage_0001 × 1 | forward | 51.02 | 0.00 | 99.00 | 262.00 | 412.02 | 1.81 |
| qwen35 | full | save | unique_stage_0001 × 1 | backward | 102.00 | 0.00 | 99.00 | 3.10 | 204.11 | 0.24 |
| qwen35 | full | save | unique_stage_0002 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.11 |
| qwen35 | full | save | unique_stage_0002 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| qwen35 | full | save | unique_stage_0003 × 3 | forward | 16.16 | 0.00 | 34.12 | 16.06 | 66.35 | 0.86 |
| qwen35 | full | save | unique_stage_0003 × 3 | backward | 50.28 | 0.00 | 29.32 | 57.29 | 136.89 | 1.66 |
| qwen35 | lora | recompute | unique_stage_0000 × 1 | forward | 62.83 | 0.00 | 3.02 | 35.36 | 101.21 | 1.20 |
| qwen35 | lora | recompute | unique_stage_0000 × 1 | backward | 65.83 | 0.00 | 1.66 | 73.59 | 141.08 | 2.37 |
| qwen35 | lora | recompute | unique_stage_0001 × 1 | forward | 19.24 | 0.00 | 3.00 | 30.00 | 52.24 | 0.59 |
| qwen35 | lora | recompute | unique_stage_0001 × 1 | backward | 22.24 | 0.00 | 4.84 | 73.50 | 100.58 | 1.32 |
| qwen35 | lora | recompute | unique_stage_0002 × 1 | forward | 51.02 | 0.00 | 0.00 | 265.00 | 316.02 | 1.27 |
| qwen35 | lora | recompute | unique_stage_0002 × 1 | backward | 51.02 | 0.00 | 3.00 | 265.01 | 319.02 | 1.34 |
| qwen35 | lora | recompute | unique_stage_0003 × 2 | forward | 17.82 | 0.00 | 3.00 | 35.36 | 56.18 | 1.01 |
| qwen35 | lora | recompute | unique_stage_0003 × 2 | backward | 20.82 | 0.00 | 4.66 | 73.69 | 99.16 | 2.28 |
| qwen35 | lora | save | unique_stage_0000 × 1 | forward | 62.83 | 0.00 | 38.63 | 16.03 | 117.49 | 1.20 |
| qwen35 | lora | save | unique_stage_0000 × 1 | backward | 49.50 | 0.00 | 1.66 | 41.16 | 92.32 | 1.68 |
| qwen35 | lora | save | unique_stage_0001 × 1 | forward | 19.24 | 0.00 | 36.89 | 10.00 | 66.13 | 0.62 |
| qwen35 | lora | save | unique_stage_0001 × 1 | backward | 54.28 | 0.00 | 4.84 | 43.02 | 102.14 | 1.04 |
| qwen35 | lora | save | unique_stage_0002 × 1 | forward | 51.02 | 0.00 | 3.00 | 262.00 | 316.02 | 1.26 |
| qwen35 | lora | save | unique_stage_0002 × 1 | backward | 6.00 | 0.00 | 3.00 | 3.10 | 12.11 | 0.14 |
| qwen35 | lora | save | unique_stage_0003 × 2 | forward | 17.82 | 0.00 | 35.70 | 16.03 | 69.55 | 1.02 |
| qwen35 | lora | save | unique_stage_0003 × 2 | backward | 51.86 | 0.00 | 4.66 | 45.46 | 101.98 | 1.81 |
| qwen35moe | full | recompute | unique_stage_0000 × 1 | forward | 81.51 | 0.00 | 3.00 | 42.10 | 126.60 | 0.81 |
| qwen35moe | full | recompute | unique_stage_0000 × 1 | backward | 84.51 | 0.00 | 159.85 | 87.58 | 331.94 | 1.73 |
| qwen35moe | full | recompute | unique_stage_0001 × 1 | forward | 51.02 | 0.00 | 0.00 | 361.00 | 412.02 | 1.78 |
| qwen35moe | full | recompute | unique_stage_0001 × 1 | backward | 51.02 | 0.00 | 99.00 | 262.01 | 412.03 | 1.90 |
| qwen35moe | full | recompute | unique_stage_0002 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.11 |
| qwen35moe | full | recompute | unique_stage_0002 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| qwen35moe | full | recompute | unique_stage_0003 × 3 | forward | 80.33 | 0.00 | 3.00 | 42.10 | 125.43 | 1.28 |
| qwen35moe | full | recompute | unique_stage_0003 × 3 | backward | 83.33 | 0.00 | 157.67 | 87.64 | 328.64 | 2.91 |
| qwen35moe | full | save | unique_stage_0000 × 1 | forward | 81.51 | 0.00 | 43.49 | 20.00 | 145.00 | 0.82 |
| qwen35moe | full | save | unique_stage_0000 × 1 | backward | 124.99 | 0.00 | 159.85 | 53.29 | 338.13 | 1.24 |
| qwen35moe | full | save | unique_stage_0001 × 1 | forward | 51.02 | 0.00 | 99.01 | 262.00 | 412.02 | 1.78 |
| qwen35moe | full | save | unique_stage_0001 × 1 | backward | 102.01 | 0.00 | 99.00 | 0.05 | 201.06 | 0.15 |
| qwen35moe | full | save | unique_stage_0002 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.11 |
| qwen35moe | full | save | unique_stage_0002 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| qwen35moe | full | save | unique_stage_0003 × 3 | forward | 80.33 | 0.00 | 39.40 | 20.00 | 139.73 | 1.26 |
| qwen35moe | full | save | unique_stage_0003 × 3 | backward | 119.73 | 0.00 | 157.67 | 55.38 | 332.77 | 2.16 |
| qwen35moe | lora | recompute | unique_stage_0000 × 1 | forward | 51.02 | 0.00 | 0.00 | 265.00 | 316.02 | 1.23 |
| qwen35moe | lora | recompute | unique_stage_0000 × 1 | backward | 51.02 | 0.00 | 3.00 | 262.01 | 316.02 | 1.24 |
| qwen35moe | lora | recompute | unique_stage_0001 × 1 | forward | 137.88 | 0.00 | 3.02 | 46.22 | 187.12 | 1.79 |
| qwen35moe | lora | recompute | unique_stage_0001 × 1 | backward | 140.88 | 0.00 | 12.53 | 130.15 | 283.56 | 3.58 |
| qwen35moe | lora | recompute | unique_stage_0002 × 1 | forward | 94.22 | 0.00 | 3.00 | 46.22 | 143.44 | 1.15 |
| qwen35moe | lora | recompute | unique_stage_0002 × 1 | backward | 97.22 | 0.00 | 15.72 | 132.09 | 245.04 | 2.30 |
| qwen35moe | lora | recompute | unique_stage_0003 × 2 | forward | 92.87 | 0.00 | 3.00 | 46.22 | 142.09 | 1.59 |
| qwen35moe | lora | recompute | unique_stage_0003 × 2 | backward | 95.87 | 0.00 | 15.53 | 124.25 | 235.65 | 3.44 |
| qwen35moe | lora | save | unique_stage_0000 × 1 | forward | 51.02 | 0.00 | 3.01 | 262.00 | 316.02 | 1.24 |
| qwen35moe | lora | save | unique_stage_0000 × 1 | backward | 6.01 | 0.00 | 3.00 | 0.00 | 9.01 | 0.02 |
| qwen35moe | lora | save | unique_stage_0001 × 1 | forward | 137.88 | 0.00 | 61.83 | 16.10 | 215.80 | 1.83 |
| qwen35moe | lora | save | unique_stage_0001 × 1 | backward | 136.87 | 0.00 | 12.53 | 48.16 | 197.56 | 2.49 |
| qwen35moe | lora | save | unique_stage_0002 × 1 | forward | 94.22 | 0.00 | 63.22 | 16.10 | 173.54 | 1.21 |
| qwen35moe | lora | save | unique_stage_0002 × 1 | backward | 144.73 | 0.00 | 15.72 | 48.09 | 208.54 | 1.65 |
| qwen35moe | lora | save | unique_stage_0003 × 2 | forward | 92.87 | 0.00 | 58.91 | 18.03 | 169.81 | 1.63 |
| qwen35moe | lora | save | unique_stage_0003 × 2 | backward | 139.24 | 0.00 | 15.53 | 48.16 | 202.93 | 2.54 |
| qwen3moe | full | recompute | unique_stage_0000 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.06 |
| qwen3moe | full | recompute | unique_stage_0000 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| qwen3moe | full | recompute | unique_stage_0001 × 1 | forward | 78.32 | 0.00 | 3.00 | 39.10 | 120.41 | 0.85 |
| qwen3moe | full | recompute | unique_stage_0001 × 1 | backward | 81.32 | 0.00 | 153.10 | 84.58 | 319.00 | 1.85 |
| qwen3moe | full | recompute | unique_stage_0002 × 3 | forward | 78.32 | 0.00 | 3.00 | 39.10 | 120.41 | 0.84 |
| qwen3moe | full | recompute | unique_stage_0002 × 3 | backward | 81.32 | 0.00 | 153.10 | 84.58 | 318.99 | 1.79 |
| qwen3moe | full | recompute | unique_stage_0003 × 1 | forward | 51.02 | 0.00 | 0.00 | 361.00 | 412.02 | 1.81 |
| qwen3moe | full | recompute | unique_stage_0003 × 1 | backward | 51.02 | 0.00 | 99.00 | 265.01 | 415.03 | 1.98 |
| qwen3moe | full | save | unique_stage_0000 × 1 | forward | 48.02 | 0.00 | 3.02 | 0.00 | 51.03 | 0.05 |
| qwen3moe | full | save | unique_stage_0000 × 1 | backward | 3.02 | 0.00 | 96.00 | 0.05 | 99.06 | 0.11 |
| qwen3moe | full | save | unique_stage_0001 × 1 | forward | 78.32 | 0.00 | 30.34 | 20.00 | 128.66 | 0.83 |
| qwen3moe | full | save | unique_stage_0001 × 1 | backward | 108.66 | 0.00 | 153.10 | 50.29 | 312.05 | 1.32 |
| qwen3moe | full | save | unique_stage_0002 × 3 | forward | 78.32 | 0.00 | 30.34 | 20.00 | 128.66 | 0.83 |
| qwen3moe | full | save | unique_stage_0002 × 3 | backward | 108.66 | 0.00 | 153.10 | 50.29 | 312.05 | 1.36 |
| qwen3moe | full | save | unique_stage_0003 × 1 | forward | 51.02 | 0.00 | 99.00 | 262.00 | 412.02 | 1.81 |
| qwen3moe | full | save | unique_stage_0003 × 1 | backward | 102.00 | 0.00 | 99.00 | 3.10 | 204.11 | 0.24 |
| qwen3moe | lora | recompute | unique_stage_0000 × 1 | forward | 135.94 | 0.00 | 3.02 | 43.22 | 182.18 | 1.21 |
| qwen3moe | lora | recompute | unique_stage_0000 × 1 | backward | 138.94 | 0.00 | 12.62 | 114.87 | 266.43 | 2.40 |
| qwen3moe | lora | recompute | unique_stage_0001 × 1 | forward | 51.02 | 0.00 | 0.00 | 265.00 | 316.02 | 1.26 |
| qwen3moe | lora | recompute | unique_stage_0001 × 1 | backward | 51.02 | 0.00 | 3.00 | 265.01 | 319.02 | 1.33 |
| qwen3moe | lora | recompute | unique_stage_0002 × 1 | forward | 90.94 | 0.00 | 3.00 | 43.09 | 137.04 | 1.17 |
| qwen3moe | lora | recompute | unique_stage_0002 × 1 | backward | 93.94 | 0.00 | 15.63 | 114.98 | 224.55 | 2.48 |
| qwen3moe | lora | recompute | unique_stage_0003 × 2 | forward | 90.94 | 0.00 | 3.00 | 43.22 | 137.16 | 1.17 |
| qwen3moe | lora | recompute | unique_stage_0003 × 2 | backward | 93.94 | 0.00 | 15.62 | 114.98 | 224.55 | 2.47 |
| qwen3moe | lora | save | unique_stage_0000 × 1 | forward | 135.94 | 0.00 | 52.91 | 16.10 | 204.94 | 1.23 |
| qwen3moe | lora | save | unique_stage_0000 × 1 | backward | 126.31 | 0.00 | 12.62 | 48.13 | 187.06 | 1.69 |
| qwen3moe | lora | save | unique_stage_0001 × 1 | forward | 51.02 | 0.00 | 3.00 | 262.00 | 316.02 | 1.27 |
| qwen3moe | lora | save | unique_stage_0001 × 1 | backward | 6.00 | 0.00 | 3.00 | 3.10 | 12.11 | 0.14 |
| qwen3moe | lora | save | unique_stage_0002 × 1 | forward | 90.94 | 0.00 | 50.03 | 16.10 | 157.07 | 1.21 |
| qwen3moe | lora | save | unique_stage_0002 × 1 | backward | 128.35 | 0.00 | 15.63 | 48.13 | 192.10 | 1.82 |
| qwen3moe | lora | save | unique_stage_0003 × 2 | forward | 90.94 | 0.00 | 50.03 | 16.10 | 157.07 | 1.20 |
| qwen3moe | lora | save | unique_stage_0003 × 2 | backward | 128.35 | 0.00 | 15.62 | 48.13 | 192.10 | 1.82 |
