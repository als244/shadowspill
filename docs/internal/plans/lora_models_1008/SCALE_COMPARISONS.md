# Full-model LoRA comparisons

BF16 base/compute; rank 32, alpha 32, FP32 factors/gradients/AdamW moments; no masters.
Llama 3 numerical preset: 12 layers, width 2048, FFN 7168, vocabulary 128256; 1.180B base parameters.
2048 tokens per step, sequence length 512; full training and rank-32 LoRA use the same data and initialization.
At least 5 exact-task warmups and 1 second at lr=0; 10 measured training updates. Fresh per-case artifact stores.

| Architecture | Mode | Variant | Trainable M | Step ms | tok/s | Peak process RSS GiB | Peak tree PSS GiB | Spill reserved / peak allocated GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llama3 | full | recompute | 1179.699 | 540.57 | 3,789 | 27.19 | 27.41 | 22 / 11.98 |
| llama3 | full | save | 1179.699 | 517.30 | 3,959 | 27.22 | 27.45 | 22 / 11.98 |
| llama3 | lora | recompute | 15.729 | 123.06 | 16,643 | 10.18 | 10.44 | 6 / 2.47 |
| llama3 | lora | save | 15.729 | 100.58 | 20,361 | 10.18 | 10.43 | 6 / 2.47 |

RSS is the case process peak, including its pinned pool; PSS is a 250 ms sampled process-tree peak that apportions shared pages and includes compiler children. These are different scopes.
Spill capacity follows the same tensor-count formula for each mode and is reserved up front. Report it separately from live allocation evidence; RSS reductions include smaller reservations.

## Individual graph pairs

All sizes are MiB. Total is input + mutated + output + workspace; totals across separate tasks are not a simultaneous model peak.
Occurrences indicate how often a structurally identical stage appears. Stage IDs are local to the case.

| Architecture | Mode | Variant | Stage × occurrences | Pass | Input | Mutated | Output | Workspace | Total | ms |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| llama3 | full | recompute | unique_stage_0000 × 1 | forward | 509.02 | 0.00 | 0.00 | 1769.51 | 2278.53 | 16.52 |
| llama3 | full | recompute | unique_stage_0000 × 1 | backward | 509.02 | 0.00 | 1010.01 | 767.52 | 2286.54 | 17.86 |
| llama3 | full | recompute | unique_stage_0001 × 12 | forward | 112.52 | 0.00 | 8.00 | 84.00 | 204.52 | 1.24 |
| llama3 | full | recompute | unique_stage_0001 × 12 | backward | 120.52 | 0.00 | 216.02 | 212.14 | 548.68 | 3.63 |
| llama3 | full | recompute | unique_stage_0002 × 1 | forward | 501.02 | 0.00 | 8.02 | 0.00 | 509.03 | 0.05 |
| llama3 | full | recompute | unique_stage_0002 × 1 | backward | 8.02 | 0.00 | 1002.00 | 0.05 | 1010.06 | 0.69 |
| llama3 | full | save | unique_stage_0000 × 1 | forward | 509.02 | 0.00 | 1010.00 | 759.51 | 2278.53 | 16.50 |
| llama3 | full | save | unique_stage_0000 × 1 | backward | 1018.00 | 0.00 | 1010.01 | 8.26 | 2036.27 | 1.53 |
| llama3 | full | save | unique_stage_0001 × 12 | forward | 112.52 | 0.00 | 92.12 | 28.00 | 232.65 | 1.24 |
| llama3 | full | save | unique_stage_0001 × 12 | backward | 204.65 | 0.00 | 216.02 | 164.16 | 584.82 | 2.75 |
| llama3 | full | save | unique_stage_0002 × 1 | forward | 501.02 | 0.00 | 8.02 | 0.00 | 509.03 | 0.05 |
| llama3 | full | save | unique_stage_0002 × 1 | backward | 8.02 | 0.00 | 1002.00 | 0.05 | 1010.06 | 0.69 |
| llama3 | lora | recompute | unique_stage_0000 × 1 | forward | 610.52 | 0.00 | 8.02 | 92.00 | 710.54 | 1.45 |
| llama3 | lora | recompute | unique_stage_0000 × 1 | backward | 618.52 | 0.00 | 5.00 | 186.48 | 810.00 | 2.79 |
| llama3 | lora | recompute | unique_stage_0001 × 1 | forward | 509.02 | 0.00 | 0.00 | 767.51 | 1276.53 | 10.75 |
| llama3 | lora | recompute | unique_stage_0001 × 1 | backward | 509.02 | 0.00 | 8.00 | 767.52 | 1284.54 | 10.88 |
| llama3 | lora | recompute | unique_stage_0002 × 11 | forward | 117.52 | 0.00 | 8.00 | 92.00 | 217.52 | 1.41 |
| llama3 | lora | recompute | unique_stage_0002 × 11 | backward | 125.52 | 0.00 | 13.00 | 186.83 | 325.35 | 3.07 |
| llama3 | lora | save | unique_stage_0000 × 1 | forward | 610.52 | 0.00 | 103.14 | 28.00 | 741.66 | 1.45 |
| llama3 | lora | save | unique_stage_0000 × 1 | backward | 195.63 | 0.00 | 5.00 | 100.04 | 300.67 | 1.72 |
| llama3 | lora | save | unique_stage_0001 × 1 | forward | 509.02 | 0.00 | 8.00 | 759.51 | 1276.53 | 10.76 |
| llama3 | lora | save | unique_stage_0001 × 1 | backward | 16.00 | 0.00 | 8.00 | 8.27 | 32.27 | 0.19 |
| llama3 | lora | save | unique_stage_0002 × 11 | forward | 117.52 | 0.00 | 95.50 | 28.00 | 241.02 | 1.41 |
| llama3 | lora | save | unique_stage_0002 × 11 | backward | 208.02 | 0.00 | 13.00 | 108.02 | 329.04 | 2.06 |
