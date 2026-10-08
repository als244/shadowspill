# Real-model validation of verified symmetric planning

Completed October 8, 2026, on fatnode, visibly in `codex:0.0`.

After the modularity cleanup, the same symmetric DP4 experiment was repeated
successfully as `llama-symmetric-20261008T052534`. All five planning points and
five updates passed; all 111 parameter tensors changed and replicas again
matched bitwise. Median actual step was 3.9048 s, with a 3.3825 s prediction.
Full artifacts remain on fatnode beside the earlier runs; small results are
copied to Della. See `evidence/modularity_validation.json` and
`logs/fatnode/dp4/llama-modularity.log`. The comparisons below refer to the
original symmetric/independent pair.

## Result

The complete 1,179,699,200-parameter Llama workload completed distributed
capture, GPU profiling, CPU search, physical admission, warmup, a traced step,
and five training updates on four GPUs. The symmetric mode activated on every
rank, without fallback. A fresh independent-planning control also completed.

| Measurement | Verified symmetric | Independent control |
|---|---:|---:|
| Geometry/ordering points | 5 | 5 |
| Ranks searching each point | 1 owner | All 4 |
| Preparation, including capture/profile/search | 193.44 s | 223.51 s |
| Selected microbatch tokens/rank | 2,048 | 2,048 |
| Predicted final callable step | 3.353 s | 3.166 s |
| Median measured step, slowest rank | 3.936 s | 4.050 s |
| Global tokens/s | 8,324 | 8,092 |
| Tokens/s/GPU | 2,081 | 2,023 |
| Save/recompute task-group occurrences | 41 / 15 | 52 / 4 |
| Final replicas within each run | Bitwise equal | Bitwise equal |
| Parameter tensors changed | 111 / 111 | 111 / 111 |

These are two short validation runs, not repeated performance trials. Each
profiles its own tasks; the selected plans differ. The preparation difference
includes capture and profiling and should not all be attributed to CPU search.
Profiling uses three minimum samples and zero minimum conditioning/measurement
time for this correctness check. Predicted and measured times are both retained;
this is not a simulator-accuracy qualification.

## Numerical comparison

| Update | Symmetric global loss | Independent global loss | Absolute difference |
|---|---:|---:|---:|
| 1 | 12.19105822 | 12.19105822 | 0 |
| 2 | 10.84449780 | 10.84449965 | 0.00000185 |
| 3 | 10.78478873 | 10.78765607 | 0.00286734 |
| 4 | 10.96311015 | 10.96368939 | 0.00057924 |
| 5 | 11.10485601 | 11.10644019 | 0.00158417 |

All losses and final compute parameters are finite. SHA-256 hashes of every
parameter confirm exact equality across the four replicas **within** each run.
The two runs are not bitwise equal to each other and selected different
save/recompute plans. This supports successful end-to-end execution and close
loss agreement; it does not establish bitwise invariance across plans.

The original October 1 DP1 weights/data hashes match exactly. Its first loss
agrees within 5.96e-8, while later losses differ by up to 0.06202 (symmetric)
or 0.06360 (independent) over these five updates. That older run used a previous
software tree, 4K-token microbatches, and a different accumulation order. The
fresh control reproduces most of that historical discrepancy. It is not
appropriate to claim strict numerical parity against the old run from this test.

## Configuration and search ownership

- Repository Llama workload: 12 blocks, width 2,048, intermediate width 7,168,
  16 query heads, 4 KV heads, vocabulary 128,256.
- FineWeb-Edu, original Llama-tokenized sequence stream and initialization.
- 2,048-token sequences; 32,768 global tokens/update, 8,192/rank.
- FP16 compute; FP32 gradients, sharded FP32 masters and optimizer moments.
- Original AdamW settings and 50-step warmup/cosine schedule; only the first
  five updates are executed. One LR=0 warmup and one traced step precede them.
- 10 GiB execution budget and 40 GiB spill capacity per rank.
- Search: 2K tokens/rank × 4 microbatches, three depth/breadth orderings;
  4K tokens/rank × 2 microbatches, two orderings.
- Owners are ranks 0/1/2 for the first geometry and 0/1 for the second. Rank 3
  participates in GPU profiling, plan verification, local admission, and all
  training. There are fewer independent CPU search points than ranks within
  either geometry. No point is redundantly searched in symmetric mode.
- Only healthy physical device minors 0, 2, 3, and 6 are exposed in a rootless
  Podman container, selected by stable GPU UUID. Broken devices are hidden
  from both CUDA and NVML/NCCL discovery.

## Evidence locations

On **fatnode**, the complete run directories are:

```text
/data/as1669/shadowspill/docs/internal/plans/qwen_moe_ep8_1007/evidence/fatnode/dp4/llama-symmetric-20261008T050111/
/data/as1669/shadowspill/docs/internal/plans/qwen_moe_ep8_1007/evidence/fatnode/dp4/llama-independent-20261008T050614/
```

Each has `rank-00000` through `rank-00003`, containing:

- `result.json`, `progress.jsonl`, `parameter_hashes.json`;
- `search.json`, `plan.json`, `symmetric_decisions.json`;
- `startup/summary.json` and `startup/timelines/{index,simulated,traced}.html`;
- `artifacts/`: complete local build/profile/planning store.

Console logs: `logs/fatnode/dp4/llama-{symmetric,independent}.log` under this
plan folder. Small result/plan files and logs are also copied to Della.
Machine-readable comparison: `evidence/fatnode/dp4/llama_comparison.json`.

Reproduction on fatnode:

```bash
python3 docs/internal/plans/qwen_moe_ep8_1007/scripts/fatnode/container.py llama --world-size 4
python3 docs/internal/plans/qwen_moe_ep8_1007/scripts/fatnode/container.py llama --world-size 4 --no-symmetric-planning
```

These create fresh timestamped output directories and use the preserved,
verified original inputs. They do not overwrite old runs or source data.

## Separate small-model failure and subsequent correction

The earlier four-rank, two-linear-layer test failed on a nine-element bias
after its first update, with and without symmetric planning. Maximum errors
were 3.3568 and 1.7034, respectively, far beyond rounding. The earlier two-rank
version passed. Both four-rank failing cases use the ordinary torch AdamW
path with parameter metrics; Llama uses MLOps AdamW and FP32 masters.

Logs: `logs/fatnode/dp4/{auto,auto-control}.log`. The Llama validation itself
required no production changes and did not resolve that failure.

The subsequent October 8 investigation isolated missing functionalization in
direct task compilation. Inductor overwrote an updated bias through scratch
reuse of an intermediate alias. A generic compiler correction passes the
original DP4 oracle, save/recompute, and independent-planning controls. See
[DP4_BIAS.md](DP4_BIAS.md) for the reproduction, precise cause, and validation.
