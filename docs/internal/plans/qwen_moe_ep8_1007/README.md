# Qwen MoE workloads and EP8 validation

## Requested experiment

- Qwen3-30B-A3B and the **text decoder** of Qwen3.5-35B-A3B.
- EP8, sequence length 1,024, **4,194,304 global tokens/update** (524,288/rank).
- First quickstart: microbatches of 8,192 / 16,384 / 32,768 / 65,536 tokens/rank,
  execution budgets of 20 / 30 / 40 / 50 / 60 / 70 GiB. Save all searched
  resolution plans, graph pairs, timings, and diagnostics, not just winners.
- Then end-to-end training. Proposed validation precision is BF16, 10 steps
  after warmup; user may override. No Nsight capture requested.
- Request an exclusive 8-GPU Della node for one hour **after** host preparation.
  Run visibly in `codex` tmux, actively monitor the allocation, retain progress
  across allocations. The expanded quickstart matrix may exceed one hour.

## Architecture evidence

Exact upstream revisions and URLs are recorded in `sources/*-source.json`;
`sources/*-config.json` are unmodified official configs. The reference modeling
files come from installed Transformers 5.16.1 and are evidence, not dependencies
of the workloads.

| Setting | Qwen3-30B-A3B | Qwen3.5-35B-A3B text |
|---|---:|---:|
| Decoder blocks | 48 | 40 |
| Hidden width | 2,048 | 2,048 |
| Attention | GQA every block | 3 Gated DeltaNet + 1 gated GQA, repeated 10 times |
| GQA query / KV heads | 32 / 4 | 16 / 2 |
| Attention head width | 128 | 256 |
| Experts / top-k | 128 / 8 | 256 / 8 |
| Expert intermediate width | 768 | 512 |
| Shared expert | none | width 512, learned sigmoid gate |
| Vocabulary | 151,936 | 248,320 |
| RoPE base / fraction | 1e6 / 1 | 1e7 / 0.25 |
| RMSNorm epsilon | 1e-6 | 1e-6, zero-centered ordinary/QK norms |
| DeltaNet K / V heads | — | 16 / 32 |
| DeltaNet K / V head width | — | 128 / 128 |
| Causal convolution width | — | 4 |
| Router | normalized top-8 | normalized top-8 |
| Auxiliary coefficient | 0.001 | 0.001 |
| Embedding / output head | untied | untied |

These are fresh-initialized language-model workloads. Qwen3.5's vision encoder,
multimodal position layout, and auxiliary multi-token-prediction module are outside
this text-training experiment. Ordinary text positions give the same RoPE phases
on all three multimodal axes. Existing dense `mlops_qwen35` remains unchanged.

## Implementation and validation agenda

- [x] Inspect official configurations and attention/router definitions.
- [x] Record the updated global batch and quickstart requirements.
- [x] Add model presets, MLOps implementations, and public workload names.
- [x] Verify counts, attention structure, routing, and tiny forward/backward
      results against independent/Transformers reference mathematics.
- [x] Wire EP through installed MLOps, with one shared MoonEP buffer per model.
- [x] Prepare reusable quickstart and training scripts and rank-local stores.
- [x] Validate host setup and prepare input data before requesting GPUs.
- [x] Reserve and actively watch an exclusive one-hour Della node (15199147 pending).
- [x] Implement opt-in verified symmetric CPU search and asymmetric fallback.
- [x] Check two-process CPU search, resume, admission, and failure handling.
- [x] Validate symmetric DP training, forward evaluation, and ordering search
      using two healthy fatnode GPUs in an isolated container.
- [x] Validate four healthy fatnode GPUs using the real 1.18B Llama workload
      and original FineWeb-Edu batches; retain search ownership and training checks.
- [x] Review module boundaries and function sizes; split long coordination
      routines and remove the planner/sweep circular import.
- [x] Rerun targeted and broader CPU regression after that cleanup.
- [x] Finish the post-refactor real-model DP4 rerun.
- [x] Commit/push the approved groups and align all four machines.
- [x] Isolate the separate DP4 small-model bias failure and correct compiler
      functionalization; pass save/recompute and independent-planning controls.
- [x] Complete the full suite; prepare the approved compiler correction and
      evidence commits for publication and rollout.
- [ ] GPU correctness, then full-size EP8 quickstart at all requested budgets.
- [ ] Train both models; report finite losses, parameter updates, memory,
      throughput, simulator error, and artifact paths.
- [ ] Update workload documentation and final evidence/status.

## Artifact policy

Source, notes and small evidence live here. Large results go under
`~/storage/shadowspill/qwen_moe_ep8_1007/`, separated by model, geometry, budget,
and rank. No imports from `dev/` or `moe_lab/`. The approved commit groups are
in [COMMIT_PLAN.md](COMMIT_PLAN.md); rollout progress is in `PROGRESS.md`.

## Running and resuming

On the allocated GPU node in `codex:0.0`:

```bash
bash docs/internal/plans/qwen_moe_ep8_1007/scripts/launch.sh --stage smoke
bash docs/internal/plans/qwen_moe_ep8_1007/scripts/launch.sh --stage all
```

Both commands are resumable: successful cases are skipped, interrupted attempts
keep their own directories, and completed compilation/profile/plan artifacts are
reused from stable stores. `--stage search`, `measure`, or `train` selects one
phase explicitly. The outer sweep stops immediately on a subprocess failure for
inspection; it never labels a failed run as complete. No training starts until
all quickstart searches and winner measurements for both models have completed.
Changing experiment settings requires a new output directory; resuming another
stage with the same settings preserves completed work.

Default physical budgets are 20–70 GiB as requested. The experiment reserves
4 GiB of each for external communication/runtime allocations; the ordinary
quickstart default remains 0.5 GiB. Spill capacity is 64 GiB/rank, 512 GiB total.
Planner workers are capped at 4/rank, and PyTorch CPU threads at 4/rank.
The experiment opts into `Distributed(..., symmetric_planning=True)` by default;
`--no-symmetric-planning` preserves the independent-rank search for comparison.
All ranks still profile communicating GPU tasks. Verified equivalent orderings
are searched once across ranks, with each ordering's ascending budget sequence
kept together. Every rank physically admits its local schedule before execution.

Search has 22 geometry/orderings per model: 7 at 8K, 6 at 16K, 5 at 32K and
4 at 64K. Across two models and six budgets this is 264 search points, each
with up to five resolution settings. All feasible resolution plans and failure
diagnostics are retained. Up to 12 budget winners get five measured updates;
then each model gets 10 real-data updates after startup warmup/tracing.

Results root: `~/storage/shadowspill/qwen_moe_ep8_1007/ep8-bf16/`:

- `smoke/`: small EP models through compiled quickstart save/recompute planning.
- `search/<model>/t<tokens>/`: every geometry's attempts and completion record.
- `combined/<model>/rank-*/`: merged geometry/order/budget search, measurements,
  and figures. `combined/<model>/measured-winners.json` also records all rank
  times and global throughput using the slowest rank's median.
- `measure/`: actual execution of each budget's winning geometry.
- `training/`: subsequent real-data training, per-rank/aggregate metrics and traces.
- `stores/<model>/`: persistent build/profile/plan stores, including graph pairs.

Quickstart uses its standard deterministic synthetic tokens. The subsequent
training uses a separate FineWeb-Edu validation sample: existing GPT-2 documents
were decoded at EOT boundaries and re-encoded with each official, pinned Qwen
tokenizer on the head node. Both 50M-token streams and provenance are under
`~/storage/shadowspill/qwen_moe_ep8_1007/data/`. Training splices fixed 1K
sequences and partitions each complete 4M-token update evenly across EP ranks.

## Current validation

CPU reference and workload regression tests pass (52 passed, 1 skipped). Both
new models match Hugging Face logits, all parameter gradients, and auxiliary
loss at small dimensions. Full-size meta parameter counts match exactly:
30,532,122,624 and 34,660,610,688. **The Qwen models' EP8 GPU validation is still
pending.** Generic symmetric planning has passed real DP2 checks and complete
Llama 1.18B training on four fatnode GPUs. See [symmetric planning evidence](SYMMETRIC_PLANNING.md)
and [the real-model DP4 comparison](REAL_MODEL_DP4.md), including its numerical
limits. The separate small-model failure is explained and corrected in
[DP4_BIAS.md](DP4_BIAS.md). The full suite passes: 1,167 tests, plus all
20 CPU and 51 GPU CTest canaries; one test is skipped.

Slurm rejected `--mem=0`; job 15199147 uses the established exclusive node
request: 8 GPUs, 96 CPUs, 960 GiB, one hour. Watcher runs on the head node in
`codex:qwen-watch`, polls every 10 seconds, and wakes this Codex thread before
launching any GPU work. At submission its estimated start was October 8 04:24 EDT.
