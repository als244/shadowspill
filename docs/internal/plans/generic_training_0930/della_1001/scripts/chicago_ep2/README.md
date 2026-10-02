# Chicago OLMoE recipe on Della EP2

Run in the reserved `codex` pane:

```bash
bash docs/internal/plans/generic_training_0930/della_1001/scripts/chicago_ep2/launch.sh
```

`train.py --config FILE --steps N` changes the configuration or short-run length.
The LR schedule always uses `schedule_total_steps`, independently of `steps`.

## Matching Chicago

- 16 layers; width 1,024; 16 attention/KV heads of width 64.
- 192 experts per layer, top-4, expert width 1,280; no shared expert.
- GPT-2 tokens, model vocabulary 50,304; maximum sequence length 2,048.
- 1,048,576 token slots per global update; normalization uses actual trained
  targets across both ranks. Each rank owns different packed microbatches.
- BF16 weights, gradients, and AdamW moments; no master parameters; stochastic
  rounding for parameter and moment stores, weight decay zero.
- LR 3e-4, 100 warmup updates, cosine decay to 3e-5 over 9,537 updates.
- Evaluation every 100 updates, checkpoints every 500 and at the final update.
- Cross-entropy, auxiliary loss, routing summaries, parameter/gradient health,
  step time and throughput are recorded. W&B has per-rank and aggregate runs.

This is a fresh initialization with the same distributions, not a replay of
Chicago's current checkpoint. Distributed reductions and local microbatch
routing statistics can differ from Chicago's single-device execution.

## Resource choices

Planning searches 16K, 32K and 64K tokens per rank per microbatch, with 32,
16 and 8 accumulated microbatches per rank respectively. The workload owns
one correctly sized MoonEP token buffer shared by every layer, plus shared
expert publication/reduction banks. Each candidate runs in fresh worker
processes so its communication resources are released before the next capacity.
The ordinary Trainer searches recomputation fractions and factor orderings for
each capacity; the outer experiment compares their admitted predicted step times.
Candidate progress is saved immediately and completed candidates are resumable.

Distinct expert parameters occupy 11.25 GiB per rank across all 16 layers.
Communication banks are allocated once: 1.40625 GiB BF16 home/replica weights
and 1.40625 GiB FP32 replica-gradient scratch, totaling 2.8125 GiB per GPU.
Returned expert gradients and optimizer state are BF16. The execution budget is
50 GiB with 6 GiB external headroom, leaving 44 GiB for the managed execution
pool, and 80 GiB host spill per rank. Physical measurements at model construction
and admission are recorded for each capacity. No MoonEP-specific behavior is
added to ShadowSpill's planner or allocator.

The data subset is under
`/home/as1669/storage/datasets/fineweb_edu_gpt2_chicago_sample`.
It contains the first 268,434,637 tokens of Chicago's prepared train stream,
ending at a document boundary, plus its validation stream. Packing uses the
ordinary text recipe with `long_documents="splice"` and window 1,024.

All run artifacts are under
`/home/as1669/storage/shadowspill/generic_training_0930/della_1001/chicago-olmoe12b-ep2-1002`:
console output and `planning-progress.json`; `candidates/tokens-N/` contains
each configuration’s per-rank artifact stores and graph pairs, plans, startup
diagnostics, metric JSONL files, W&B files, and checkpoints. W&B HTTPS is relayed
through the Della head node because the compute node has no public DNS/network.
Each rank's `plan-diagnostics.json` retains both graph-pair variants, their
input/mutation/output/workspace sizes, measured runtimes, and object mappings.
`launch.sh` expects that loopback SSH relay on port 18375; online machines can
run `train.py` directly without it.

## Validation before launch

Both save and recompute passed a two-block EP2 test with a BF16 router and one
shared communication buffer: three updates, identical replicated parameters,
and an independent PyTorch full-model loss/gradient reference. Data checks
verify disjoint microbatch assignment, global target normalization and the
unchanged LR schedule endpoints.
