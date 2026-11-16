# Qwen3.5 MoE LoRA from a training client

The [Python application](../qwen35_lora.py) and
[CLI](../train_qwen35_lora.py) connect to an existing ShadowSpill service. The
client needs NumPy and `shadowspill-client`; it does not load the engine, Torch,
model code or checkpoint files. Tokenized `train.bin`, `val.bin` and `meta.json`
live on the client. An absolute base-checkpoint path is interpreted on the server;
a supported Hugging Face repository name also works.

```bash
python -m examples.train_qwen35_lora \
  --ssh chicago --server-port 8000 \
  --data /data/fineweb-qwen35 \
  --checkpoint /checkpoints/Qwen3.5-35B-A3B \
  --run-dir runs/qwen35-lora \
  --sequence-length 32768 --sequences-per-step 8 \
  --microbatch-sizes 1 2 4 8 \
  --execution-gib 28 --spill-gib 144
```

Authenticate using the service's `SHADOWSPILL_TOKEN`. Use `--server` for a direct
HTTP(S) connection. The service runs inside an existing GPU allocation and owns
build, profiling, planning, bounded initialization and execution workers.

The default LoRA rank and alpha are 32. Compute weights and adapters use BF16;
gradients, masters and AdamW moments use FP32. `--seed` selects fresh adapters and
runtime RNG. Loaded base weights retain their exact content identity. The server
pins local checkpoint checksums automatically; no caller-supplied frozen-base ID
is needed. `--compute-dtype float32` also selects FP32 adapters.

`--sequences-per-step` is the **global** sequence count, including distributed
runs. It must divide evenly across `--ranks` in this fixed-window application.
`--microbatch-sizes` lists sequences per physical worker microbatch, each dividing
that worker's sequence count. Repeated choices are built once. Changing the number
of workers does not implicitly enlarge the global batch. The scheduled token cap
is sequence length times the global sequence count, including padding. For example,
a 128K cap cannot execute three 64K microbatches.

The application combines cross-entropy with the reusable `shadowspill/moe_balance@1`
objective around one model evaluation. Balancing excludes artificial padding by
default and uses the checkpoint's auxiliary coefficient; `--balance-weight` overrides
that coefficient. Input weights are one. A single `grad_scale=1/global_tokens`
normalizes the complete objective before each optimizer update, and reported
losses use the same normalization. `--backward-pieces` submits separate
`forward_backward()` contributions followed by `optim_step()`. Router balancing
can differ between those contributions and a whole `step()`; each describes its
own microbatch statistic.

[Settings](qwen/settings.py) contain portable training choices.
[Setup](qwen/setup.py) resolves model metadata and search controls and returns
a ready client. The [runtime loop](qwen/loop.py) accepts that client and owns
only batches, schedules, evaluation and checkpoint cadence. Setup can run in a
separate component without changing the runtime loop. The installed engine's
build/profile/plan/execute phases and generic parallelism remain the same.

The [application integration test](../../packages/workloads/tests/workload_tests/test_qwen_client_service.py)
runs this CLI from a NumPy-only Tubingen client against Chicago. Whole and split
updates match independent small FP32 references, including validation and bitwise
checkpoint continuation. Production BF16 models and long contexts have separate
qualification requirements.

Chunk and retention candidates are explicit:

```text
--chunk-sizes 8192 --moe-chunk-sizes 8192 32768
--graph-pair-policies cost-input cost-attention attention-input
--resolution-options 0 1/4 1/2 3/4 1 --tail-save-counts 0 1 2 4
```

These reuse the installed text policies and PressureFit search. Each complete
build case carries its physical token capacity. Dense-attention retention and the
hybrid model's full-sequence linear-attention boundaries keep their existing
meaning. Defaults use one unchunked cost/input policy. Memory budgets are per
execution worker and do not guarantee that a requested checkpoint or context fits.

Training defaults to five updates with a ten-update warmup/cosine schedule. It
validates two fixed held-out updates every two training updates and at the final
update. `--eval-every 0` requires no validation file. `--checkpoint-every 0` disables
periodic saves. Optional `--wandb-project` uses the caller's installed W&B package;
ordinary operation has no W&B dependency.

Each new run directory contains `run.json`, resolved `setup.json`, `plans.json`,
training/evaluation JSONL, checkpoint-reference JSON files and `training.html`.
The server retains checkpoints and Plans. Resume through a saved reference:

```bash
python -m examples.train_qwen35_lora --ssh chicago \
  --run-dir runs/qwen35-resumed \
  --resume runs/qwen35-lora/checkpoints/step-00000002.json
```

Resume verifies saved settings, both corpus identities, checkpoint progress and
Plan identity before starting workers. It restores weights, optimizer state and
RNG, then continues at the next corpus window with the original schedule. Frozen
base parameters are omitted from native checkpoints and reconstructed from the
pinned source through the bounded loader. Each resume uses a fresh local output
directory. Saved measurement reuse and independent planning are service API
operations.
