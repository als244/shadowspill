# Command-line training applications

Use the [standalone demos](../demos/README.md) for short, editable training
examples. These applications add learning-rate schedules, evaluation and
checkpoint continuation around the same public client API.

Install the client and [start a service](../../README.md#start-the-service).
Run each command once on the client, with the service's `SHADOWSPILL_TOKEN`.
Datasets stay on the client; model weights and training checkpoints stay on the
server. The client needs NumPy and `shadowspill-client`.

| Application | Input and configuration |
| --- | --- |
| [Llama](../train_llama3.py) | Token corpus; [model and loop settings](llama3/settings.py) |
| [Qwen3 pretraining](../train_qwen3_distributed.py) | Token corpus, random preset or pretrained checkpoint; DP or EP |
| [Qwen3.5 LoRA](../train_qwen35_lora.py) | Token corpus and pretrained checkpoint; [Qwen settings](qwen/settings.py) |
| [Regression](../regression_training.py) | Generated NumPy batches; [model and loop settings](regression/settings.py) |

## Run and resume

For example, train Qwen3.5 adapters on a tokenized corpus:

```bash
python -m examples.train_qwen35_lora \
    --ssh training-host --data data/fineweb-qwen35 \
    --checkpoint Qwen/Qwen3.5-35B-A3B --run-dir runs/qwen35-lora \
    --sequence-length 32768 --sequences-per-step 8 \
    --microbatch-sizes 1 2 4 8 --execution-gib 28 --spill-gib 144
```

The corpus contains `train.bin` and `meta.json`, plus `val.bin` when evaluation
is enabled. The binary files hold uint32 token IDs. See
[dataset preparation](../../docs/examples/client-training.md).
These CLI applications read fixed-length token windows; the fine-tuning demos
use variable-length conversations and assistant-token masks.

Use `--server URL` for a direct connection or `--server-port` to change the SSH
forwarded port. Each command's `--help` lists its options. Memory budgets are per
worker; `--sequences-per-step` is the global count. In the Qwen CLI,
`--microbatch-sizes` counts sequences per worker, whereas the public
`BuildCandidates.microbatch_sizes` API counts tokens.

The Qwen applications combine cross entropy and MoE balancing, excluding padding.
They take the balancing coefficient from model configuration unless
`--balance-weight` is supplied. BF16 compute uses FP32 gradients, masters and
optimizer state. These application settings are explicit choices, separate from
the fine-tuning demos' BF16 optimizer settings.

The default loop submits `step()`. `--backward-pieces 2` instead submits two
`forward_backward()` contributions and one `optim_step()`, preserving global
loss normalization. MoE balancing still depends on each physical microbatch.
Each application prepares only the operations its loop submits.

Each submitted `step()`, `forward_backward()`, `optim_step()` or `forward()`
appears in the [command table](../../docs/python/advanced/command-table.md).
Its time column uses server execution time. Text rows include token counts,
sequence statistics and the selected Plan's prediction. Per-command losses keep
the objective's reduction; `metrics.jsonl` records the normalized update loss.
`execution_seconds` sums server compute durations across a split update;
`client_elapsed_seconds` separately includes data preparation, uploads and waits.

The run directory records connection IDs, resolved setup, metrics, an HTML report
and checkpoint references. Resume with the same command and a fresh output
directory:

```bash
python -m examples.train_qwen35_lora --ssh training-host \
    --run-dir runs/qwen35-resumed \
    --resume runs/qwen35-lora/checkpoints/step-00000002.json
```

Resume checks saved settings, data identity and progress, restores the saved Plans
and training state, and continues the original schedule. See
[checkpoint semantics](../../docs/python/client.md#save-and-resume) and the
[complete workflow](../../docs/python/advanced/training-workflow.md).

## Source organization

`application/` owns shared client creation, resume records and update submission.
`text/` adds corpus, batch and preparation settings. `llama3/`, `qwen/` and
`regression/` supply application choices and evaluation policy. `text_data.py`
provides the corpus readers also used by the standalone demos. Setup returns a
ready training client; runtime loops only submit inputs and read results.

Model implementations live in the workloads package. Independent numerical
oracles and loss-curve comparison tools live in [reference](../../qualification/reference/training/README.md).
