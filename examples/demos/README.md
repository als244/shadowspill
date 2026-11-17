# Training demos

Run these files from the repository root on your client. Each training example
shows its model, optimizer, batch limits and memory budget, then submits steps
through the public client API. Model weights stay on the server.

Install `packages/client`, [start a service](../../README.md#start-the-service),
and set the same `SHADOWSPILL_TOKEN` on both machines. Use your server's OpenSSH
alias in `--host`. Prepare or copy the
[tokenized datasets](../../docs/examples/client-training.md) onto the client.

```bash
python -m examples.demos.qwen35_35b_lora --host training-host
```

| File | Training | Client data | GPU / spill budget per worker |
| --- | --- | --- | --- |
| [qwen35_9b_full.py](qwen35_9b_full.py) | Full-parameter fine-tuning; 64K tokens per step, conversations up to 8K | `data/ultrachat-qwen35` | 24 / 128 GiB |
| [qwen35_35b_lora.py](qwen35_35b_lora.py) | LoRA fine-tuning; the same conversations and batch limits | `data/ultrachat-qwen35` | 24 / 128 GiB |
| [qwen3_30b_dp8.py](qwen3_30b_dp8.py) | Pretraining on eight GPUs; 2M tokens per step, 4K sequences | `data/fineweb-qwen3` | 20 / 96 GiB |
| [qwen35_35b_32k.py](qwen35_35b_32k.py) | LoRA; one 32K sequence, 8K computation chunks | `data/fineweb-qwen35` | 12 / 128 GiB |
| [qwen35_35b_128k.py](qwen35_35b_128k.py) | LoRA; one 128K sequence, 8K computation chunks | `data/fineweb-qwen35` | 31 / 128 GiB |
| [qwen36_27b_64k.py](qwen36_27b_64k.py) | Text-only LoRA; one 64K sequence, 8K projection chunks | `data/fineweb-qwen35` | 30 / 128 GiB |

The default is four updates. Use `--data`, `--output`, `--port` and `--steps` to
change paths or run length. `--repeat-first-batch` repeats the first real batch
for a short learning check. Use `--url` instead of `--host` for an HTTPS service
or an existing loopback tunnel. Model and training settings are explicit in each file.
Long-context settings are explained in the
[Advanced guide](../../docs/python/advanced/long-context.md).

Each run saves `plan.html` before training, queues the next batch before waiting
for the previous result, and traces its final update into `runtime.html`.
Both HTML reports open offline on your laptop. The command table displays
server execution time, its Plan prediction, token throughput, sequence statistics
and loss. `results.json` retains the returned command records.
`run.json` records the session and run IDs as soon as creation is submitted,
so a disconnected client can reattach to the same preparation or training run.

## Reuse saved Plans

The training demos also save `plans.ssp` and `training-config.json`. To start
another Qwen3.5-35B-A3B UltraChat run from those Plans:

```bash
python -m examples.demos.saved_plan \
    --host training-host --prepared runs/demos/qwen35_35b_lora
```

[saved_plan.py](saved_plan.py) uploads the saved Plans and uses the saved setup.
This skips build, profiling and planning. It initializes weights from the
configured pretrained checkpoint; resuming trained weights uses a
[training checkpoint](../../docs/python/client.md#save-and-resume).

## Explore hypothetical budgets offline

Export measured preparation inputs using the
[offline planning workflow](../../docs/python/advanced/offline-sweeps.md).
Install the optional CPU planner on your client:

```bash
python -m pip install ./packages/engine \
    --config-setting cmake.define.SHADOWSPILL_CPU_ONLY=ON
python -m examples.demos.offline_sweep measurements.zip
```

[offline_sweep.py](offline_sweep.py) compares three execution budgets and three
transfer-rate assumptions. Its HTML index links to the timeline and memory
report for each feasible result. It needs neither a service connection nor GPU
libraries; all timings are predictions for the measured server.
