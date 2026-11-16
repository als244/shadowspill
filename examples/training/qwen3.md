# Qwen3 MoE full training from a client

The [CLI](../train_qwen3_distributed.py) and [Python example](../qwen3.py) use the
public training service. Install NumPy and `shadowspill-client` on the client;
the engine, model code and accelerator dependencies belong on the server.
Run the client once, outside torchrun, against an existing service allocation:

```bash
python -m examples.train_qwen3_distributed \
  --ssh chicago --server-port 8000 \
  --data /data/fineweb-qwen3 --preset 30b-a3b \
  --run-dir runs/qwen3-dp8 --ranks 8 \
  --sequence-length 4096 --sequences-per-step 512 \
  --microbatch-sizes 1 2 4 8 16 32 \
  --execution-gib 77 --spill-gib 96
```

Authenticate with `SHADOWSPILL_TOKEN`; `--server` selects a direct HTTP(S)
connection. The token corpus (`train.bin` and `meta.json`) lives on the client.
`--preset` selects random initialization from an installed Qwen3 configuration;
`tiny` supports small checks. Alternatively, `--checkpoint` selects an absolute
server path or a supported Hub reference. The server pins pretrained content.

`--sequences-per-step` is now the **global** count. The previous eight-worker
recipe's 64 sequences per rank is 512 here, preserving 2,097,152 tokens per
update. Changing worker count does not silently change the global batch.
Microbatch sizes remain sequences per worker and must divide each worker's
sequence count. Scheduled positions, including padding, stay within the global
token cap. A 128K cap cannot execute three 64K microbatches.

Defaults retain BF16 compute, FP32 gradients/master weights/AdamW moments,
random seed zero, learning rate `3e-4`, five updates and a ten-update schedule.
The model's router-balancing coefficient is used unless `--balance-weight`
overrides it. Cross-entropy and generic MoE balancing share one model evaluation;
balancing excludes artificial padding by default. The application normalizes
the complete objective once using `grad_scale=1/global_tokens`.

`--backward-pieces 2` uses `forward_backward()` contributions and one
`optim_step()`. With the default of one piece, it calls `step()`. Router balancing
uses each physical microbatch's statistics, so whole and split objectives can
differ. Each path must match its own reference.

DP shards weights, gradients and optimizer state. `--parallelism ep` selects the
existing generic expert ownership: local expert shards, replicated dense weights
and gradients, and sharded dense optimizer state. EP requires at least two
workers, BF16 full training and unchunked execution.

Memory limits are per worker. The application retains the 768 MiB profiling
reserve (`--external-headroom-mib`, including explicit zero); ordinary engine
preparation still defaults to 512 MiB. External/whole-process memory overruns
are reported by default. `--reject-overbudget` opts into strict rejection.
Pool bounds and planned memory admission always apply.

[Settings](qwen/settings.py) and [metadata resolution](qwen/setup.py) are shared
with the Qwen3.5 LoRA application. [Shared setup](application/setup.py) creates
or resumes the ready client using the same lifecycle as Llama and regression.
The [runtime loop](application/loop.py) handles data, schedules, calls and results.
[Shared text configuration](text/README.md) owns batch, chunk, profiling and
search choices; Qwen settings retain model selection and presets.

Advanced measurement reuse and CPU planning use the service APIs described in
the [client guide](../../docs/python/client.md).

Evaluation defaults to off; `--eval-every` enables it and requires `val.bin`.
Checkpointing defaults to every five updates and the final update. Resume with
the reference from an earlier run, using a fresh client output directory:

```bash
python -m examples.train_qwen3_distributed --ssh chicago \
  --run-dir runs/qwen3-resumed \
  --resume runs/qwen3-dp8/checkpoints/step-00000005.json
```

The saved settings, corpus identity, progress and Plans must match. Full
checkpoints restore weights, optimizer state and RNG, continuing at the same
corpus window and schedule. Resuming performs no build/profile/search.
Training/evaluation JSONL, resolved setup, Plan/checkpoint references and the
HTML report are written under the client run directory; artifacts stay on the
server. `--wandb-project` enables optional caller-side logging.
