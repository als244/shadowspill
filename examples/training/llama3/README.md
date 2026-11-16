# Llama training through the public client

The application reads a local token corpus and submits global Datum batches to an
existing service. It needs only `shadowspill-client` and NumPy. The server needs
the engine, workloads package and MLOps provider.

Start the server inside its assigned allocation, with `SHADOWSPILL_TOKEN` set:

```python
from shadowspill.server import Allocation, Server

Server(
    allocation=Allocation.from_environment(),
    service_root="runs/service",
    workload_packages=["shadowspill", "shadowspill-workloads"],
).serve(host="127.0.0.1", port=8000)
```

On the client, set the same credential and run from a copy of the examples tree:

```bash
python -m examples.train_llama3 --ssh chicago --server-port 8000 \
  --data /path/to/tokenized-corpus --run-dir runs/llama-client \
  --model llama3-tiny --sequence-length 16 --sequences-per-step 4 \
  --sequences-per-example 2 --steps 3 --checkpoint-every 1 \
  --execution-gib 2 --spill-gib 0.125
```

Use `--server https://service.example` for a direct connection. The corpus contains
`train.bin` and `val.bin` as uint32 token IDs and `meta.json` with `vocab_size` and
optional `dtype: "uint32"`. Tokenization stays with the application. Each sequence
attends independently, and its targets are the next tokens. Weights normalize by
the complete update's token count, preserving this application's mean-loss
behavior. The installed objective itself sums caller-weighted losses.

[`settings.py`](settings.py) supplies model, objective, optimizer, batch and memory
settings. [Shared setup](../application/setup.py) creates or resumes the ready
client and validates saved progress. The server owns the existing
build/profile/plan phases and worker initialization. [`loop.py`](loop.py) receives
that client and deals only with inputs, training settings and results. No model
factory, executor, partition or worker configuration appears in the loop.

The default submits one `step()` per update. `--backward-pieces 2` divides each
global batch into two groups, submits `forward_backward()` for each, and calls
`optim_step()` once. Groups may have unequal sizes. The same global normalization
and learning-rate schedule apply. `--sequences-per-example` suggests an initial
physical size; preparation also requests a whole-update size for coverage. Every
eligible schedule must fit the token cap including padding.

Each run directory retains reconnect IDs in `run.json`, local scalar JSONL files,
an HTML report and checkpoint-reference JSON files. Model and optimizer payloads
stay in durable service storage. Resume a saved boundary into a fresh directory:

```bash
python -m examples.train_llama3 --ssh chicago --run-dir runs/llama-resumed \
  --resume runs/llama-client/checkpoints/step-00000001.json
```

Resume uses the saved settings, original corpus and next update, restores optimizer
state and reuses saved Plans. Explicit settings that differ are rejected before
worker creation. A corpus path, metadata, file size or modification-time change
also requires an explicit new training setup. References need the original
service storage and authenticated owner; use the public checkpoint/Plan
[download and upload APIs](../../../docs/python/client.md) to transfer state to
another service. This application never retries an uncertain update. `run.json`
retains the session/run IDs for explicit inspection after an interrupted client.
