# Train on a tokenized corpus

This example trains a small Llama model from scratch. The GPU machine runs the
service; the client reads token IDs, submits batches and saves a checkpoint.

## Start the service

Install the engine, workloads and MLOps provider using the
[server setup instructions](../development/repository.md#setup). Set
`SHADOWSPILL_TOKEN` to the same credential on server and client, then run this
on the GPU machine:

```python
from shadowspill.server import Allocation, Server

Server(
    allocation=Allocation.from_environment(),
    storage_root="runs/service",
    workload_packages=["shadowspill", "shadowspill-workloads"],
).serve(host="127.0.0.1", port=8000)
```

Use an existing GPU allocation. The service manages its workers.

## Prepare the data

On the client, install `./packages/client`. Put your tokenized corpus in
`data/train.bin` as a flat array of uint32 token IDs, with vocabulary size in
`data/meta.json`, for example `{"vocab_size": 32000}`. Use the same tokenizer
throughout the corpus. This example needs at least 2,580 tokens.

Save the following two blocks together as `train.py`. Replace `training-host`
with your SSH alias for the GPU machine.

```python
import json
from pathlib import Path

import numpy as np

from shadowspill_client import (
    SSH, BatchConfig, Memory, Model, Objective, Optimizer, ServiceClient, Weights,
)
from shadowspill_client.text import text_datum

vocab_size = json.loads(Path("data/meta.json").read_text())["vocab_size"]
tokens = np.memmap("data/train.bin", dtype=np.uint32, mode="r")
sequence_length, sequences_per_step, steps = 128, 4, 5
window = (sequence_length + 1) * sequences_per_step
if len(tokens) < steps * window:
    raise ValueError(f"The corpus needs at least {steps * window} tokens.")


def prepare_training(service):
    return service.create_training_client(
        model=Model(
            "llama3",
            config={"preset": "tiny", "architecture": {
                "vocab_size": vocab_size, "max_seq_len": sequence_length,
            }},
            initialize=Weights.random(seed=0),
        ),
        objective=Objective("cross_entropy"),
        optimizer=Optimizer(
            "adamw",
            precision={"gradient_dtype": "float32", "opt_state_dtype": "float32",
                       "master_dtype": "float32"},
        ),
        batch=BatchConfig(limits={
            "token_capacity": sequence_length * sequences_per_step,
            "max_sequence_length": sequence_length,
            "max_sequences": sequences_per_step,
        }),
        memory=Memory(4, 1),  # GPU and spill budgets in GiB per worker.
    ).result()


def make_batch(step):
    start = step * window
    weight = 1 / (sequence_length * sequences_per_step)
    return [
        text_datum(tokens[offset:offset + sequence_length + 1], weights=weight)
        for offset in range(start, start + window, sequence_length + 1)
    ]
```

Setup selects the model, objective and memory limits. The input weights make each
update's loss the mean over its 512 target tokens. Preparation and worker setup
finish before the loop below starts.

## Train and save

```python
with ServiceClient(ssh=SSH("training-host", 8000)) as service:
    training = prepare_training(service)
    try:
        for step in range(steps):
            result = training.step(make_batch(step), optim={"lr": 0.0003}).result()
            print(f"Update {result.completed_updates}: loss={result.loss:.4f}")
        checkpoint = training.save_checkpoint(
            name="latest", metadata={"next_data_step": steps},
        ).result()
        print("Checkpoint:", checkpoint.id)
    finally:
        training.close().result()
```

Run `python train.py`. The first setup builds and measures the model; subsequent
calls submit inputs to ready workers. The checkpoint stays in service storage.

Next, [resume from a checkpoint](../python/client.md#save-and-resume), use the
[Llama corpus application](../../examples/training/llama3/README.md) for longer runs,
or follow the [advanced example](../python/advanced/training-workflow.md) for split
updates and evaluation. [Custom definitions](../python/advanced/definitions.md) use the
same client interface.
