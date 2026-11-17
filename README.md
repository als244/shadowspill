# ShadowSpill

Train PyTorch models within a memory budget. ShadowSpill plans which tensors to
keep on the GPU, spill to other storage, and recompute during backward.

A lightweight Python client prepares training on a GPU server and submits batches.
Inspect the plan's estimated step time, execution lanes and memory usage in an
interactive HTML report before training begins.

## Install

Clone the repository on each machine you will use:

```bash
git clone https://github.com/als244/shadowspill.git
cd shadowspill
```

On the **GPU server**, use Linux with an NVIDIA GPU, its driver and the CUDA
toolkit compiler installed. Setup installs Python 3.12 and the training packages:

```bash
./scripts/setup.sh
source .venv/bin/activate
```

On a separate **client**, only Python 3.12+ and the client package are needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./packages/client
```

The server installation already includes the client. See
[installation options](docs/development/repository.md#setup) for other environments.

## Start the service

Set `SHADOWSPILL_TOKEN` to the same private credential on server and client.
Save this as `serve.py` on the GPU server and run `python serve.py`:

```python
from shadowspill.server import Server

Server(service_root="runs/service", model_root="/path/to/models").serve()
```

The server uses its visible GPUs and installed built-in definitions. `service_root`
holds sessions, logs, outputs and training checkpoints. The optional `model_root`
points to existing model directories such as `Qwen3.5-9B/`, read without modification.
Datasets stay on the client, which submits batches.

## Fine-Tune Qwen3.5-9B [Full-Parameter]

[Prepare UltraChat](docs/examples/client-training.md#prepare-ultrachat) with the
Qwen3.5 tokenizer and assistant-token masks. Both fine-tuning examples use this
same dataset. This one trains all text-decoder parameters on complete,
variable-length conversations: up to 8,192 model positions per conversation and
65,536 tokens per step. Run from the repository root on the client;
replace `training-host` with your server's SSH alias.

```python
from examples.training.text_data import ConversationBatches
from shadowspill_client import (
    SSH, BatchConfig, Memory, Model, Objective, Optimizer, ServiceClient, Weights,
)

data = ConversationBatches(
    "data/ultrachat-qwen35", max_sequence_length=8192, tokens_per_step=65536,
)
model = Model.from_pretrained("Qwen/Qwen3.5-9B")
optimizer = Optimizer("adamw", precision={
    "gradient_dtype": "bfloat16", "opt_state_dtype": "bfloat16", "master_dtype": "bfloat16",
    "parameter_rounding": "stochastic", "opt_state_rounding": "stochastic",
})

service = ServiceClient(ssh=SSH("training-host", 8000))
training = service.create_training_client(
    model=model,
    objective=Objective("cross_entropy"),
    optimizer=optimizer,
    batch=BatchConfig(limits={
        "token_capacity": 65536, "max_sequence_length": 8192, "max_sequences": 1024,
    }),
    memory=Memory(device_gib=24, spill_gib=128),
).result()

report = training.plans.report().result()
report.download("plan.html").result()

pending = None
for step, batch in enumerate(data):
    following = training.step(
        batch, optim={"lr": 0.00001}, runtime_trace=step == 99,
    )
    if pending is not None:
        result = pending.result()
        print(result.completed_updates, result.loss)
    pending = following
    if step == 99:
        break

if pending is not None:
    result = pending.result()
    print(result.completed_updates, result.loss)
    if result.traces:
        result.traces[0].download_report("runtime.html").result()

training.save_checkpoint(name="latest").result()
training.close().result()
service.close()
```

Open `plan.html` on your laptop to inspect all prepared variants, estimated step
times, and GPU/spill memory breakdowns. It works offline. Conversations retain
their individual attention boundaries and are packed automatically. The reader
skips conversations over the length limit and averages cross entropy across all
unmasked target tokens in each global batch.
The loop queues the next batch before waiting, keeping at most two steps in flight.

The final step records a runtime trace. Open `runtime.html` for measured compute
and transfer timings, alongside that step's predicted Plan. Its memory breakdown
is reconstructed on the traced timeline, with interpolated times labeled. Both
reports work offline; tracing adds overhead to the selected step.

Memory values are **budgets per worker**, not measured usage: this example allows
24 GiB for GPU execution and 128 GiB for spill. The server also needs room for
runtime overhead and data staging; the report shows the planned allocations.
BF16 gradients and optimizer state reduce storage, with stochastic rounding for
parameter and optimizer-state updates. BF16 master dtype reuses parameter storage;
the model's small FP32 parameters stay FP32, without separate master copies.

## Fine-Tune Qwen3.5-35B-A3B [LoRA]

Use the same UltraChat batches and assistant-token masks. In the example above,
replace `model` with:

```python
from shadowspill_client import LoRA

model = Model.from_pretrained("Qwen/Qwen3.5-35B-A3B", lora=LoRA(rank=16))
```

The server loads the text decoder's checkpoint and trains adapters; the client
submits variable-length conversations with loss only on assistant tokens. The same report,
checkpoint and cleanup calls apply.

## Pretrain Qwen3-30B-A3B on eight GPUs

Start the service once inside an **eight-GPU allocation on one host**. Prepare
FineWeb-Edu with the Qwen3 tokenizer, as described in the
[distributed example](docs/examples/client-training.md#eight-gpu-data-parallel-pretraining).
Reuse the connection, reports and loop above with this setup and
`optim={"lr": 0.0003}` for pretraining updates.

```python
from examples.training.text_data import TokenBatches
from shadowspill_client import Preparation
from shadowspill_client.parallelism import data_parallel
from shadowspill_client.text import build_candidates

data = TokenBatches("data/fineweb-qwen3", sequence_length=4096, sequences_per_step=512)
model = Model("qwen3moe", config={"preset": "30b-a3b"}, initialize=Weights.random(seed=0))
parallelism = data_parallel(8)
optimizer = Optimizer("adamw", precision={
    "gradient_dtype": "float32", "opt_state_dtype": "float32", "master_dtype": "float32",
    "parameter_rounding": "stochastic",
})

training = service.create_training_client(
    model=model,
    objective=Objective.sum({
        "tokens": Objective("cross_entropy"), "balance": Objective("moe_balance"),
    }, weights={"balance": 0.001}),
    optimizer=optimizer,
    parallelism=parallelism,
    batch=BatchConfig(limits={
        "token_capacity": 2097152, "max_sequence_length": 4096, "max_sequences": 8192,
    }),
    memory=Memory(device_gib=20, spill_gib=96),
    prepare=Preparation(candidates=build_candidates(
        microbatch_tokens=[65536], projection_chunks=[0],
    )),
).result()
```

`parallelism` enters preparation: it determines rank-specific builds, parameter
storage, input placement and Plans. This DP configuration shards parameter storage
across eight GPUs. Each `step()` submits **one global batch** of 512 × 4,096 =
2,097,152 tokens (2M); workers accumulate microbatches and apply one update.
Scheduled padding counts toward that token cap. The execution budget is **20 GiB
per GPU**; eight 96 GiB spill budgets require 768 GiB of spill capacity in total.
Each rank accumulates four unchunked 64K microbatches, with FP32 gradients,
master weights and Adam state. The default allows 128 sequences per microbatch
(8× sixteen 4K sequences); the attention length bound remains 4K.

See the [client guide](docs/python/client.md), [model catalog](docs/python/catalog.md)
and [documentation index](docs/README.md). [Advanced usage](docs/python/advanced/README.md)
covers custom models, objectives, optimizers, split updates and direct engine APIs.
