# Long-context LoRA training

## Continue pretraining Qwen3.5-35B-A3B at 32K

Continue training on FineWeb-Edu using LoRA, 32,768-token sequences, 8K internal
computation chunks, and a 12 GiB GPU execution budget. Chunking bounds temporary
computation storage; it does not turn one long sequence into independent 8K
sequences. Attention retains the full sequence's boundaries.

Start the service as in the [homepage](../../../README.md#start-the-service).
Prepare FineWeb-Edu with the model's tokenizer in the server's training environment:

```bash
python -m tools.datasets.prepare_text \
    --tokenizer Qwen/Qwen3.5-35B-A3B \
    --dataset HuggingFaceFW/fineweb-edu \
    --files 'sample/10BT/{:03d}_00000.parquet' \
    --max-train-tokens 100000000 \
    --out data/fineweb-qwen35
```

Copy that data directory to the client, as in [data preparation](../../examples/client-training.md).
Run from the repository root on the client, replacing `training-host` with the
server's SSH alias:

```python
from examples.training.text_data import TokenBatches
from shadowspill_client import (
    SSH, BatchConfig, LoRA, Memory, Model, Objective, Optimizer,
    Preparation, ServiceClient,
)
from shadowspill_client.text import build_candidates

data = TokenBatches(
    "data/fineweb-qwen35", sequence_length=32768, sequences_per_step=1,
)
service = ServiceClient(ssh=SSH("training-host", 8000))
training = service.create_training_client(
    model=Model.from_pretrained("Qwen/Qwen3.5-35B-A3B", lora=LoRA(rank=16)),
    objective=Objective("cross_entropy"),
    optimizer=Optimizer("adamw", precision={
        "gradient_dtype": "bfloat16", "opt_state_dtype": "bfloat16",
        "master_dtype": "bfloat16", "parameter_rounding": "stochastic",
        "opt_state_rounding": "stochastic",
    }),
    batch=BatchConfig(limits={
        "token_capacity": 32768, "max_sequence_length": 32768, "max_sequences": 8,
    }),
    memory=Memory(device_gib=12, spill_gib=128),
    prepare=Preparation(candidates=build_candidates(
        microbatch_tokens=[32768], projection_chunks=[8192], moe_chunks=[8192],
    )),
    progress=True,
).result()

training.plans.report().result().download("plan.html").result()

pending = None
for step, batch in enumerate(data):
    following = training.step(batch, optim={"lr": 0.00001}, runtime_trace=step == 99)
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

`microbatch_tokens` describes the physical input buffer. `projection_chunks`
and `moe_chunks` divide work inside the captured model while keeping the same
logical sequence. The bounded output head, retention policy and ordering keep
their normal defaults. The batch limits also allow shorter independent sequences
to pack into the same 32K capacity; this corpus reader chooses fixed long windows.
With this sequence bound, the default allows eight sequences per microbatch;
`max_sequences` remains a separate global batch limit. Use
`max_sequences_per_microbatch` in `build_candidates` for more packing headroom,
and `validate_batch()` to check a batch against the prepared layouts.

Open `plan.html` before training to inspect predicted time and GPU/spill memory.
The last step's `runtime.html` shows measured execution alongside its prediction.
The loop queues the next batch before waiting for the previous result. Memory
settings are per-worker budgets; the machine also needs room for worker overhead
and input staging. Gradients and Adam state use BF16 with stochastic rounding;
BF16 master dtype reuses parameter storage.

For more GPU workspace, set `Memory(device_gib=24, spill_gib=128)`; the model,
batch and chunk settings stay the same.

To compare budgets or hypothetical host-transfer bandwidth without repeating
GPU measurements, see [offline sweeps](offline-sweeps.md). For continuation on
other GPUs, see [saved Plans](saved-plans.md).
