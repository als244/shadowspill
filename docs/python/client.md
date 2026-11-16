# Training client

Use `ServiceClient` to connect and prepare a run, then use `TrainingClient` to
submit batches. Model construction and worker setup happen on the server before
the training loop starts.

The client needs Python 3.12+ and NumPy. Start with the
[complete text-training example](../examples/client-training.md), which includes
server startup, settings, input data and a training loop.

## Connection and setup

Connect through an existing SSH alias:

```python
from shadowspill_client import SSH, ServiceClient

service = ServiceClient(ssh=SSH("training-host", 8000))
```

Set `SHADOWSPILL_TOKEN` to the same credential on the server and client. For a
direct connection, use `ServiceClient(base_url="https://service.example")`.

## Create a training run

Pass the training choices as named arguments:

```python
from shadowspill_client import BatchConfig, Memory, Model, Objective, Optimizer, Weights

training = service.create_training_client(
    model=Model(
        "llama3",
        config={"preset": "tiny"},
        initialize=Weights.random(seed=0),
    ),
    objective=Objective("cross_entropy"),
    optimizer=Optimizer(
        "adamw",
        precision={"gradient_dtype": "float32", "opt_state_dtype": "float32",
                   "master_dtype": "float32"},
    ),
    batch=BatchConfig(limits={
        "token_capacity": 512,
        "max_sequence_length": 128,
        "max_sequences": 4,
    }),
    memory=Memory(4, 1),
).result()
```

| Argument | What you choose |
| --- | --- |
| `model` | The installed model, its configuration and initial weights. Here, a small Llama with random weights. |
| `objective` | What training minimizes. Here, next-token cross entropy. |
| `optimizer` | The update rule and its state. Here, AdamW with FP32 gradients, moments and master weights. Supply the learning rate on each update below. |
| `batch` | Limits for one submitted batch. Here, up to four sequences, each at most 128 tokens, and 512 scheduled tokens in total. |
| `memory` | GPU and spill budgets in GiB per worker. `Memory(4, 1)` means 4 GiB on the GPU and 1 GiB for spill. |

The server builds, measures and selects a fitting execution plan, initializes
weights and starts workers. `.result()` waits until the training client is ready.
Do this once before the loop; each `step()` then supplies only a batch and any
per-update settings. The [complete example](../examples/client-training.md) also
sets the vocabulary size from your corpus and constructs its input batches.

The [catalog](catalog.md) lists built-in models, optimizers and objectives, with
examples for LoRA, freezing parameters and combining objectives. Installed custom
definitions use the same settings.

Python's `**settings` notation would expand a dictionary into these same named
arguments. It is ordinary Python syntax, not an additional setup object or a
required part of this API.

## Inspect the plan

Before submitting the first batch, download the interactive report:

```python
report = training.plans.report().result()
report.download("plan.html").result()
```

Open `plan.html` in your laptop's browser. It includes every prepared build and
retained worker Plan, estimated step times, compute and transfer lanes, and
execution/spill memory over time with category breakdowns. The page works offline.
These are simulated predictions from the measured profiles.

Reports remain available after closing training. See the
[Plan guide](advanced/plan-report.md) for saved summaries and direct engine usage.

To compare a real training step with its predicted Plan:

```python
result = training.step(batch, optim={"lr": 0.0003}, runtime_trace=True).result()
result.traces[0].download_report("runtime.html").result()
```

The offline page opens on measured compute and transfer timings; switch views to
see the matching prediction. Memory occupancy is reconstructed from the Plan's
lifetimes on the traced clock, with interpolation labeled. Distributed calls
return one trace per worker. Tracing adds overhead only when requested.

## Runtime operations

| Call | What it does |
| --- | --- |
| `training.step(batch, optim={"lr": 0.0003}).result()` | Forward, backward and one optimizer update. |
| `training.forward_backward(batch).result()` | Accumulate gradients without updating weights. |
| `training.optim_step(optim={"lr": 0.0003}).result()` | Apply accumulated gradients once. |
| `training.forward(batch).result()` | Evaluate without updating training state. |
| `training.save_checkpoint(name="latest").result()` | Save weights, optimizer state and progress. |
| `training.close().result()` | Release the run's workers. |

A batch is a list of `Datum` records containing model inputs and objective inputs.
For text, `text_batch(token_sequences)` creates aligned next-token inputs and
targets, weighted by the whole batch's unmasked target count. Optional
`loss_masks=...` marks supervised tokens in the original sequences. Tokenization
stays in your application. Use `text_datum(tokens)` for explicit weights or a
token sum; it does not normalize a batch.

The setup chooses which compute operations are available. The default prepares
`step`; see the [split-update example](advanced/training-workflow.md) to prepare
backward accumulation and evaluation as well.

The completed result includes `loss`, `metrics` and `completed_updates`. Set the
learning rate on each `step` or `optim_step` with `optim={"lr": ...}`; this can come
from your learning-rate schedule. `forward_backward` only accumulates gradients
and takes no learning rate. Optional `Optimizer(defaults=...)` values are fallbacks
for omitted controls; a call's settings do not change the next call's defaults.
Objective sums are not automatically averaged: use input weights or an explicit
`grad_scale`. `text_batch` supplies weights for a mean token loss. Construct it
once before splitting the resulting list across backward calls for one update.

Batch limits include scheduled padding. For example, a 128K-token cap cannot
execute three 64K microbatches.

## Save and resume

Save at a completed update boundary, then reuse the saved Plans and checkpoint:

```python
checkpoint = training.save_checkpoint(name="latest").result()
plans = training.plans
training.close().result()

training = service.create_training_client(
    plans=plans, checkpoint=checkpoint,
).result()
```

Checkpoints stay in service storage. Use `checkpoint.download("./checkpoint").result()`
for a local copy. Close the training client when finished, then close the service
connection. Closing a connection alone does not stop a run.

A wait timeout does not cancel or repeat an update. If the connection is lost,
[reconnect and inspect the original request](advanced/service-client.md#saved-plans-and-reconnect).
See [advanced usage](advanced/README.md) for full operation options, custom
objectives, preparation tuning and artifact management.
