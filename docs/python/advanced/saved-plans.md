# Download and reuse saved Plans

Saved Plans contain captured programs, measurements and selected schedules.
Starting from compatible Plans skips build, profiling and planning. Model
initialization and worker startup still happen. A checkpoint is separate: it
restores trained weights, optimizer state and progress.

## Inspect Plans before initializing training

Use the named setup arguments from the [basic example](../../../README.md), or
an existing preview's resolved configuration:

```python
plans = service.plan_training(config=preview.config, progress=True).result()
plans.report().result().download("plan.html").result()
plans.download("plans.ssp").result()

training = service.create_training_client(
    config=preview.config, plans=plans, progress=True,
).result()
```

`plan_training()` runs build, profiling and planning, then releases its temporary
resources. It publishes all prepared variants and their predicted HTML report
without loading training weights or starting persistent training workers.
Creation from those Plans initializes the workers without repeating preparation.
The report opens offline on your laptop.

Planning returns a future with a `run_id`. After reconnecting to the same session,
`service.get_planning(run_id).result()` retrieves its outcome without repeating
the job. For measurement exports used by laptop sweeps, use the separate
[`prepare_training()` workflow](offline-sweeps.md).

## Save a prepared run

After `create_training_client(...).result()`, even before the first step:

```python
import json
from pathlib import Path

training.plans.download("plans.ssp").result()
Path("training-config.json").write_text(json.dumps(training.get_config().to_dict()))
training.plans.report().result().download("plan.html").result()
training.close().result()
service.close()
```

Keep `plans.ssp` and its configuration together. `plan.html` is an independent,
offline inspection page. Plans do not contain your dataset or initialized weights.

## Start on one GPU

On the target server, select a visible GPU and start the same `serve.py` used in
the [basic example](../../../README.md#start-the-service):

```bash
CUDA_VISIBLE_DEVICES=3 python serve.py
```

The service uses physical GPU 3 as its one visible device. From the laptop:

```python
import json
from pathlib import Path
from shadowspill_client import SSH, ServiceClient
from shadowspill_client.protocol.training import ResolvedTraining

service = ServiceClient(ssh=SSH("target-host", 8000))
config = ResolvedTraining.from_dict(json.loads(Path("training-config.json").read_text()))
plans = service.artifacts.upload("plans.ssp").result()
training = service.create_training_client(config=config, plans=plans).result()

result = training.step(batch, optim={"lr": 0.0001}, runtime_trace=True).result()
result.traces[0].download_report("runtime.html").result()
training.close().result()
service.close()
```

`batch` is the caller's next batch, built as in the basic example. The saved
configuration retains the original initializer: random initialization recreates
its seed; a pretrained initializer loads its pinned checkpoint. The target must
have that source available, including the same absolute path for a local source.
To choose a different compatible initializer, supply the ordinary explicit setup
arguments alongside `plans=plans`; the target resolves those settings and checks
compatibility with the captured computation.

## Reuse distributed Plans on different physical GPUs

Prepare the first run with `parallelism=data_parallel(8)` as in the
[eight-GPU example](../../../README.md#pretrain-qwen3-30b-a3b-on-eight-gpus), then
download its Plans and configuration using the same save code above.

On a compatible target host with an assigned eight-GPU allocation, this example
changes the physical ordering of the eight devices:

```bash
CUDA_VISIBLE_DEVICES=7,6,5,4,3,2,1,0 python serve.py
```

Use the same upload and `create_training_client(config=config, plans=plans)` code.
The saved configuration retains eight logical workers, their parameter ownership
and input placement. The service maps those workers onto its assigned physical
devices; the client still submits one global batch. No client process per rank,
`torchrun`, or manual tensor redistribution is needed.

Select only devices assigned to you. A compatible target needs the captured GPU
architecture/toolchain, installed definition sources, enough memory and the saved
logical parallelism. The server checks compatibility before execution; changing
physical placement does not change the saved logical world size. Predictions
retain their original measurements, so compare a traced step on the new host
with the predicted report before relying on its timing estimate.

Changing GPU count or logical sharding requires compatible target Plans and
checkpoint conversion; that is separate from remapping the same logical workers.
See [checkpoint transfer and resume](service-client.md#uploads)
for continuing trained state across services.
