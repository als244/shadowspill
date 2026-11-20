"""Upload saved Plans and restart Qwen3.5-35B-A3B UltraChat training."""

import json
from itertools import repeat

from examples.training.text_data import ConversationBatches
from shadowspill_client import SSH, ResolvedTraining, ServiceClient
from shadowspill_client.display import CommandTable
from shadowspill_client.text import text_batch_statistics

from ._options import arguments

options = arguments("saved_plan", "ultrachat-qwen35", saved_plan=True)
config = ResolvedTraining.from_dict(
    json.loads((options.prepared / "training-config.json").read_text())
)
data = ConversationBatches(
    options.data,
    max_sequence_length=8192,
    tokens_per_step=65536,
)
service = (
    ServiceClient(options.url)
    if options.url
    else ServiceClient(ssh=SSH(options.host, options.port))
)
plans = service.artifacts.upload(options.prepared / "plans.ssp").result()
creation = service.create_training_client(config=config, plans=plans, progress=True)
(options.output / "run.json").write_text(
    json.dumps({"session_id": creation.session_id, "run_id": creation.run_id}, indent=2)
)
training = creation.result()
training.plans.report().result().download(options.output / "plan.html").result()

table = CommandTable()
batches = iter(data)
if options.repeat_first_batch:
    batches = repeat(next(batches))
results, pending = [], None
for step in range(options.steps):
    batch = next(batches)
    validation = training.validate_batch(batch).result()
    following = training.step(
        batch,
        optim={"lr": 0.00001},
        runtime_trace=step == options.steps - 1,
    )
    table.track(
        "step",
        following,
        batch=text_batch_statistics(batch),
        planned_seconds=validation.estimated_seconds,
    )
    # Prepare and submit the next batch before waiting for the previous result.
    if pending is not None:
        results.append(pending.result())
    pending = following
results.append(pending.result())
results[-1].traces[0].download_report(options.output / "runtime.html").result()
(options.output / "results.json").write_text(
    json.dumps([result.record.to_dict() for result in results], indent=2) + "\n"
)
training.close().result()
service.close()
print(f"Reports: {options.output.resolve()}")
