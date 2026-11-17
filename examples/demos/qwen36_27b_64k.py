"""Qwen/Qwen3.6-27B text-only LoRA at 65,536 tokens.

Run with python -m examples.demos.qwen36_27b_64k.
"""

import json
from itertools import repeat

from examples.training.text_data import TokenBatches
from shadowspill_client import (
    SSH,
    BatchConfig,
    LoRA,
    Memory,
    Model,
    Objective,
    Optimizer,
    Preparation,
    ServiceClient,
)
from shadowspill_client.display import CommandTable
from shadowspill_client.text import build_candidates, text_batch_statistics

from ._options import arguments

options = arguments("qwen36_27b_64k", "fineweb-qwen35")
data = TokenBatches(options.data, sequence_length=65536, sequences_per_step=1)
service = (
    ServiceClient(options.url)
    if options.url
    else ServiceClient(ssh=SSH(options.host, options.port))
)
creation = service.create_training_client(
    model=Model.from_pretrained("Qwen/Qwen3.6-27B", lora=LoRA(rank=32, alpha=32)),
    objective=Objective("cross_entropy"),
    optimizer=Optimizer(
        "adamw",
        precision={
            "gradient_dtype": "bfloat16",
            "opt_state_dtype": "bfloat16",
            "master_dtype": "bfloat16",
            "parameter_rounding": "stochastic",
            "opt_state_rounding": "stochastic",
        },
    ),
    batch=BatchConfig(
        limits={
            "token_capacity": 65536,
            "max_sequence_length": 65536,
            "max_sequences": 8,
        }
    ),
    memory=Memory(device_gib=30, spill_gib=128),
    prepare=Preparation(
        candidates=build_candidates(
            microbatch_tokens=[65536],
            projection_chunks=[8192],
            moe_chunks=[0],
        )
    ),
    progress=True,
)
(options.output / "run.json").write_text(
    json.dumps({"session_id": creation.session_id, "run_id": creation.run_id}, indent=2)
)
training = creation.result()

# These artifacts are available before the first update.
training.plans.report().result().download(options.output / "plan.html").result()
training.plans.download(options.output / "plans.ssp").result()
(options.output / "training-config.json").write_text(
    json.dumps(training.get_config().to_dict(), indent=2) + "\n"
)

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
print(f"Reports and saved Plans: {options.output.resolve()}")
