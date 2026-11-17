"""Qwen3-30B-A3B pretraining on eight GPUs.

Run with python -m examples.demos.qwen3_30b_dp8.
"""

import json
from itertools import repeat

from examples.training.text_data import TokenBatches
from shadowspill_client import (
    SSH,
    BatchConfig,
    Memory,
    Model,
    Objective,
    Optimizer,
    Preparation,
    ServiceClient,
    Weights,
)
from shadowspill_client.display import CommandTable
from shadowspill_client.parallelism import data_parallel
from shadowspill_client.text import build_candidates, text_batch_statistics

from ._options import arguments

options = arguments("qwen3_30b_dp8", "fineweb-qwen3")
data = TokenBatches(options.data, sequence_length=4096, sequences_per_step=512)
service = (
    ServiceClient(options.url)
    if options.url
    else ServiceClient(ssh=SSH(options.host, options.port))
)
creation = service.create_training_client(
    model=Model(
        "qwen3moe", config={"preset": "30b-a3b"}, initialize=Weights.random(seed=0)
    ),
    objective=Objective.sum(
        {
            "tokens": Objective("cross_entropy"),
            "balance": Objective("moe_balance"),
        },
        weights={"balance": 0.001},
    ),
    optimizer=Optimizer(
        "adamw",
        precision={
            "gradient_dtype": "float32",
            "opt_state_dtype": "float32",
            "master_dtype": "float32",
            "parameter_rounding": "stochastic",
            "opt_state_rounding": "stochastic",
        },
    ),
    parallelism=data_parallel(8),
    batch=BatchConfig(
        limits={
            "token_capacity": 2097152,
            "max_sequence_length": 4096,
            "max_sequences": 4096,
        }
    ),
    memory=Memory(device_gib=20, spill_gib=96),
    prepare=Preparation(
        candidates=build_candidates(
            microbatch_tokens=[65536],
            projection_chunks=[0],
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
        optim={"lr": 0.0003},
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
