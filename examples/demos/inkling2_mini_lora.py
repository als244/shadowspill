"""Fine-tune Inkling-2-Mini on 65,536 real text tokens per optimizer update."""

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


def main():
    options = arguments(
        "inkling2_mini_lora",
        "fineweb-inkling",
        checkpoint="thinkingmachines/Inkling-2-Mini",
    )
    data = TokenBatches(options.data, sequence_length=8192, sequences_per_step=8)
    batches = repeat(data[0]) if options.repeat_first_batch else iter(data)
    service = (
        ServiceClient(options.url)
        if options.url
        else ServiceClient(ssh=SSH(options.host, options.port))
    )
    with service:
        creation = service.create_training_client(
            model=Model.from_pretrained(
                options.checkpoint, lora=LoRA(rank=32, alpha=32)
            ),
            objective=Objective("cross_entropy"),
            optimizer=Optimizer(
                "adamw",
                precision={
                    "gradient_dtype": "float32",
                    "master_dtype": "float32",
                    "opt_state_dtype": "float32",
                },
            ),
            batch=BatchConfig(
                limits={
                    "token_capacity": 65536,
                    "max_sequence_length": 8192,
                    "max_sequences": 8,
                }
            ),
            memory=Memory(device_gib=30, spill_gib=160),
            prepare=Preparation(
                candidates=build_candidates(microbatch_tokens=[8192, 16384, 32768])
            ),
            operations={"step": ["metrics"]},
            progress=True,
        )
        (options.output / "run.json").write_text(
            json.dumps(
                {"session_id": creation.session_id, "run_id": creation.run_id}, indent=2
            )
            + "\n"
        )
        training = creation.result()
        try:
            training.plans.report().result().download(
                options.output / "plan.html"
            ).result()
            training.plans.download(options.output / "plans.ssp").result()
            (options.output / "training-config.json").write_text(
                json.dumps(training.get_config().to_dict(), indent=2) + "\n"
            )
            table = CommandTable()
            results, pending = [], None
            for step in range(1, options.steps + 1):
                batch = next(batches)
                validation = training.validate_batch(batch).result()
                following = training.step(
                    batch, optim={"lr": 1e-4}, runtime_trace=step == options.steps
                )
                table.track(
                    "step",
                    following,
                    batch=text_batch_statistics(batch),
                    planned_seconds=validation.estimated_seconds,
                )
                if pending is not None:
                    results.append(pending.result())
                pending = following
            results.append(pending.result())
            results[-1].traces[0].download_report(
                options.output / "runtime.html"
            ).result()
            (options.output / "results.json").write_text(
                json.dumps([r.record.to_dict() for r in results], indent=2) + "\n"
            )
        finally:
            training.close().result()


if __name__ == "__main__":
    main()
