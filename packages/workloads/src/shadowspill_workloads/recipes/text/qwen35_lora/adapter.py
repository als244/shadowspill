"""Qwen35 model/checkpoint hooks; the shared runner owns every phase."""

import json
from contextlib import contextmanager
from pathlib import Path

from shadowspill_workloads.recipes.text import training
from shadowspill_workloads.recipes.text.training import candidates as candidates
from shadowspill_workloads.recipes.text.training import (
    trainer_options as trainer_options,
)

evaluation_metric = training.global_loss
path_options = ("data", "checkpoint")

defaults = {
    **training.DEFAULTS,
    "checkpoint": None,
    "frozen_state_id": None,
    "lora_rank": 32,
    "lora_alpha": 32,
    "sequences_per_step": 8,
    "microbatch_sizes": [1, 2, 4, 8],
    "eval_updates": 2,
}


def validate(options):
    training.validate(options)
    if not options["checkpoint"]:
        raise ValueError("checkpoint metadata is required")
    if options["initialization"] == "checkpoint" and not options["frozen_state_id"]:
        raise ValueError("frozen_state_id is required for checkpoint initialization")
    if (
        type(options["lora_rank"]) is not int
        or options["lora_rank"] <= 0
        or options["lora_alpha"] <= 0
    ):
        raise ValueError("LoRA rank and alpha must be positive")


def inputs(options, *, rank=0, split="train"):
    from shadowspill_workloads.recipes.text.data.calls import Inputs

    return training.check_source(
        Inputs(options, rank=rank, world_size=options["ranks"], split=split),
        options,
        split,
    )


def describe(options):
    metadata = json.loads((Path(options["checkpoint"]) / "config.json").read_text())
    return dict(
        frozen_state_id=training.frozen_state_identity(options, metadata),
        initialization=options["initialization"],
        checkpoint_config=metadata,
        dataset=json.loads((Path(options["data"]) / "meta.json").read_text()),
    )


@contextmanager
def setup(options, *, rank=0):
    import torch
    from mlops.optim import AdamW

    from shadowspill import Parallelism
    from shadowspill.integrations.mlops import sonicmoe_gradient_destination_rules
    from shadowspill.training.preparation.contracts import BuildSetup
    from shadowspill_workloads.models.construction import build_on_meta
    from shadowspill_workloads.models.qwen35moe.config import from_huggingface
    from shadowspill_workloads.models.qwen35moe.implementations.mlops import Model
    from shadowspill_workloads.models.shared.preparation import operator_preparations
    from shadowspill_workloads.recipes.text.objectives import (
        normalized_next_token_prediction,
    )
    from shadowspill_workloads.recipes.text.precision import Precision

    from .policies import build_options

    precision = Precision(
        master_dtype="float32", grad_dtype="float32", opt_state_dtype="float32"
    )
    precision.apply()
    metadata = describe(options)
    config = from_huggingface(
        metadata["checkpoint_config"], max_seq_len=options["sequence_length"]
    )
    if metadata["dataset"]["vocab_size"] > config.vocab_size:
        raise ValueError("corpus vocabulary exceeds model vocabulary")
    model = build_on_meta(
        Model,
        config=config,
        dtype="bfloat16",
        lora={"rank": options["lora_rank"], "alpha": options["lora_alpha"]},
    )
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=torch.tensor(options["learning_rate"]),
        **precision.optimizer_arguments(),
    )
    yield BuildSetup(
        model,
        optimizer,
        normalized_next_token_prediction,
        dict(
            parallelism=Parallelism.dp(ranks_per_replica=options["ranks"]),
            operator_preparation=operator_preparations("mlops", "bfloat16"),
            gradient_destination_rules=sonicmoe_gradient_destination_rules(),
            **precision.plan_arguments(),
            **build_options(options, options["sequences_per_example"]),
        ),
        metadata,
    )


@contextmanager
def initializer(options, metadata, *, rank=0):
    from .initialize import Initialize

    if metadata != describe(options):
        raise ValueError("checkpoint/corpus metadata differs from the build")
    if options["initialization"] == "random":
        from shadowspill_workloads.models.qwen35moe.config import from_huggingface
        from shadowspill_workloads.recipes.text.initialization import (
            Initialize as RandomInitialize,
        )

        config = from_huggingface(
            metadata["checkpoint_config"], max_seq_len=options["sequence_length"]
        )
        yield RandomInitialize("qwen35moe", config, seed=options["seed"])
        return
    with Initialize(options) as initialize:
        yield initialize
