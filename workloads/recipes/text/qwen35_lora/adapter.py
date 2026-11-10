"""Qwen35 model/checkpoint hooks; the shared runner owns every phase."""

import json
from contextlib import contextmanager
from pathlib import Path

from workloads.recipes.text import training
from workloads.recipes.text.training import candidates as candidates
from workloads.recipes.text.training import trainer_options as trainer_options

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
    if not options["checkpoint"] or not options["frozen_state_id"]:
        raise ValueError("checkpoint and frozen_state_id are required")
    if (
        type(options["lora_rank"]) is not int
        or options["lora_rank"] <= 0
        or options["lora_alpha"] <= 0
    ):
        raise ValueError("LoRA rank and alpha must be positive")


def inputs(options, *, rank=0, split="train"):
    from workloads.recipes.text.data.calls import Inputs

    return training.check_source(
        Inputs(options, rank=rank, world_size=options["ranks"], split=split),
        options,
        split,
    )


def describe(options):
    return dict(
        frozen_state_id=options["frozen_state_id"],
        checkpoint_config=json.loads(
            (Path(options["checkpoint"]) / "config.json").read_text()
        ),
        dataset=json.loads((Path(options["data"]) / "meta.json").read_text()),
    )


@contextmanager
def setup(options, *, rank=0):
    import torch
    from mlops.integrations.shadowspill import sonicmoe_gradient_destination_rules
    from mlops.optim import AdamW

    from shadowspill import Parallelism
    from workloads.models.construction import build_on_meta
    from workloads.models.qwen35moe.checkpoint import Qwen35MoECheckpoint
    from workloads.models.qwen35moe.implementations.mlops import Model
    from workloads.models.shared.preparation import operator_preparations
    from workloads.recipes.text.objectives import normalized_next_token_prediction
    from workloads.recipes.text.precision import Precision
    from workloads.recipes.training.contracts import BuildSetup

    from .policies import build_options

    precision = Precision(
        master_dtype="float32", grad_dtype="float32", opt_state_dtype="float32"
    )
    precision.apply()
    with Qwen35MoECheckpoint(
        options["checkpoint"], max_seq_len=options["sequence_length"]
    ) as checkpoint:
        config = checkpoint.config
    metadata = describe(options)
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
    with Initialize(options) as initialize:
        yield initialize
