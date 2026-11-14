"""GLM checkpoint, FP8/provider contracts and data hooks for the shared runner."""

from contextlib import contextmanager
from functools import partial

from shadowspill_workloads.recipes.text import training
from shadowspill_workloads.recipes.text.training import (
    trainer_options as trainer_options,
)

from .settings import DEFAULTS
from .settings import validate as validate_glm

defaults = {**training.DEFAULTS, **DEFAULTS}
path_options = ("data", "checkpoint")


def validate(options):
    training.validate(options)
    if not options["checkpoint"]:
        raise ValueError("checkpoint metadata is required")
    if options["initialization"] == "checkpoint" and not options["frozen_state_id"]:
        raise ValueError("frozen_state_id is required for checkpoint initialization")
    validate_glm(options)


def candidates(options):
    return training.candidates(options, alignment=32)


def inputs(options, *, rank=0, split="train"):
    from .inputs import Inputs

    return training.check_source(
        Inputs(options, rank=rank, world_size=options["ranks"], split=split),
        options,
        split,
    )


@contextmanager
def setup(options, *, rank=0):
    import torch
    from mlops.optim import AdamW

    from shadowspill import Parallelism
    from shadowspill.training.preparation.contracts import BuildSetup
    from shadowspill_workloads.models.shared.preparation import operator_preparations
    from shadowspill_workloads.recipes.text.precision import Precision

    from .inputs import loss
    from .model import checkpoint, describe
    from .policies import build_options

    precision = Precision(
        master_dtype="float32", grad_dtype="float32", opt_state_dtype="float32"
    )
    precision.apply()
    with checkpoint(options) as (model, source, _):
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=torch.tensor(options["learning_rate"]),
            **precision.optimizer_arguments(),
        )
        yield BuildSetup(
            model,
            optimizer,
            partial(loss, head_chunk_size=options["head_chunk_size"]),
            dict(
                parallelism=Parallelism.dp(ranks_per_replica=options["ranks"]),
                operator_preparation=operator_preparations("mlops", "bfloat16"),
                **precision.plan_arguments(),
                **build_options(options, options["sequences_per_example"]),
            ),
            describe(options, model, source),
        )


@contextmanager
def initializer(options, metadata, *, rank=0):
    from .model import checkpoint, describe

    with checkpoint(options) as (model, source, initialize):
        if describe(options, model, source) != metadata:
            raise ValueError("checkpoint/corpus metadata differs from the build")
        yield initialize


evaluation_metric = training.global_loss


def finish(root, options, records, *, rank):
    from .metrics import learning_report

    learning_report(
        root, options, [item["scalars"]["global_mean_loss"] for item in records], rank
    )
