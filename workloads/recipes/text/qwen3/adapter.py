"""Qwen3 MoE construction and ownership hooks for the shared training runner."""

from contextlib import contextmanager

from workloads.recipes.text import training
from workloads.recipes.text.training import candidates as candidates
from workloads.recipes.text.training import trainer_options as trainer_options

from .common import model_config

defaults = {
    **training.DEFAULTS,
    "model": "qwen3moe-30b-a3b",
    "parallelism": "dp",
    "ranks": 8,
    "sequence_length": 4096,
    "sequences_per_step": 64,
    "microbatch_sizes": [1, 2, 4, 8, 16, 32],
    "learning_rate": 3e-4,
    "execution_gib": 77,
    "spill_gib": 96,
    "external_headroom_mib": 768,
    "eval_every": 0,
    "checkpoint_every": 5,
}
evaluation_metric = training.global_loss
path_options = ("data",)


def validate(options):
    training.validate(options)
    if options["parallelism"] not in {"dp", "ep"}:
        raise ValueError("parallelism must be dp or ep")
    if options["parallelism"] == "ep":
        if options["ranks"] < 2:
            raise ValueError("expert parallelism requires at least two ranks")
        if any(options["chunk_sizes"]):
            raise ValueError(
                "the EP provider has no execution-axis contract yet; "
                "use chunk_sizes=[0]"
            )
        options["preparation_ranks"] = options["ranks"]
    elif options["preparation_ranks"] != 1:
        raise ValueError("DP preparation uses one representative worker")


def inputs(options, *, rank=0, split="train"):
    from workloads.recipes.text.data.calls import Inputs

    return training.check_source(
        Inputs(options, rank=rank, world_size=options["ranks"], split=split),
        options,
        split,
    )


def _expert_setup(options, precision):
    from workloads.models.catalog import entry
    from workloads.recipes.text.expert_parallel import resources

    item, config = entry(options["model"]), model_config(options)
    if not item.expert_parallel or config.n_experts % options["ranks"]:
        raise ValueError("model experts must support the requested EP group")
    parallelism = resources.ownership(options["ranks"], 1)
    provider, resource = resources.description(
        config,
        item.family,
        precision,
        parallelism,
        options["sequence_length"] * options["sequences_per_example"],
    )
    return item, config, provider, resource, parallelism


@contextmanager
def setup(options, *, rank=0):
    import torch
    from mlops.optim import AdamW

    from shadowspill import Parallelism
    from workloads.models.catalog import entry
    from workloads.models.shared.preparation import operator_preparations
    from workloads.recipes.text.execution_policies import build_options
    from workloads.recipes.text.objectives import normalized_next_token_prediction
    from workloads.recipes.text.precision import Precision
    from workloads.recipes.training.contracts import BuildSetup

    precision = Precision(
        master_dtype="float32", grad_dtype="float32", opt_state_dtype="float32"
    )
    precision.apply()
    if options["parallelism"] == "ep":
        from workloads.recipes.text.expert_parallel import resources

        item, config, provider, resource, parallelism = _expert_setup(
            options, precision
        )
        model = resources.model(item, config, provider, precision)
        provider_options = dict(
            resources={"experts": resource}, operator_preparation=()
        )
    else:
        config = model_config(options)
        with torch.device("meta"):
            model = (
                entry(options["model"]).model_class("mlops")(config).to(torch.bfloat16)
            )
        parallelism = Parallelism.dp(ranks_per_replica=options["ranks"])
        provider_options = dict(
            operator_preparation=operator_preparations("mlops", "bfloat16")
        )
    if inputs(options, rank=rank).data.vocab_size > config.vocab_size:
        raise ValueError("corpus vocabulary exceeds model vocabulary")
    optimizer = AdamW(
        model.parameters(),
        lr=torch.tensor(options["learning_rate"]),
        **precision.optimizer_arguments(),
    )
    yield BuildSetup(
        model,
        optimizer,
        normalized_next_token_prediction,
        dict(
            parallelism=parallelism,
            **provider_options,
            **precision.plan_arguments(),
            **build_options(options, options["sequences_per_example"]),
        ),
    )


@contextmanager
def initializer(options, metadata, *, rank=0):
    from workloads.recipes.text.initialization import Initialize

    initialize = Initialize("qwen3moe", model_config(options), seed=options["seed"])
    if options["parallelism"] == "ep":
        from workloads.recipes.text.expert_parallel.initialization import (
            Initialize as ExpertInitialize,
        )
        from workloads.recipes.text.precision import Precision

        _, _, provider, _, _ = _expert_setup(options, Precision(grad_dtype="float32"))
        initialize = ExpertInitialize(
            initialize,
            expert_rank=rank,
            std=provider.init_std,
            router_dtype=provider.router_dtype,
        )
    yield initialize
