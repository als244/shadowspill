"""One exact-token EP geometry using the common saved-phase text recipe."""

from functools import partial

from workloads.models import catalog

from ..benchmark.factory import experiment as text_experiment
from ..precision import Precision
from . import resources
from .initialization import Initialize


def experiment(
    *,
    device,
    model="qwen3moe-30b-a3b",
    ranks=8,
    num_replicas=1,
    logical_rank=0,
    sequence_length=4096,
    sequences_per_step=64,
    sequences_per_example=16,
    model_dtype="bfloat16",
    master_dtype="float32",
    grad_dtype="float32",
    opt_state_dtype="float32",
    seed=0,
    config_overrides=None,
):
    """Declare local tensors and portable provider resources; no runtime setup.

    Expert slices are local; dense parameters and routers are replicas. The
    optimizer shards dense state, and reductions SUM each shared gradient once.
    All resource participants are required during build finalization/profiling.
    """
    if model not in catalog.expert_parallel_names():
        raise ValueError(f"model must be one of {catalog.expert_parallel_names()}")
    if type(ranks) is not int or ranks < 2:
        raise ValueError("EP needs at least two ranks per replica")
    if model_dtype != "bfloat16":
        raise ValueError("Quack EP requires BF16 activations and compute weights")
    if grad_dtype not in {"float32", "bfloat16"}:
        raise ValueError("Quack EP gradients must be float32 or bfloat16")
    for name, value in (
        ("sequence_length", sequence_length),
        ("sequences_per_step", sequences_per_step),
        ("sequences_per_example", sequences_per_example),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    entry = catalog.entry(model)
    changes = {**(config_overrides or {}), "max_seq_len": sequence_length}
    config = entry.config(**changes)
    if config.n_layers < 1 or config.n_experts % ranks:
        raise ValueError("EP requires layers and experts evenly divisible by ranks")
    precision = Precision(
        model_dtype=model_dtype,
        master_dtype=master_dtype,
        grad_dtype=grad_dtype,
        opt_state_dtype=opt_state_dtype,
    )
    setup = text_experiment(
        device=device,
        model=model,
        model_config=changes,
        sequence_length=sequence_length,
        sequences_per_step=sequences_per_step,
        sequences_per_example=sequences_per_example,
        seed=seed,
        precision=dict(
            model_dtype=model_dtype,
            master_dtype=master_dtype,
            grad_dtype=grad_dtype,
            opt_state_dtype=opt_state_dtype,
        ),
        ranks=ranks,
        num_replicas=num_replicas,
        logical_rank=logical_rank,
        weights="replicated",
        gradients="replicated",
        optimizer="sharded",
    )
    parallelism = resources.ownership(ranks, num_replicas)
    options, resource = resources.description(
        config,
        entry.family,
        precision,
        parallelism,
        sequence_length * sequences_per_example,
    )
    setup["model_factory"] = partial(resources.model, entry, config, options, precision)
    setup["fill_tensor"] = Initialize(
        setup["fill_tensor"],
        expert_rank=logical_rank % ranks,
        std=options.init_std,
    )
    setup["build_options"].update(
        parallelism=parallelism,
        resources={"experts": resource},
    )
    setup["metadata"].update(
        parallelism=parallelism.to_dict(),
        expert_parallel_size=ranks,
        expert_backend="quack",
        gemm_tuned=False,
    )
    return setup
