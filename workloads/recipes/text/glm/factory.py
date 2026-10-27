"""GLM construction, objective and data for the ordinary phase quickstart."""

from contextlib import contextmanager, nullcontext
from functools import partial

import mlops
import torch

from shadowspill import Parallelism
from workloads.recipes.text.precision import Precision

from .initialize import Initialize
from .inputs import candidates
from .model import declare
from .partition import GLMStages


def experiment(
    architecture,
    config_type,
    checkpoint_type,
    model_type,
    compiler_context=nullcontext,
    /,
    *,
    device,
    checkpoint=None,
    initialization="random",
    tiny=False,
    sequence_length=1024,
    sequences_per_step=8,
    sequences_per_example=(1, 2, 4),
    lora_rank=32,
    lora_alpha=32.0,
    lora_dtype="bfloat16",
    router_weight_dtype=None,
    lora_head=False,
    grad_dtype="float32",
    master_dtype=None,
    opt_state_dtype="bfloat16",
    opt_state_rounding=None,
    gemm_precision="bf16",
    expert_storage=None,
    layer_limit=None,
    head_chunk_size=1024,
    partition=None,
    seed=0,
    ranks=1,
    num_replicas=1,
    logical_rank=0,
    weights="sharded",
    gradients="sharded",
    optimizer="sharded",
):
    """Use metadata from a local checkpoint, or a tiny meta fixture.

    Random initialization is the default; checkpoint loading is opt-in. Both
    fill bounded CPU spill chunks and retain declared tensor dtypes. Checkpoint
    loading currently requires direct expert copies (BF16 or block FP8), without
    format conversion. No CUDA allocation or live process group is needed to
    construct the recipe. The normal quickstart runner owns all four phases.
    """
    dtypes = ("float32", "bfloat16", "float16")
    if lora_rank <= 0 or head_chunk_size <= 0:
        raise ValueError("LoRA rank and head chunk size must be positive")
    if lora_dtype not in dtypes or grad_dtype not in dtypes:
        raise ValueError("LoRA/gradient dtypes must be float32, bfloat16 or float16")
    if master_dtype not in (None, *dtypes):
        raise ValueError("master_dtype must be None, float32, bfloat16 or float16")
    if opt_state_dtype not in (*dtypes, "parameter"):
        raise ValueError("invalid optimizer state dtype")
    if partition not in (None, "glm_stages"):
        raise ValueError("partition must be None or 'glm_stages'")
    rounding = opt_state_rounding or (
        "stochastic" if opt_state_dtype == "bfloat16" else "nearest"
    )
    precision = Precision(
        grad_dtype=grad_dtype,
        master_dtype=master_dtype,
        opt_state_dtype=opt_state_dtype,
        opt_state_rounding=rounding,
    )
    parallelism = Parallelism.dp(
        ranks_per_replica=ranks,
        num_replicas=num_replicas,
        weights=weights,
        gradients=gradients,
        optimizer=optimizer,
    )
    if not 0 <= logical_rank < parallelism.world_size:
        raise ValueError("logical_rank is outside the declared parallelism")
    model, source = declare(
        config_type,
        checkpoint_type,
        model_type,
        checkpoint=checkpoint,
        tiny=tiny,
        initialization=initialization,
        gemm_precision=gemm_precision,
        expert_storage=expert_storage,
        layer_limit=layer_limit,
        router_weight_dtype=router_weight_dtype,
        lora=dict(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_factor_dtype=getattr(torch, lora_dtype),
            lora_head=lora_head,
        ),
    )
    try:
        initialize = Initialize(
            model, source=source, initialization=initialization, seed=seed
        )
        inputs = candidates(
            model.config,
            sequence_length=sequence_length,
            sequences_per_step=sequences_per_step,
            sequences_per_example=sequences_per_example,
            logical_rank=logical_rank,
            seed=seed,
        )
    except BaseException:
        if source is not None:
            source.close()
        raise

    @contextmanager
    def resources():
        try:
            with compiler_context():
                yield
        finally:
            if source is not None:
                source.close()

    def model_factory():
        precision.apply()
        return model

    total = sequence_length * sequences_per_step * parallelism.world_size

    def objective(model, *args):
        return model.loss(*args, chunk_size=head_chunk_size, reduction="sum") / total

    return {
        "model_factory": model_factory,
        "objective": objective,
        "optimizer": partial(
            mlops.optim.AdamW, lr=torch.tensor(1e-4), **precision.optimizer_arguments()
        ),
        "fill_tensor": initialize,
        "candidates": inputs,
        "build_options": {
            **precision.plan_arguments(),
            "parallelism": parallelism,
            **({"partition": GLMStages()} if partition == "glm_stages" else {}),
        },
        "hyperparams": {"optimizer_group.0.lr": 1e-4},
        "units_per_step": total,
        "unit_label": "tokens",
        "context": resources,
        "metadata": {
            "architecture": architecture,
            "checkpoint": None if source is None else str(source.directory),
            "tiny": tiny,
            "initialization": initialization,
            "initialization_device": "cpu",
            "seed": seed,
            "sequence_length": sequence_length,
            "sequences_per_step": sequences_per_step,
            "layer_limit": layer_limit,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dtype": lora_dtype,
            "grad_dtype": grad_dtype,
            "master_dtype": master_dtype,
            "optimizer": "mlops.optim.AdamW",
            "opt_state_dtype": opt_state_dtype,
            "opt_state_rounding": rounding,
            "gemm_precision": gemm_precision,
            "expert_storage": model.config.expert_storage,
            "partition": "GLMStages()" if partition == "glm_stages" else "auto",
            "parallelism": parallelism.to_dict(),
            "units_scope": "global",
        },
    }
