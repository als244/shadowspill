"""Model choices and initialization for phase quickstart; no live process groups."""

from dataclasses import replace

import torch

from shadowspill import Parallelism
from shadowspill.training._model import build_optimizer
from workloads.full_model import build_model, full_model_objective, spec
from workloads.lora import configure_lora
from workloads.models.shared.training import LEARNING_RATE

from ..precision import Precision
from .initialize import Initialize
from .inputs import candidates


def experiment(
    *,
    device,
    model="llama3-tiny",
    implementation="mlops",
    model_config=None,
    sequence_length=None,
    sequences_per_step=None,
    sequences_per_example=None,
    min_tokens_per_example=None,
    max_tokens_per_example=None,
    seed=0,
    precision=None,
    lora=None,
    ranks=1,
    num_replicas=1,
    logical_rank=0,
    weights="sharded",
    gradients="sharded",
    optimizer="sharded",
):
    precision = Precision(**(precision or {}))
    base = spec(model, implementation, **(model_config or {}))
    manifest = replace(
        base,
        sequence_length=base.sequence_length
        if sequence_length is None
        else sequence_length,
        sequences_per_microbatch=(
            base.sequences_per_microbatch * base.accumulation_count
        )
        if sequences_per_step is None
        else sequences_per_step,
        accumulation_count=1,
        model_dtype=precision.model_dtype,
        master_dtype=precision.master_dtype or "none",
        grad_dtype=precision.grad_dtype or precision.model_dtype,
        opt_state_dtype=(precision.master_dtype or precision.model_dtype)
        if precision.opt_state_dtype == "parameter"
        else precision.opt_state_dtype or base.opt_state_dtype,
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
    initializer = Initialize(manifest, seed=seed)
    inputs, skipped = candidates(
        manifest,
        size=sequences_per_example,
        minimum=min_tokens_per_example,
        maximum=max_tokens_per_example,
        seed=seed,
        logical_rank=logical_rank,
    )

    def model_factory():
        precision.apply()
        result = build_model(manifest)
        return result if lora is None else configure_lora(result, **lora)

    def objective(model, *args):
        result = full_model_objective(manifest, model, *args)
        loss, metrics = (
            result
            if isinstance(result, tuple)
            else (result, {"head_loss": result.detach()})
        )
        return loss / parallelism.world_size, {
            key: value / parallelism.world_size for key, value in metrics.items()
        }

    def make_optimizer(parameters):
        return build_optimizer(
            precision.optimizer(manifest.dtypes.optimizer()),
            parameters,
            {"lr": torch.tensor(LEARNING_RATE)},
        )

    return {
        "model_factory": model_factory,
        "optimizer": make_optimizer,
        "objective": objective,
        "candidates": inputs,
        "fill_tensor": initializer,
        "build_options": {**precision.plan_arguments(), "parallelism": parallelism},
        "hyperparams": {"optimizer_group.0.lr": LEARNING_RATE},
        "metric_reducer": metrics,
        "units_per_step": manifest.tokens_per_step * parallelism.world_size,
        "unit_label": "tokens",
        "metadata": {
            **manifest.as_dict(),
            "seed": seed,
            "lora": lora,
            "parallelism": parallelism.to_dict(),
            "units_scope": "global",
            "skipped": skipped,
        },
    }


def metrics(observed):
    return {
        "loss": sum(
            value.get("head_loss", loss) if isinstance(value, dict) else loss
            for loss, value in zip(observed.losses, observed.metrics, strict=True)
        )
    }
