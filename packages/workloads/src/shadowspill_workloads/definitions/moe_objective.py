"""Reusable router balancing, independent of a model family or primary loss."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from shadowspill.training.definitions import ObjectiveDefinition
from shadowspill_client.protocol.definitions import (
    DefinitionSource,
    OutputSet,
    Participation,
)
from shadowspill_client.protocol.inputs import InputSchema, TensorField

from . import token
from .token_evaluation import evaluation


@dataclass(frozen=True)
class Config(token.TokenConfig):
    include_padding: bool = False
    layer_reduction: str = "combined"
    normalize_top_k: bool = False
    participation_field: str = "loss_mask"
    weight_field: str | None = "weights"

    def __post_init__(self):
        super().__post_init__()
        if (
            type(self.include_padding) is not bool
            or type(self.normalize_top_k) is not bool
        ):
            raise TypeError("router padding and top-k normalization must be boolean")
        if self.layer_reduction not in ("combined", "sum", "mean"):
            raise ValueError("router layer_reduction must be combined, sum or mean")
        if self.participation_field not in ("loss_mask", "response_mask"):
            raise ValueError(
                "router participation_field must be loss_mask or response_mask"
            )
        if self.weight_field is not None and (
            not isinstance(self.weight_field, str) or not self.weight_field
        ):
            raise ValueError("router weight_field must name a field or be None")


def fields(config):
    geometry = dict(
        shape=(config.alignment,),
        batching=config.batching,
        pack_axis=0 if config.batching == "pack" else None,
    )
    policy = config.participation_field == "response_mask"
    result = [
        TensorField(
            (config.participation_field,),
            "bool",
            required=policy,
            default=None if policy else True,
            padding=False,
            **geometry,
        )
    ]
    if config.weight_field is not None:
        result.append(
            TensorField(
                (config.weight_field,),
                "float32",
                required=False,
                default=1.0,
                validity=(config.participation_field,),
                **geometry,
            )
        )
    if not config.include_padding:
        result.append(
            TensorField(
                ("router_mask",),
                "bool",
                required=False,
                default=True,
                padding=False,
                **geometry,
            )
        )
    return InputSchema(result)


def outputs(config):
    return {
        "metrics": OutputSet(
            model_inputs=config.model_inputs,
            loss_inputs=tuple(f.path for f in fields(config).fields),
            metrics={"loss_sum": "sum", "valid_tokens": "sum"},
        )
    }


def compute(evaluated, inputs, loss_inputs, loss_params, *, config):
    import torch

    from shadowspill.pytorch.contracts import ObjectiveResult
    from shadowspill_workloads.objectives.moe import router_balance

    auxiliary = router_balance(
        evaluated["router_counts"],
        evaluated["router_probability_sum"],
        evaluated["router_rows"],
        top_k=evaluated["router_top_k"],
        layer_reduction=config.layer_reduction,
        normalize_top_k=config.normalize_top_k,
    )
    # Share text_batch's global token normalization by default. The router
    # statistic still uses its own mask and is computed per physical microbatch.
    valid = loss_inputs[config.participation_field]
    count = valid.sum()
    mass = (
        count
        if config.weight_field is None
        else torch.where(valid, loss_inputs[config.weight_field], 0.0).sum()
    )
    loss = auxiliary * mass
    return ObjectiveResult(loss, {"loss_sum": loss.detach(), "valid_tokens": count})


def bind(config, key):
    from functools import partial

    return partial(compute, config=config)


def evaluate(config, key):
    return evaluation(
        {"router_statistics": True, "include_padding": config.include_padding}
    )


def register(registry):
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for relative in (
        "definitions/moe_objective.py",
        "definitions/token.py",
        "definitions/token_evaluation.py",
        "objectives/moe.py",
    ):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    source = DefinitionSource(
        "shadowspill-workloads", version("shadowspill-workloads"), digest.hexdigest()
    )
    registry.add_objective(
        ObjectiveDefinition(
            "shadowspill/moe_balance@1",
            compute,
            fields,
            config=Config,
            configure=token.configure,
            outputs=outputs,
            bind=bind,
            evaluate=evaluate,
            requires=("router_statistics",),
            participation=lambda config: Participation((config.participation_field,)),
            reduction="custom",
            padding_safe=lambda config: not config.include_padding,
            physical_batch_loss=lambda config: config.include_padding,
        ),
        source=source,
    )
