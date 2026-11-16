"""Explicit Qwen cross entropy plus its existing physical-microbatch balancing."""

from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from math import isfinite
from pathlib import Path

from shadowspill.training.definitions import ObjectiveDefinition
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.definitions import DefinitionSource, Participation

from . import token


@dataclass(frozen=True)
class Config(token.TokenConfig):
    router_aux_loss_coef: float

    def __post_init__(self):
        super().__post_init__()
        value = self.router_aux_loss_coef
        if type(value) not in (int, float) or not isfinite(value) or value < 0:
            raise ValueError("router_aux_loss_coef must be finite and nonnegative")


def configure(values, model):
    values = token.configure(values, model)
    return {
        "router_aux_loss_coef": model.config["architecture"]["router_aux_loss_coef"],
        **values,
    }


def controls(config):
    return ControlSchema(
        {"router_aux_loss_coef": ScalarControl(config.router_aux_loss_coef, minimum=0)}
    )


def outputs(config):
    return {
        name: replace(
            output,
            controls=("router_aux_loss_coef",),
            metrics={**output.metrics, "ce_sum": "sum", "auxiliary_sum": "sum"},
        )
        for name, output in token.outputs(config, policy=False).items()
    }


def compute(model, inputs, loss_inputs, loss_params, *, tensors=False, packed=False):
    import torch

    from shadowspill.pytorch.contracts import ObjectiveResult
    from shadowspill_workloads.objectives.token import cross_entropy

    valid = loss_inputs["loss_mask"]
    targets = torch.where(valid, loss_inputs["target_tokens"], 0)
    current, auxiliary = model.target_logprobs_with_auxiliary(targets=targets, **inputs)
    current = current.float()
    ce = cross_entropy(current, valid, loss_inputs["weights"])
    count = valid.sum()
    # Preserve Qwen's existing sum reduction: all physical tokens participate
    # in routing statistics, scaled by the supervised count, not CE weights.
    auxiliary_sum = auxiliary * count
    loss = ce + loss_params["router_aux_loss_coef"] * auxiliary_sum
    values = {
        "loss_sum": loss.detach(),
        "ce_sum": ce.detach(),
        "auxiliary_sum": auxiliary_sum.detach(),
        "valid_tokens": count,
    }
    if tensors:
        output = torch.where(valid, current, 0.0)
        values["logprobs"] = output.reshape(-1) if packed else output
    return ObjectiveResult(loss, values)


def bind(config, key):
    if key.operation == "forward" and key.outputs == "logprobs":
        return partial(token.forward_logprobs, packed=config.batching == "pack")
    return partial(
        compute, tensors=key.outputs == "logprobs", packed=config.batching == "pack"
    )


def register(registry):
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for relative in (
        "definitions/moe_objective.py",
        "definitions/token.py",
        "objectives/token.py",
    ):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    source = DefinitionSource(
        "shadowspill-workloads", version("shadowspill-workloads"), digest.hexdigest()
    )
    registry.add_objective(
        ObjectiveDefinition(
            "shadowspill/qwen_moe_cross_entropy@1",
            compute,
            partial(token.fields, policy=False),
            config=Config,
            configure=configure,
            controls=controls,
            outputs=outputs,
            bind=bind,
            requires=("target_logprobs", "qwen_router_auxiliary"),
            participation=Participation(("loss_mask",)),
            reduction="custom",
            physical_batch_loss=True,
        ),
        source=source,
    )
