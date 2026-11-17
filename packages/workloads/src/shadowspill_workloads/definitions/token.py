"""Framework-free descriptions of the supplied token objectives."""

from dataclasses import dataclass
from functools import partial

from shadowspill.training.definitions import ObjectiveDefinition
from shadowspill_client._records import plain
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.definitions import (
    OutputSet,
    Participation,
    TensorOutput,
)
from shadowspill_client.protocol.inputs import InputSchema, TensorField

from .source import definition_source
from .token_evaluation import token_evaluation


@dataclass(frozen=True)
class TokenConfig:
    vocab_size: int
    alignment: str
    batching: str
    model_inputs: tuple[tuple[str, ...], ...]

    def __post_init__(self):
        if type(self.vocab_size) is not int or self.vocab_size < 1:
            raise ValueError("token objectives require a positive vocabulary size")
        if not isinstance(self.alignment, str) or not self.alignment:
            raise ValueError("token objectives require a declared position alignment")
        if self.batching not in ("pack", "stack"):
            raise ValueError("token objectives require packed or stacked positions")
        object.__setattr__(
            self, "model_inputs", tuple(tuple(p) for p in self.model_inputs)
        )


@dataclass(frozen=True)
class PolicyConfig(TokenConfig):
    reference_penalty: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.reference_penalty not in (None, "k3"):
            raise ValueError("reference_penalty must be absent or k3")


def configure(values, model):
    tokens = next((f for f in model.inputs.fields if f.path == ("tokens",)), None)
    if (
        tokens is None
        or tokens.dtype not in ("int32", "int64")
        or len(tokens.shape) != 1
        or not isinstance(tokens.shape[0], str)
        or tokens.batching not in ("pack", "stack")
        or tokens.minimum != 0
        or tokens.maximum is None
        or int(tokens.maximum) != tokens.maximum
        or tokens.maximum < 0
    ):
        raise ValueError(
            "token objectives require a bounded, aligned integer tokens field"
        )
    inferred = {
        "vocab_size": int(tokens.maximum) + 1,
        "alignment": tokens.shape[0],
        "batching": tokens.batching,
        "model_inputs": tuple(tuple(f.path) for f in model.inputs.fields),
    }
    for key, value in inferred.items():
        if key in values and plain(values[key]) != plain(value):
            raise ValueError(f"objective {key} differs from the resolved model")
    return {**values, **inferred}


def fields(config, *, policy):
    mask = "response_mask" if policy else "loss_mask"
    geometry = dict(
        shape=(config.alignment,),
        batching=config.batching,
        pack_axis=0 if config.batching == "pack" else None,
    )
    values = [
        TensorField(
            ("target_tokens",),
            "int64",
            minimum=0,
            maximum=config.vocab_size - 1,
            validity=(mask,),
            **geometry,
        ),
        TensorField(
            (mask,),
            "bool",
            required=policy,
            default=None if policy else True,
            padding=False,
            **geometry,
        ),
        TensorField(
            ("weights",),
            "float32",
            required=False,
            default=1.0,
            validity=(mask,),
            **geometry,
        ),
    ]
    if policy:
        values.extend(
            TensorField((name,), "float32", validity=(mask,), **geometry)
            for name in ("old_logprobs", "advantages")
        )
        if config.reference_penalty == "k3":
            values.append(
                TensorField(
                    ("reference_logprobs",), "float32", validity=(mask,), **geometry
                )
            )
    return InputSchema(values)


def controls(config):
    fields = {
        "clip_ratio": ScalarControl(0.2, minimum=0, maximum=1, maximum_inclusive=False)
    }
    if config.reference_penalty == "k3":
        fields["reference_beta"] = ScalarControl(0.0, minimum=0)
    return ControlSchema(fields)


def outputs(config, *, policy):
    loss_fields = fields(config, policy=policy)
    required_controls = tuple(controls(config).fields) if policy else ()
    metrics = dict(loss_sum="sum", valid_tokens="sum")
    common = dict(
        model_inputs=config.model_inputs,
        loss_inputs=tuple(f.path for f in loss_fields.fields),
        controls=required_controls,
        metrics=metrics,
    )
    tensors = {
        "logprobs": TensorOutput(
            "float32",
            (config.alignment,),
            "aligned",
            alignment=config.alignment,
            source=("loss_inputs", "target_tokens"),
        )
    }
    return {
        "metrics": OutputSet(**common),
        "logprobs": OutputSet(
            **common,
            tensors=tensors,
            forward=OutputSet(
                model_inputs=config.model_inputs,
                loss_inputs=(("target_tokens",),),
                tensors=tensors,
            ),
        ),
    }


def forward_logprobs(evaluated, inputs, loss_inputs, loss_params, *, packed=False):
    from shadowspill_workloads.objectives.head.execution import logprobs

    values = logprobs(evaluated)
    return {"logprobs": values.reshape(-1) if packed else values}


def compute(
    evaluated,
    inputs,
    loss_inputs,
    loss_params,
    *,
    policy=False,
    tensors=False,
    packed=False,
):
    import torch

    from shadowspill.pytorch.contracts import ObjectiveResult
    from shadowspill_workloads.objectives.head import execution

    valid = loss_inputs["response_mask" if policy else "loss_mask"]
    loss, current = execution.loss(evaluated, loss_inputs, loss_params, policy=policy)
    values = {"loss_sum": loss.detach(), "valid_tokens": valid.sum()}
    if tensors:
        output = torch.where(valid, current, 0.0)
        values["logprobs"] = output.reshape(-1) if packed else output
    return ObjectiveResult(loss, values)


def bind(config, key, *, policy):
    if key.operation == "forward" and key.outputs == "logprobs":
        return partial(forward_logprobs, packed=config.batching == "pack")
    return partial(
        compute,
        policy=policy,
        tensors=key.outputs == "logprobs",
        packed=config.batching == "pack",
    )


def register(registry):
    source = definition_source(
        ("definitions/token.py", "definitions/token_evaluation.py", "objectives/head"),
        providers=("mlops",),
    )
    for name, policy in (("cross_entropy", False), ("clipped_policy", True)):
        registry.add_objective(
            ObjectiveDefinition(
                id=f"shadowspill/{name}@1",
                compute=partial(compute, policy=policy),
                config=PolicyConfig if policy else TokenConfig,
                configure=configure,
                inputs=partial(fields, policy=policy),
                controls=controls if policy else ControlSchema({}),
                outputs=partial(outputs, policy=policy),
                bind=partial(bind, policy=policy),
                evaluate=partial(token_evaluation, policy=policy),
                requires=("target_logprobs",),
                participation=Participation(
                    ("response_mask" if policy else "loss_mask",)
                ),
                padding_safe=True,
            ),
            source=source,
        )
