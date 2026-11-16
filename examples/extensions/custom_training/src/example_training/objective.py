"""A model-independent weighted regression objective with explicit padding."""

from shadowspill_client.protocol.definitions import OutputSet, Participation
from shadowspill_client.protocol.inputs import InputSchema, TensorField

INPUTS = InputSchema(
    fields=[
        TensorField(("target",), "float32"),
        TensorField(("weight",), "float32", required=False, default=1.0),
        TensorField(("valid",), "bool", required=False, default=True, padding=False),
    ]
)
PARTICIPATION = Participation(("valid",))
OUTPUTS = {
    "metrics": OutputSet(
        model_inputs=(("features",),),
        loss_inputs=(("target",), ("weight",), ("valid",)),
        metrics={"valid_examples": "sum"},
    )
}


def compute(model, inputs, loss_inputs, loss_params):
    import torch

    from shadowspill.pytorch.contracts import ObjectiveResult

    prediction = model(**inputs)
    valid = loss_inputs["valid"]
    error = prediction - torch.where(valid, loss_inputs["target"], 0)
    weight = torch.where(valid, loss_inputs["weight"], 0)
    loss = (error.square() * weight).sum()
    return ObjectiveResult(loss, {"valid_examples": valid.sum()})
