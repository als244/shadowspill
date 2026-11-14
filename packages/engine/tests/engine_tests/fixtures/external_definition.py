"""A non-text integration using only public definition/capture interfaces."""

from dataclasses import dataclass

from shadowspill.training.definitions import ModelDefinition, ObjectiveDefinition
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl
from shadowspill_client.protocol.definitions import (
    DefinitionSource,
    OutputSet,
    Participation,
)
from shadowspill_client.protocol.inputs import Capacity, InputSchema, TensorField

SOURCE = DefinitionSource("external-classifier-fixture", "1", "a" * 64)


@dataclass(frozen=True)
class ClassifierConfig:
    num_classes: int = 3

    def __post_init__(self):
        if type(self.num_classes) is not int or self.num_classes < 2:
            raise ValueError("num_classes must be at least two")


def create(config):
    import torch

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, config.num_classes)

        def forward(self, pixels):
            return self.head(pixels.flatten(start_dim=1))

    return Classifier()


def classification_loss(model, inputs, loss_inputs, loss_params):
    import torch

    from shadowspill.pytorch.contracts import ObjectiveResult

    logits = model(**inputs) / loss_params["temperature"]
    valid = loss_inputs["valid"]
    labels = torch.where(valid, loss_inputs["class_id"], 0)
    per_item = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weighted = torch.where(valid, per_item * loss_inputs["weights"], 0)
    return ObjectiveResult(weighted.sum(), {"valid_examples": valid.sum()})


def register(registry):
    registry.add_model(
        ModelDefinition(
            id="external/image_classifier@1",
            config=ClassifierConfig,
            create=create,
            inputs=InputSchema(
                [
                    TensorField(("pixels",), "float32", (1, 2, 2)),
                ],
                [Capacity("max_examples", "examples", "count", scheduled=True)],
            ),
            capabilities=("class_logits",),
        ),
        source=SOURCE,
    )
    registry.add_objective(
        ObjectiveDefinition(
            id="external/classification@1",
            compute=classification_loss,
            inputs=InputSchema(
                [
                    TensorField(("class_id",), "int64", minimum=0),
                    TensorField(
                        ("valid",), "bool", required=False, default=True, padding=False
                    ),
                    TensorField(("weights",), "float32", required=False, default=1.0),
                ]
            ),
            controls=ControlSchema(
                {"temperature": ScalarControl(1.0, minimum=0, minimum_inclusive=False)}
            ),
            outputs={
                "metrics": OutputSet(
                    model_inputs=(("pixels",),),
                    loss_inputs=(("class_id",), ("valid",), ("weights",)),
                    controls=("temperature",),
                    metrics={"valid_examples": "sum"},
                )
            },
            requires=("class_logits",),
            participation=Participation(("valid",)),
        ),
        source=SOURCE,
    )
