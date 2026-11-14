"""Same classifier with an explicit provider buffer and an ordinary eager oracle."""

from dataclasses import asdict, dataclass

from shadowspill.resources import Resource
from shadowspill.training.definitions import ModelBuild


@dataclass(frozen=True)
class Config:
    num_classes: int = 3
    events: str = ""
    resource: bool = False
    capacity: int = 0


def create(config):
    import torch

    from engine_tests.resources._provider import scale

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(4, config.num_classes)

        def forward(self, pixels):
            values = (
                scale(scale(pixels, "lookup"), "lookup")
                if config.resource
                else pixels * 4
            )
            return self.head(values.flatten(start_dim=1))

    return Classifier()


def describe(config, context):
    capacity = max(layout.buffer.max_datums for layout in context.family)
    return ModelBuild(
        {**asdict(config), "resource": True, "capacity": capacity},
        resources={
            "lookup": Resource(
                "engine_tests.resources._provider:initialize",
                {"events": config.events, "elements": capacity * 65536},
            )
        },
    )
