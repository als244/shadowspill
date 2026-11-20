"""An ordinary model and its bounded, reproducible random initializer."""

from contextlib import contextmanager
from dataclasses import dataclass

from shadowspill.training.components import ModelInitialization
from shadowspill_client.extensions import Capacity, InputSchema, TensorField


@dataclass(frozen=True)
class Config:
    width: int = 4

    def __post_init__(self):
        if type(self.width) is not int or self.width < 1:
            raise ValueError("width must be a positive integer")


def create(config):
    import torch

    class Regressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(config.width, 1)

        def forward(self, features):
            return self.projection(features).squeeze(-1)

    return Regressor()


def inputs(config):
    return InputSchema(
        fields=[TensorField(("features",), "float32", (config.width,))],
        capacities=[Capacity("max_examples", "examples", "count", scheduled=True)],
    )


def initialization(config, context):
    if context.weights.kind != "random":
        raise ValueError("this example initializes from Weights.random(seed=...)")

    @contextmanager
    def open_loader(runtime):
        import torch

        from shadowspill.checkpoint.initialization import fill_generated

        shapes = {"projection.weight": (1, config.width), "projection.bias": (1,)}

        def sample(name, shape, start, stop, generator):
            if shapes.get(name) != shape:
                raise ValueError("unknown parameter or shape")
            bound = config.width**-0.5
            return torch.empty(stop - start).uniform_(
                -bound, bound, generator=generator
            )

        def fill(request, destination):
            fill_generated(
                request,
                destination,
                sample=sample,
                seed=context.weights.seed,
                block_elements=256,
            )

        yield fill

    return ModelInitialization(
        peak_bytes=1 << 20, chunk_bytes=64 << 10, open=open_loader
    )
