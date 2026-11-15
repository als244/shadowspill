"""Two-layer stochastic model with exposed noise for an independent oracle."""

from contextlib import contextmanager
from dataclasses import replace

from shadowspill.training.definitions import ModelInitialization
from shadowspill_client.protocol.definitions import TensorOutput


def create(config):
    import torch

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 8)
            self.head = torch.nn.Linear(8, config.num_classes)

        def forward(self, pixels):
            hidden = torch.tanh(self.encoder(pixels.flatten(start_dim=1)))
            noise = torch.rand_like(hidden) + 0.5
            return self.head(hidden * noise), noise

    return Network()


def initialization(config, context):
    if context.weights.kind != "random":
        raise ValueError("the stochastic fixture requires explicit random weights")

    @contextmanager
    def open_loader(runtime):
        import torch

        from shadowspill.checkpoint.initialization import fill_generated

        shapes = {
            "encoder.weight": (8, 4),
            "encoder.bias": (8,),
            "head.weight": (config.num_classes, 8),
            "head.bias": (config.num_classes,),
        }

        def sample(name, shape, start, stop, generator):
            if shapes.get(name) != shape:
                raise ValueError("unrecognized stochastic parameter geometry")
            return torch.empty(stop - start).uniform_(-0.5, 0.5, generator=generator)

        def fill(request, destination):
            fill_generated(
                request,
                destination,
                sample=sample,
                seed=context.weights.seed,
                block_elements=8,
            )

        yield fill

    return ModelInitialization(peak_bytes=1 << 20, chunk_bytes=32, open=open_loader)


def outputs(ordinary):
    return {
        name: replace(
            output,
            tensors={
                "prediction": TensorOutput("float32", (3,), "datum"),
                "noise": TensorOutput("float32", (8,), "datum"),
            },
        )
        for name, output in ordinary.items()
    }


def predictions(model, inputs, loss_inputs, loss_params):
    prediction, noise = model(**inputs)
    return {"prediction": prediction, "noise": noise}


def loss(model, inputs, loss_inputs, loss_params):
    import torch

    from shadowspill.pytorch.contracts import ObjectiveResult

    prediction, noise = model(**inputs)
    logits = prediction / loss_params["temperature"]
    valid = loss_inputs["valid"]
    labels = torch.where(valid, loss_inputs["class_id"], 0)
    values = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weighted = torch.where(valid, values * loss_inputs["weights"], 0)
    return ObjectiveResult(
        weighted.sum(),
        {
            "valid_examples": valid.sum(),
            "prediction": prediction,
            "noise": noise,
        },
    )


def bind(config, selected):
    return loss if selected == "metrics" else predictions
