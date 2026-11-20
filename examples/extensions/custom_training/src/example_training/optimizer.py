"""Wrap an existing optimizer; the engine captures and executes its update."""

from shadowspill_client.extensions import ControlSchema, ScalarControl

CONTROLS = ControlSchema({"lr": ScalarControl(0.01, minimum=0)})
BINDINGS = {"lr": ("optimizer_group.0.lr",)}


def create(model, config, controls):
    import torch

    return torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=controls["lr"],
        foreach=False,
    )
