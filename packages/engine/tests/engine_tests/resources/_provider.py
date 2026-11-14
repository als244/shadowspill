"""A generic fixed lookup buffer, shared by two custom-op calls in a model."""

import json
import os
from pathlib import Path

import torch

_active = {}


class Buffer:
    def __init__(self, name, device, options):
        if name in _active:
            raise ValueError("resource opened twice")
        self.name, self.options = name, options
        self.values = torch.full((options["elements"],), 2.0, device=device)
        _active[name] = self
        self.record("open")

    def record(self, event):
        with Path(self.options["events"]).open("a") as file:
            file.write(
                json.dumps({"pid": os.getpid(), "name": self.name, "event": event})
                + "\n"
            )

    def close(self):
        _active.pop(self.name)
        self.values = None
        self.record("close")


def initialize(*, name, device, groups, options):
    assert not groups
    if options.get("fail"):
        raise ValueError("intentional initialization failure")
    return Buffer(name, device, options)


@torch.library.custom_op("ss_resource_test::scale", mutates_args=())
def scale(value: torch.Tensor, resource: str) -> torch.Tensor:
    return value * _active[resource].values[0]


@scale.register_fake
def fake(value, resource):
    return torch.empty_like(value)


def setup_context(ctx, inputs, output):
    ctx.resource = inputs[1]


def backward(ctx, gradient):
    return scale(gradient, ctx.resource), None


scale.register_autograd(backward, setup_context=setup_context)
