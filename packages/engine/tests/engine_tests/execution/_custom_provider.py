"""An importable stateless provider, separate from its caller's model."""

import torch


@torch.library.custom_op("shadowspill_fixture::square", mutates_args=())
def square(x: torch.Tensor) -> torch.Tensor:
    return x.square()


@square.register_fake
def fake_square(x):
    return torch.empty_like(x)


def setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def backward(ctx, dy):
    (x,) = ctx.saved_tensors
    return 2 * x * dy


square.register_autograd(backward, setup_context=setup_context)
