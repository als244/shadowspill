"""Apply the same bounded tensor initializer to an ordinary reference model."""

from contextlib import ExitStack, contextmanager

import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.training import config, reset_parameters


@contextmanager
def initializer(options):
    with ExitStack() as stack:
        fill = config.resolve(options.get("initialize"))
        if hasattr(fill, "__enter__"):
            fill = stack.enter_context(fill)

        @torch.no_grad()
        def initialize(model):
            if fill is None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(options.get("seed", 0))
                    reset_parameters(model)
                return
            for name, tensor in (*model.named_parameters(), *model.named_buffers()):
                if tensor.device.type != "cpu" or not tensor.is_contiguous():
                    raise ValueError(
                        "bounded reference initialization requires contiguous CPU "
                        f"storage: {name}"
                    )
                flat = tensor.view(-1)
                width = (64 << 20) // tensor.element_size()
                for start in range(0, flat.numel(), width):
                    request = TensorChunk(
                        name, tuple(tensor.shape), tensor.dtype, start, tensor.numel()
                    )
                    fill(request, flat[start : start + width])

        yield initialize
