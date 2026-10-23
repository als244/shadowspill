"""Reuse immutable layer weights across shapes, without retaining old graphs."""

import gc

from .implementations.registry import BACKENDS, build


class ModelSession:
    def __init__(self):
        self.implementation = None
        self.key = None
        self.reused = False

    def acquire(self, backend, **configuration):
        share_tokens = BACKENDS[backend].reuse_model_across_tokens
        key = (
            backend,
            tuple(
                sorted(
                    (name, repr(value))
                    for name, value in configuration.items()
                    if name != "tokens" or not share_tokens
                )
            ),
        )
        self.reused = self.implementation is not None and key == self.key
        if self.reused:
            self.implementation.prepare_case(tokens=configuration["tokens"])
            return self.implementation
        self.close()
        self.implementation = build(backend, **configuration)
        self.key = key
        return self.implementation

    def close(self):
        if self.implementation is not None:
            previous, self.implementation = self.implementation, None
            previous.zero_grad()
            previous.close()
            del previous
            gc.collect()
            import torch

            torch.cuda.empty_cache()
        self.key = None
