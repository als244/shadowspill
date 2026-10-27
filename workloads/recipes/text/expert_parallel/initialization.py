"""Bounded initialization: shared dense tensors, distinct local expert subsets."""

from dataclasses import replace

import torch

from workloads.models.shared.initialization import fill_generated


class Initialize:
    def __init__(self, catalog_initializer, *, expert_rank, std):
        self.catalog = catalog_initializer
        self.expert_rank, self.std = expert_rank, std

    def __call__(self, request, destination):
        # Quack exposes physical BF16 payloads under logical FP32 parameters.
        request = replace(request, name=request.name.removesuffix("._data"))
        if ".moe.experts." not in request.name:
            self.catalog(request, destination)
            return
        expert = request.name.endswith(("gate_up_weight", "down_weight"))
        fill_generated(
            request,
            destination,
            sample=self.values,
            seed=self.catalog.seed + (self.expert_rank * 1_000_003 if expert else 0),
        )

    def values(self, name, shape, start, stop, generator):
        return torch.empty(stop - start, dtype=torch.float32).normal_(
            std=self.std, generator=generator
        )
