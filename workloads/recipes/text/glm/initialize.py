"""Checkpoint-precision GLM values for the common bounded tensor loader."""

import math

import torch

from workloads.models.shared.glm.checkpoint import physical_state
from workloads.models.shared.glm.initialization import _roles
from workloads.models.shared.initialization import fill_generated, validate_chunk


class Initialize:
    block_elements = 256 << 10

    def __init__(self, model, *, source=None, initialization="random", seed=0):
        self.seed, self.std = seed, model.config.initializer_range
        self.source = source if initialization == "checkpoint" else None
        tensors = physical_state(model)
        self.geometry = {
            name: (tuple(tensor.shape), tensor.dtype)
            for name, tensor in tensors.items()
        }
        if self.source is None:
            self.roles = _roles(model, tensors)
        else:
            if source.conversions:
                raise ValueError(
                    "bounded initialization requires checkpoint expert storage; "
                    "expert format conversion is not supported by this recipe yet"
                )
            self.roles = {
                name: "lora_a" if is_a else "zero"
                for name, is_a in source.lora_initializers.items()
            }

    def __call__(self, request, destination):
        validate_chunk(request, destination)
        if self.geometry[request.name] != (tuple(request.shape), destination.dtype):
            raise ValueError(
                f"initialization shape or dtype differs for {request.name}"
            )
        if request.name not in self.roles:
            assert self.source is not None
            self.source.copy_chunk(
                request.name, request.shape, request.offset, destination
            )
        else:
            fill_generated(
                request,
                destination,
                sample=self.values,
                seed=self.seed,
                block_elements=self.block_elements,
            )

    def values(self, name, shape, start, stop, generator):
        role = self.roles[name]
        sample = torch.empty(stop - start, dtype=torch.float32)
        if role in ("zero", "one", "fp8_scale"):
            return sample.fill_(
                {"zero": 0.0, "one": 1.0, "fp8_scale": self.std / 64}[role]
            )
        if role == "lora_a":
            bound = shape[-1] ** -0.5
            return sample.uniform_(-bound, bound, generator=generator)
        if role == "dt_bias":
            sample.uniform_(math.log(1e-3), math.log(1e-1), generator=generator)
            dt = sample.exp_()
            return dt + torch.log(-torch.expm1(-dt))
        sample.normal_(std=64 if role == "fp8" else self.std, generator=generator)
        return sample.clamp_(-448, 448) if role == "fp8" else sample
