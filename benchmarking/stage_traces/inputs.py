"""Materialize one stage, preserving geometry, aliases and authentic controls."""

from __future__ import annotations

import hashlib

import torch
from torch._subclasses.fake_tensor import FakeTensor

from shadowspill.pytorch.compilation.compiler import materialize_example_arguments

from .api import resolve


def policy(artifact, workload, position):
    provenance = artifact.input_provenance[position]
    value = artifact.example_arguments[position]
    name = (provenance.source or "").removeprefix("model.")
    if artifact.input_components:
        components = artifact.input_components[position][1]
        if components:
            name = ".".join((name, *components))
    if name in workload.state_initializers:
        return {
            "kind": "initialize",
            "source": name,
            **workload.state_initializers[name],
        }
    reference = provenance.representative_value
    if isinstance(reference, FakeTensor):
        reference = reference.constant
    if isinstance(reference, torch.Tensor) and not reference.is_meta:
        if reference.numel() > 1_000_000:
            raise ValueError(f"large unclassified literal input: {provenance}")
        return {"kind": "literal", "source": name, "value": reference.detach().cpu()}
    if provenance.role.value in {"parameter", "buffer", "constant"}:
        raise ValueError(f"no initializer for state {name}")
    if not value.is_floating_point():
        raise ValueError(f"no authentic integer/boolean input: {provenance}")
    seed = int.from_bytes(
        hashlib.sha256(f"{artifact.compatibility_digest}:{position}".encode()).digest()[
            :8
        ],
        "little",
    )
    return {"kind": "normal", "source": name, "seed": seed % 2**63, "std": 1.0}


@torch.no_grad()
def materialize(artifact, policies, device=0):
    args = materialize_example_arguments(
        artifact.example_arguments, device_ordinal=device
    )
    filled = {}
    for index, (target, spec) in enumerate(zip(args, policies, strict=True)):
        identity = (
            target.untyped_storage()._cdata,
            target.storage_offset(),
            tuple(target.shape),
            tuple(target.stride()),
            target.dtype,
        )
        if identity in filled:
            # The same named parameter may be passed several times. Do not
            # overwrite it with different random values on another occurrence.
            if spec["source"] != filled[identity]:
                raise ValueError(f"conflicting aliases at input {index}")
            continue
        filled[identity] = spec["source"]
        destination = _distinct_locations(target)
        if spec["kind"] == "literal":
            reference = spec["value"]
            for dimension, stride in enumerate(target.stride()):
                if stride == 0:
                    reference = reference.narrow(dimension, 0, 1)
            destination.copy_(reference)
        elif spec["kind"] == "initialize":
            resolve(spec["callable"])(
                destination, **spec.get("kwargs", {}), device=target.device
            )
        else:
            generator = torch.Generator(device=target.device).manual_seed(spec["seed"])
            destination.normal_(std=spec["std"], generator=generator)
    return args


def _distinct_locations(value):
    """Initialize broadcast storage once, preserving its original view."""
    shape = tuple(
        1 if stride == 0 else size
        for size, stride in zip(value.shape, value.stride(), strict=True)
    )
    return value.as_strided(shape, value.stride(), value.storage_offset())
