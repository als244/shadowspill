"""Portable authentic tensor samples shared by preparation and CPU bindings."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily

from shadowspill.errors import CaptureError


def encode_sample(value: torch.Tensor) -> dict[str, Any]:
    # Capture can surround an authentic CPU sample with FakeTensor dispatch.
    # Encoding must read that sample, rather than dispatch its copies as fake.
    with unset_fake_temporarily():
        if value.numel() * value.element_size() <= 1 << 20:
            return {"value": value.tolist()}
        raw = value.contiguous().reshape(-1).view(torch.uint8)
        return {"value_bytes": base64.b64encode(raw.numpy().tobytes()).decode()}


def copy_sample(value: torch.Tensor, source: Mapping[str, Any]) -> None:
    if "value" in source:
        sample = torch.as_tensor(
            source["value"], dtype=value.dtype, device=value.device
        )
    else:
        raw = bytearray(base64.b64decode(source["value_bytes"], validate=True))
        if len(raw) != value.numel() * value.element_size():
            raise CaptureError(
                "task control sample byte count differs from its geometry"
            )
        sample = torch.frombuffer(raw, dtype=value.dtype)
    value.copy_(sample.reshape(value.shape))
