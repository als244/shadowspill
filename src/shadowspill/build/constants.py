"""Preserve small tensor literals exported from fake-device model execution."""

import torch
from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.aot import ExportCapture


def preserve_literals(exported: ExportCapture) -> None:
    # Dynamo can materialize an explicitly CUDA tensor literal while exporting.
    # Its values are static data, not weights or runtime device state. Keep a
    # bounded host copy before input provenance and the portable manifest are made.
    for name, value in exported.exported_program.constants.items():
        if not isinstance(value, torch.Tensor) or isinstance(value, FakeTensor):
            continue
        if value.device.type == "cpu":
            continue
        if value.numel() > 4096:
            raise CaptureError(
                f"device literal {name!r} exceeds the portable constant limit; "
                "register it as a model buffer"
            )
        with unset_fake_temporarily():
            exported.exported_program.constants[name] = value.cpu()
