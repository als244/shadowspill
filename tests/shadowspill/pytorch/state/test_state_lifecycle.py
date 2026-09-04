"""State primitives that need no accelerator: file import and meta construction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(256, 256, bias=False)
        self.register_buffer("count", torch.zeros(4))


def test_meta_construction_takes_its_values_from_a_mapped_checkpoint() -> None:
    """A model reaches CPU-resident state without an anonymous copy of itself.

    This is the shape `import_model_state` is meant to receive when host
    memory is scarce: no storage at construction, file-backed pages after the
    assignment, so only the pool copy is ever anonymous. What that saves is
    measured in the E003 experiment; this fixes the behaviour it relies on.
    """

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "net.pt"
        torch.manual_seed(3)
        saved = _Net()
        torch.save(dict(saved.state_dict()), checkpoint)

        with torch.device("meta"):
            model = _Net()
        assert all(value.device.type == "meta" for value in model.parameters())

        values = torch.load(
            checkpoint, map_location="cpu", mmap=True, weights_only=True
        )
        model.load_state_dict(values, assign=True)

        assert all(value.device.type == "cpu" for value in model.parameters())
        assert torch.equal(model.a.weight, saved.a.weight)
        assert torch.equal(model.count, saved.count)
