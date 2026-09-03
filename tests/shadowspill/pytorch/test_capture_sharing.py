"""Same-structure accumulation positions share one export, never their inputs."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.capture.aot import capture_training_objective
from shadowspill.pytorch.capture.fake import fake_device_model
from shadowspill.pytorch.guards import capture_training_signatures
from shadowspill.pytorch.planning import training as planning_training


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.linear(tokens)


def _objective(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens).sum()


class _Archive:
    def __init__(self) -> None:
        self.positions: list[int] = []

    def archive_export(self, capture: Any, *, mode: str, position: int) -> str:
        self.positions.append(position)
        return "digest"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fake CUDA inputs")
def test_positions_with_one_structure_export_once(monkeypatch: Any) -> None:
    microbatches = tuple((torch.zeros(2, 4),) for _ in range(4))
    signatures = capture_training_signatures(microbatches)
    exports = 0

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal exports
        exports += 1
        return capture_training_objective(*args, **kwargs)

    monkeypatch.setattr(planning_training, "capture_training_objective", counting)
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_device_model(_Tiny(), fake_mode, device_index=0)
    archive = _Archive()
    timer = planning_training.PlanningTimer(verbose=False)
    captures = planning_training._capture_training_objectives(
        model,
        _objective,
        microbatches,
        signatures=signatures,
        fake_mode=fake_mode,
        device_ordinal=0,
        stores=archive,  # type: ignore[arg-type]
        timer=timer,
    )
    assert exports == 1
    assert archive.positions == [0]
    assert len(captures) == 4
    programs = {id(item.exported.exported_program) for item in captures}
    assert len(programs) == 1
    inputs = {id(item.exported.flat_inputs) for item in captures}
    assert len(inputs) == 4
