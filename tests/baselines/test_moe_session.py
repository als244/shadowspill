from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from baselines.moe_layer import session


@pytest.mark.parametrize(
    "backend,reused", [("sonicmoe", True), ("deepep", True), ("quackmoe", False)]
)
def test_model_reuse_tracks_shape_capacity_and_precision(monkeypatch, backend, reused):
    import torch

    models = []

    def build(*args, **kwargs):
        model = SimpleNamespace(zero_grad=Mock(), close=Mock(), prepare_case=Mock())
        models.append(model)
        return model

    monkeypatch.setattr(session, "build", build)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    state = session.ModelSession()
    first = state.acquire(backend, tokens=128, precision="bf16", seed=73)
    second = state.acquire(backend, tokens=256, precision="bf16", seed=73)
    assert (first is second) == reused
    assert state.reused == reused
    third = state.acquire(backend, tokens=256, precision="fp8_current", seed=73)
    assert third is not second
    second.close.assert_called_once()
    state.close()
    third.zero_grad.assert_called_once()
    third.close.assert_called_once()
    assert state.implementation is None
