"""Shared application operations preserve caller controls and checkpoint records."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from examples.training.application.loop import update
from examples.training.application.records import save_checkpoint
from shadowspill_client.protocol.requests import ArtifactRef


def future(value):
    return SimpleNamespace(result=lambda: value)


@pytest.mark.parametrize("pieces", [1, 2, 3])
@pytest.mark.parametrize("grad_scale", [None, 0, 0.25, -0.5])
@pytest.mark.parametrize("optim", [None, {"lr": 0.25}])
def test_shared_update_matches_a_weighted_linear_oracle(pieces, grad_scale, optim):
    inputs = np.array([1, 2, -3, 0.5, 4.0])
    targets = np.array([2, -1, 0.5, 4, -2.0])
    weights = np.array([0, 0.5, 1, 2, 0.25])
    batch = list(zip(inputs, targets, weights, strict=True))
    theta, gradient, updates = 0.5, 0.0, 0
    received = []

    def backward(rows):
        nonlocal gradient
        loss = 0.0
        for x, y, weight in rows:
            residual = theta * x - y
            gradient += weight * residual * x
            loss += 0.5 * weight * residual**2
        return future(SimpleNamespace(loss=loss))

    def optimize(**controls):
        nonlocal theta, updates
        received.append(controls)
        theta -= (
            controls.get("optim", {}).get("lr", 0.125)
            * gradient
            * controls.get("grad_scale", 1)
        )
        updates += 1
        return future(SimpleNamespace(weight=theta))

    def step(rows, **controls):
        loss = backward(rows).result().loss
        result = optimize(**controls).result()
        result.loss = loss
        return future(result)

    training = SimpleNamespace(
        step=step, forward_backward=backward, optim_step=optimize
    )
    result, loss = update(
        training, batch, pieces=pieces, optim=optim, grad_scale=grad_scale
    )
    scale = 1 if grad_scale is None else grad_scale
    residuals = 0.5 * inputs - targets
    expected_gradient = np.dot(weights * inputs, residuals)
    rate = 0.125 if optim is None else optim["lr"]
    assert result.weight == pytest.approx(0.5 - rate * expected_gradient * scale)
    assert loss == pytest.approx(0.5 * np.dot(weights, residuals**2) * scale)
    assert updates == 1
    expected_controls = {}
    if optim is not None:
        expected_controls["optim"] = optim
    if grad_scale is not None:
        expected_controls["grad_scale"] = grad_scale
    assert received == [expected_controls]


@pytest.mark.parametrize("fail", [False, True])
def test_checkpoint_reference_is_published_only_after_server_commit(tmp_path, fail):
    @dataclass(frozen=True)
    class Settings:
        steps: int = 3
        sizes: tuple[int, ...] = (1, 2)

    reference = ArtifactRef("saved", "a" * 64, 128)
    handle = SimpleNamespace(reference=reference)
    settings = Settings()
    data = {"source": "fixture", "seed": 7}
    destination = tmp_path / "step-00000002.json"

    def save(*, name, metadata):
        assert name == "step-00000002"
        assert metadata == {
            "application": "external/application@1",
            "settings": {"steps": 3, "sizes": [1, 2]},
            "data": data,
            "next_update": 2,
        }

        def result():
            assert not destination.exists()
            if fail:
                raise OSError("server commit failed")
            return handle

        return SimpleNamespace(result=result)

    training = SimpleNamespace(save_checkpoint=save, plans=handle)

    def invoke():
        return save_checkpoint(
            training,
            tmp_path,
            step=2,
            application="external/application@1",
            settings=settings,
            data=data,
        )

    if fail:
        with pytest.raises(OSError, match="server commit failed"):
            invoke()
        assert not destination.exists()
    else:
        assert invoke() is handle
        assert json.loads(destination.read_text()) == {
            "application": "external/application@1",
            "settings": {"steps": 3, "sizes": [1, 2]},
            "checkpoint": reference.to_dict(),
            "plans": reference.to_dict(),
        }
