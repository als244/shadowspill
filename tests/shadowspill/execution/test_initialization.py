"""Initialization must cover every required lease and stay within the chunk bound."""

from types import SimpleNamespace

import pytest
import torch

from shadowspill.execution.controls import Controls
from shadowspill.execution.tensors import verify_initialization, visit_tensors


def storage(bindings, geometry, values):
    return SimpleNamespace(
        plan=SimpleNamespace(program=SimpleNamespace(bindings=bindings)),
        geometry=geometry,
        aliases={name: name for name in geometry},
        initial=set(values),
        sizes={
            name: value.numel() * value.element_size() for name, value in values.items()
        },
        host=lambda name: values[name].view(torch.uint8),
    )


def test_parameter_initialized_masters_are_chunked_and_converted():
    data = {"w": torch.empty(17, dtype=torch.bfloat16), "m": torch.empty(17)}
    geometry = {
        name: {"dtype": str(value.dtype), "shape": (17,), "stride": (1,)}
        for name, value in data.items()
    }
    state = storage(
        {
            "kind": "step",
            "root_inputs": [],
            "root_sources": [],
            "registrations": [{"name": "weight", "object_id": "w"}],
            "optimizer_bindings": [
                {
                    "name": "master",
                    "role": "state",
                    "object_id": "m",
                    "initialization": {"kind": "parameter", "name": "weight"},
                }
            ],
        },
        geometry,
        data,
    )
    verify_initialization(state)
    calls = []

    def fill(request, chunk):
        assert chunk.numel() * chunk.element_size() <= 12
        calls.append(request.name)
        chunk.copy_(torch.arange(request.offset, request.offset + chunk.numel()))

    visit_tensors(state, fill, 12, initialize=True)
    assert set(calls) == {"weight"}
    torch.testing.assert_close(data["m"], data["w"].float())
    state.initial.add("unknown")
    with pytest.raises(ValueError, match="without a loader"):
        verify_initialization(state)


def test_cpu_controls_preserve_counters_and_reset_omitted_hyperparameters():
    bindings = [
        {
            "name": name,
            "role": role,
            "object_id": None,
            "device": "cpu",
            "shape": (),
            "stride": (),
            "dtype": dtype,
            "initialization": {"kind": "constant", "value": value},
        }
        for name, role, dtype, value in (
            ("step", "state", "torch.int64", 0),
            ("lr", "hyperparameter", "torch.float32", 0.01),
        )
    ]
    state = SimpleNamespace(
        plan=SimpleNamespace(
            program=SimpleNamespace(bindings={"optimizer_bindings": bindings})
        )
    )
    controls = Controls(state)
    controls.initialize(lambda *_: pytest.fail("defaults need no loader"))
    controls.host["step"].add_(1)
    controls.apply({"lr": 0.1})
    assert controls.host["lr"].item() == pytest.approx(0.1)
    controls.apply({})
    assert controls.host["lr"].item() == pytest.approx(0.01)
    assert controls.host["step"].item() == 1
    with pytest.raises(ValueError, match="unknown runtime hyperparameters"):
        controls.apply({"step": 5})
    with pytest.raises(ValueError, match="wrong shape"):
        controls.apply({"lr": [0.1, 0.2]})
    assert controls.host["lr"].item() == pytest.approx(0.01)
