from __future__ import annotations

import copy

import pytest
import torch

from shadowspill.pytorch import optimizer as optimizer_module
from shadowspill.pytorch.contracts import CaptureError
from shadowspill.pytorch.optimizer import capture_optimizer


def _initialized(
    optimizer_type: type[torch.optim.Optimizer], **options: object
) -> tuple[torch.nn.Parameter, torch.optim.Optimizer]:
    parameter = torch.nn.Parameter(torch.linspace(-1, 1, 16).reshape(4, 4))
    parameter.grad = torch.linspace(1, -1, 16).reshape(4, 4)
    optimizer = optimizer_type([parameter], lr=1e-2, foreach=False, **options)
    optimizer.step()
    return parameter, optimizer


@pytest.mark.parametrize(
    ("optimizer_type", "options"),
    [
        (torch.optim.AdamW, {}),
        (torch.optim.Adam, {}),
        (torch.optim.SGD, {"momentum": 0.9}),
    ],
)
def test_recurrent_graph_matches_standard_optimizer(
    optimizer_type: type[torch.optim.Optimizer], options: dict[str, object]
) -> None:
    parameter, optimizer = _initialized(optimizer_type, **options)
    reference_parameter = torch.nn.Parameter(parameter.detach().clone())
    reference_parameter.grad = parameter.grad.detach().clone()
    reference = optimizer_type([reference_parameter], lr=1e-2, foreach=False, **options)
    reference.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    before_parameter = parameter.detach().clone()
    before_state = copy.deepcopy(optimizer.state_dict())
    captured = capture_optimizer({"weight": parameter}, optimizer)
    torch.testing.assert_close(parameter, before_parameter)
    after_state = optimizer.state_dict()
    assert after_state["param_groups"] == before_state["param_groups"]
    assert after_state["state"].keys() == before_state["state"].keys()
    for parameter_id, state in after_state["state"].items():
        for name, value in state.items():
            expected = before_state["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected)
            else:
                assert value == expected
    assert captured.recurrent is not None
    with torch.no_grad():
        captured.recurrent.graph_module(
            *(binding.tensor for binding in captured.bindings)
        )
    reference.step()

    captured_parameter = next(
        binding.tensor for binding in captured.bindings if binding.name == "weight"
    )
    torch.testing.assert_close(captured_parameter, reference_parameter)
    captured_state = {
        binding.name: binding.tensor
        for binding in captured.bindings
        if binding.name.startswith("optimizer.weight")
    }
    for name, value in reference.state[reference_parameter].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(
                captured_state[f"optimizer.weight.{name}"], value
            )


def test_lazy_state_gets_distinct_opaque_first_step_without_mutation() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=False)
    captured = capture_optimizer({"weight": parameter}, optimizer)
    assert captured.first_step_is_opaque
    assert captured.created_state_names == (
        "optimizer.weight.exp_avg",
        "optimizer.weight.exp_avg_sq",
        "optimizer.weight.step",
    )
    assert captured.recurrent is not None
    assert optimizer.state == {}
    torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_cuda_only_registered_optimizer_uses_fake_contract() -> None:
    mlops = pytest.importorskip("mlops")
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.ones_like(parameter)
    optimizer = mlops.optim.AdamW([parameter], lr=1e-3)

    captured = capture_optimizer({"weight": parameter}, optimizer)

    assert captured.first_step_is_opaque
    assert captured.recurrent is not None
    assert "mlops.master_adamw_.default" in captured.recurrent.operator_targets
    assert captured.created_state_names == (
        "optimizer.weight.exp_avg",
        "optimizer.weight.exp_avg_sq",
        "optimizer.weight.master_parameter",
        "optimizer.weight.step",
    )
    assert optimizer.state == {}
    assert all(binding.tensor.device.type == "cuda" for binding in captured.bindings)


def test_optimizer_state_container_conversion_preserves_structure() -> None:
    source = torch.ones(2)
    converted = optimizer_module._map_optimizer_tensors(
        {"list": [source, 3], "tuple": (source, "value")},
        lambda tensor: tensor + 1,
    )

    assert isinstance(converted, dict)
    assert isinstance(converted["list"], list)
    assert isinstance(converted["tuple"], tuple)
    torch.testing.assert_close(converted["list"][0], torch.full((2,), 2.0))
    assert converted["list"][1] == 3
    assert converted["tuple"][1] == "value"


class _CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters: object, scale: float = 0.25) -> None:
        super().__init__(parameters, {"scale": scale})

    @torch.no_grad()
    def step(self, closure: object = None) -> None:
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-group["scale"])


class _SubclassStateOptimizer(_CustomOptimizer):
    def __init__(self, parameters: object) -> None:
        super().__init__(parameters)
        self.required_subclass_state = 2.0

    @torch.no_grad()
    def step(self, closure: object = None) -> None:
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.add_(
                        parameter.grad,
                        alpha=-group["scale"] * self.required_subclass_state,
                    )


class _DataDependentOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters: object) -> None:
        super().__init__(parameters, {"lr": 0.1})

    @torch.no_grad()
    def step(self, closure: object = None) -> None:
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    scale = float(parameter.grad.flatten()[0].item())
                    parameter.add_(parameter.grad, alpha=-group["lr"] * scale)


def test_unrelated_custom_optimizer_is_captured_without_allowlist() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    parameter.grad = torch.full_like(parameter, 2)
    optimizer = _CustomOptimizer([parameter])
    captured = capture_optimizer({"parameter": parameter}, optimizer)
    assert not captured.first_step_is_opaque
    assert captured.recurrent is not None
    assert "aten.add_.Tensor" in captured.recurrent.operator_targets


def test_valid_data_dependent_optimizer_becomes_bounded_opaque_task() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    parameter.grad = torch.full_like(parameter, 2)
    first = capture_optimizer(
        {"parameter": parameter}, _DataDependentOptimizer([parameter])
    )
    second = capture_optimizer(
        {"parameter": parameter}, _DataDependentOptimizer([parameter])
    )

    assert first.recurrent_is_opaque
    assert first.recurrent is not None
    assert first.recurrent.compatibility_digest == second.recurrent.compatibility_digest
    assert "data-dependent" in (first.opaque_reason or "")
    assert torch.is_grad_enabled()
    torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_optimizer_copy_preserves_subclass_state_omitted_by_base_protocol() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    parameter.grad = torch.ones_like(parameter)
    captured = capture_optimizer(
        {"parameter": parameter}, _SubclassStateOptimizer([parameter])
    )
    assert captured.recurrent is not None


def test_parameter_coverage_mismatch_is_rejected() -> None:
    first = torch.nn.Parameter(torch.ones(1))
    second = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([first], lr=0.1)
    with pytest.raises(Exception, match="coverage"):
        capture_optimizer({"first": first, "second": second}, optimizer)


def test_optimizer_contract_errors_are_field_specific() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with pytest.raises(TypeError, match="derive"):
        capture_optimizer({"parameter": parameter}, object())  # type: ignore[arg-type]
    with pytest.raises(CaptureError, match="non-empty"):
        capture_optimizer({"": parameter}, optimizer)
    with pytest.raises(CaptureError, match="not a Parameter"):
        capture_optimizer({"value": torch.ones(1)}, optimizer)  # type: ignore[dict-item]

    optimizer.param_groups[0]["params"].append(parameter)
    with pytest.raises(CaptureError, match="more than once"):
        capture_optimizer({"parameter": parameter}, optimizer)


class _UncopyableOptimizer(_CustomOptimizer):
    def __deepcopy__(self, memo: object) -> _UncopyableOptimizer:
        del memo
        raise RuntimeError("copy disabled")


class _FailingOptimizer(_CustomOptimizer):
    @torch.no_grad()
    def step(self, closure: object = None) -> None:
        del closure
        raise RuntimeError("step disabled")


class _FailingAfterStateOptimizer(_CustomOptimizer):
    @torch.no_grad()
    def step(self, closure: object = None) -> None:
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                self.state[parameter]["buffer"] = torch.zeros_like(parameter)
        raise RuntimeError("kernel unavailable")


def test_opaque_fallbacks_preserve_the_original_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    parameter.grad = None
    uncopyable = capture_optimizer(
        {"parameter": parameter}, _UncopyableOptimizer([parameter])
    )
    assert uncopyable.recurrent_is_opaque
    assert "cannot be copied" in (uncopyable.opaque_reason or "")

    failing = capture_optimizer(
        {"parameter": parameter}, _FailingOptimizer([parameter])
    )
    assert failing.recurrent_is_opaque
    assert "discovery step failed" in (failing.opaque_reason or "")

    optimizer = torch.optim.SGD([parameter], lr=0.1)

    def fail_export(_optimizer: torch.optim.Optimizer) -> torch.fx.GraphModule:
        raise RuntimeError("graph disabled")

    monkeypatch.setattr(optimizer_module, "_export_optimizer_graph", fail_export)
    opaque_graph = capture_optimizer({"parameter": parameter}, optimizer)
    assert opaque_graph.recurrent_is_opaque
    assert opaque_graph.bindings[1].name == "gradient.parameter"
    torch.testing.assert_close(opaque_graph.bindings[1].tensor, torch.ones(4))
    assert "recurrent optimizer graph is opaque" in (opaque_graph.opaque_reason or "")

    def fail_fake(*arguments: object) -> object:
        del arguments
        raise RuntimeError("fake inventory disabled")

    monkeypatch.setattr(optimizer_module, "_fake_cuda_optimizer", fail_fake)
    failed_fake = capture_optimizer(
        {"parameter": parameter}, _FailingAfterStateOptimizer([parameter])
    )
    assert failed_fake.recurrent is None
    assert "fake CUDA inventory failed" in (failed_fake.opaque_reason or "")


def test_optimizer_option_identity_covers_bounded_containers() -> None:
    marker = object()
    identity = optimizer_module._optimizer_value_identity(
        {
            "tensor": torch.ones(2),
            "tuple": (1, None),
            "list": [True, "value"],
            "object": marker,
        }
    )

    assert identity["tensor"]["tensor"]["shape"] == (2,)
    assert identity["tuple"] == {"tuple": [1, None]}
    assert identity["list"] == {"list": [True, "value"]}
    assert identity["object"]["type"] == "object"
