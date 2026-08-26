from __future__ import annotations

import copy

import pytest
import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.optimizer import capture as optimizer_module
from shadowspill.pytorch.optimizer import (
    capture_optimizer,
    restore_optimizer_checkpoint_structure,
)
from shadowspill.pytorch.optimizer.artifacts import optimizer_value_identity


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
    assert (
        next(
            binding for binding in captured.bindings if binding.name == "weight"
        ).tensor.device.type
        == "cuda"
    )
    assert all(
        binding.tensor.device.type == "cuda"
        for binding in captured.bindings
        if binding.tensor.ndim != 0
    )
    assert captured.recurrent.operator_targets


def test_lazy_state_is_initialized_at_step_zero_without_parameter_mutation() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=False)
    captured = capture_optimizer({"weight": parameter}, optimizer)
    assert not captured.first_step_is_opaque
    assert captured.created_state_names == ()
    assert captured.preinitialized_state_names == (
        "optimizer.weight.exp_avg",
        "optimizer.weight.exp_avg_sq",
        "optimizer.weight.step",
    )
    assert captured.recurrent is not None
    state = optimizer.state[parameter]
    assert state["step"].item() == 0
    torch.testing.assert_close(state["exp_avg"], torch.zeros_like(parameter))
    torch.testing.assert_close(state["exp_avg_sq"], torch.zeros_like(parameter))
    torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_step_hooks_do_not_prevent_side_effect_free_state_initialization() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=False)
    hook_calls: list[int] = []
    optimizer.register_step_post_hook(
        lambda _optimizer, _args, _kwargs: hook_calls.append(1)
    )

    captured = capture_optimizer({"weight": parameter}, optimizer)

    assert hook_calls == []
    assert captured.recurrent_is_opaque
    assert captured.created_state_names == ()
    assert captured.preinitialized_state_names == (
        "optimizer.weight.exp_avg",
        "optimizer.weight.exp_avg_sq",
        "optimizer.weight.step",
    )
    assert optimizer.state[parameter]["step"].item() == 0
    torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_cuda_only_registered_optimizer_uses_fake_contract() -> None:
    mlops = pytest.importorskip("mlops")
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.ones_like(parameter)
    optimizer = mlops.optim.AdamW([parameter], lr=1e-3)

    captured = capture_optimizer({"weight": parameter}, optimizer)

    assert not captured.first_step_is_opaque
    assert captured.recurrent is not None
    assert "mlops.master_adamw_.default" in captured.recurrent.operator_targets
    assert captured.created_state_names == ()
    assert captured.preinitialized_state_names == (
        "optimizer.weight.exp_avg",
        "optimizer.weight.exp_avg_sq",
        "optimizer.weight.master_parameter",
        "optimizer.weight.step",
    )
    assert optimizer.state[parameter]["step"].item() == 0
    assert all(binding.tensor.device.type == "cuda" for binding in captured.bindings)


def test_cuda_only_discovery_inventories_every_parameter() -> None:
    mlops = pytest.importorskip("mlops")
    first = torch.nn.Parameter(torch.ones(8))
    second = torch.nn.Parameter(torch.full((4,), 2.0))
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    optimizer = mlops.optim.AdamW([first, second], lr=1e-3)
    captured = capture_optimizer(
        {"first": first, "second": second},
        optimizer,
    )

    expected_suffixes = {
        "exp_avg",
        "exp_avg_sq",
        "master_parameter",
        "step",
    }
    state_names = {
        binding.name for binding in captured.bindings if binding.role == "state"
    }
    assert state_names == {
        f"optimizer.{name}.{suffix}"
        for name in ("first", "second")
        for suffix in expected_suffixes
    }
    assert captured.created_state_names == ()
    assert len(captured.preinitialized_state_names) == 8
    assert captured.recurrent is not None
    assert (
        sum(
            node.op == "call_function"
            and str(node.target) == "mlops.master_adamw_.default"
            for node in captured.recurrent.graph_module.graph.nodes
        )
        == 2
    )
    assert len(captured.recurrent_tasks) == 2
    assert {
        next(
            name.removeprefix("optimizer.").split(".", 1)[0]
            for name in task.binding_names
            if name.startswith("optimizer.")
        )
        for task in captured.recurrent_tasks
    } == {"first", "second"}
    assert optimizer.state[first]["step"].item() == 0
    assert optimizer.state[second]["step"].item() == 0
    torch.testing.assert_close(first, torch.ones_like(first))
    torch.testing.assert_close(second, torch.full_like(second, 2.0))
    assert all(
        isinstance(binding.tensor, torch._subclasses.fake_tensor.FakeTensor)
        for binding in captured.bindings
    )


def test_output_created_lazy_state_retains_distinct_initial_plan() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.SGD([parameter], lr=1e-3, momentum=0.9)

    captured = capture_optimizer({"weight": parameter}, optimizer)

    assert captured.first_step_is_opaque
    assert captured.created_state_names == ("optimizer.weight.momentum_buffer",)
    assert captured.preinitialized_state_names == ()
    assert captured.initial is not None
    assert captured.initial.profile_output_names == (
        "optimizer.weight.momentum_buffer",
    )
    assert captured.initial.compatibility_digest != (
        captured.recurrent.compatibility_digest
    )
    assert optimizer.state == {}
    torch.testing.assert_close(parameter, torch.ones_like(parameter))


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


def test_optimizer_checkpoint_restore_preserves_tensor_objects() -> None:
    parameter, optimizer = _initialized(torch.optim.AdamW)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    state = optimizer.state[parameter]
    tensor_ids = {
        name: id(value)
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    for value in state.values():
        if isinstance(value, torch.Tensor):
            value.zero_()
    optimizer.param_groups[0]["lr"] = 0.25

    restored = restore_optimizer_checkpoint_structure(
        {"weight": parameter}, optimizer, checkpoint
    )
    for item in restored.tensors:
        item.destination.copy_(item.source)

    assert restored.initialized
    assert optimizer.param_groups[0]["lr"] == checkpoint["param_groups"][0]["lr"]
    for name, value in optimizer.state[parameter].items():
        if isinstance(value, torch.Tensor):
            assert id(value) == tensor_ids[name]
            torch.testing.assert_close(
                value,
                checkpoint["state"][0][name],
            )


def test_optimizer_checkpoint_restore_rejects_incompatible_tensor() -> None:
    parameter, optimizer = _initialized(torch.optim.AdamW)
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["state"][0]["exp_avg"] = torch.zeros(17)

    with pytest.raises(RuntimeError, match="incompatible geometry"):
        restore_optimizer_checkpoint_structure(
            {"weight": parameter}, optimizer, checkpoint
        )


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
    assert opaque_graph.bindings[1].tensor.device.type == "meta"
    assert tuple(opaque_graph.bindings[1].tensor.shape) == (4,)
    assert parameter.grad is None
    assert "recurrent optimizer graph is opaque" in (opaque_graph.opaque_reason or "")

    def fail_fake(*arguments: object) -> object:
        del arguments
        raise RuntimeError("fake inventory disabled")

    monkeypatch.setattr(optimizer_module, "_fake_cuda_optimizer", fail_fake)
    failed_fake = capture_optimizer(
        {"parameter": parameter}, _FailingAfterStateOptimizer([parameter])
    )
    assert failed_fake.recurrent is None
    assert "fake/meta behavior" in (failed_fake.opaque_reason or "")


def test_optimizer_option_identity_covers_bounded_containers() -> None:
    marker = object()
    identity = optimizer_value_identity(
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
