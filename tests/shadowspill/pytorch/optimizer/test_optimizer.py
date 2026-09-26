from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.optimizer import (
    OptimizerTensorRole,
    capture_optimizer,
    restore_optimizer_checkpoint_structure,
)
from shadowspill.pytorch.optimizer import discovery as discovery_module
from shadowspill.pytorch.optimizer import sandbox as sandbox_module
from shadowspill.pytorch.optimizer import trace as trace_module
from shadowspill.pytorch.optimizer.artifacts import optimizer_value_identity
from shadowspill.pytorch.optimizer.capture import declare_optimizer_state


def _initialized(
    optimizer_type: type[torch.optim.Optimizer], **options: object
) -> tuple[torch.nn.Parameter, torch.optim.Optimizer]:
    parameter = torch.nn.Parameter(torch.linspace(-1, 1, 16).reshape(4, 4))
    parameter.grad = torch.linspace(1, -1, 16).reshape(4, 4)
    optimizer = optimizer_type([parameter], lr=1e-2, foreach=False, **options)
    optimizer.step()
    return parameter, optimizer


def _install_declared_state(
    named_parameters: dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> None:
    """The state planning installs before it captures, at zero."""

    for entry in declare_optimizer_state(named_parameters, optimizer):
        parameter = named_parameters[entry.parameter_name]
        optimizer.state[parameter][entry.entry_name] = torch.zeros(
            entry.shape, dtype=entry.dtype
        )


@pytest.mark.parametrize(
    ("optimizer_type", "options"),
    [
        (torch.optim.AdamW, {}),
        (torch.optim.Adam, {}),
        (torch.optim.SGD, {"momentum": 0.9}),
    ],
)
def test_update_graph_matches_standard_optimizer(
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
    assert captured.update is not None
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
    assert captured.update.operator_targets


def test_device_only_registered_optimizer_uses_fake_contract() -> None:
    mlops = pytest.importorskip("mlops")
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.ones_like(parameter)
    optimizer = mlops.optim.AdamW([parameter], lr=1e-3)
    _install_declared_state({"weight": parameter}, optimizer)

    captured = capture_optimizer({"weight": parameter}, optimizer)

    assert captured.update is not None
    assert "mlops.adamw_.default" in captured.update.operator_targets
    # Everything the update computes with is on the device. A hyperparameter
    # is not: it is a scalar the update reads and passes to its kernel, so it
    # stays on the host, where writing the next step's value copies nothing.
    for binding in captured.bindings:
        expected = (
            "cpu" if binding.role is OptimizerTensorRole.HYPERPARAMETER else "cuda"
        )
        assert binding.tensor.device.type == expected


def test_state_kept_at_another_precision_is_captured_at_it() -> None:
    """Moments kept apart from the parameter's dtype keep theirs.

    Discovery puts its copy of the optimizer back as it found it. Doing that
    with ``Optimizer.load_state_dict`` cast every floating-point entry but
    ``step`` to the parameter's dtype, so fp32 moments over bf16 weights were
    captured as bf16 ones, and profiling refused the planned fp32 state.
    """

    mlops = pytest.importorskip("mlops")
    parameter = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    parameter.grad = torch.ones_like(parameter)
    optimizer = mlops.optim.AdamW([parameter], lr=1e-3, state_dtype=torch.float32)
    _install_declared_state({"weight": parameter}, optimizer)

    captured = capture_optimizer({"weight": parameter}, optimizer)

    dtypes = {binding.name: binding.tensor.dtype for binding in captured.bindings}
    assert dtypes["weight"] == torch.bfloat16
    for entry in ("exp_avg", "exp_avg_sq"):
        assert dtypes[f"optimizer.weight.{entry}"] == torch.float32, entry
    # What the traced update declares for a moment, which profiling checks the
    # planned fp32 moment against.
    assert captured.update is not None
    names = [binding.name for binding in captured.bindings]
    position = names.index("optimizer.weight.exp_avg")
    assert captured.update.tensor_inputs[position].dtype == torch.float32


def test_device_only_discovery_inventories_every_parameter() -> None:
    mlops = pytest.importorskip("mlops")
    first = torch.nn.Parameter(torch.ones(8))
    second = torch.nn.Parameter(torch.full((4,), 2.0))
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    optimizer = mlops.optim.AdamW([first, second], lr=1e-3)
    _install_declared_state({"first": first, "second": second}, optimizer)
    installed = {
        (id(parameter), name): value
        for parameter, state in optimizer.state.items()
        for name, value in state.items()
    }
    captured = capture_optimizer(
        {"first": first, "second": second},
        optimizer,
    )

    expected_suffixes = {"exp_avg", "exp_avg_sq", "step"}
    state_names = {
        binding.name for binding in captured.bindings if binding.role == "state"
    }
    assert state_names == {
        f"optimizer.{name}.{suffix}"
        for name in ("first", "second")
        for suffix in expected_suffixes
    }
    assert captured.update is not None
    assert (
        sum(
            node.op == "call_function" and str(node.target) == "mlops.adamw_.default"
            for node in captured.update.graph_module.graph.nodes
        )
        == 2
    )
    assert len(captured.update_tasks) == 2
    assert {
        next(
            name.removeprefix("optimizer.").split(".", 1)[0]
            for name in task.binding_names
            if name.startswith("optimizer.")
        )
        for task in captured.update_tasks
    } == {"first", "second"}
    # Capture leaves the caller's optimizer untouched: the same state tensors,
    # still at zero, and the parameters unchanged.
    assert {
        (id(parameter), name): value
        for parameter, state in optimizer.state.items()
        for name, value in state.items()
    } == installed
    assert all(not value.any() for value in installed.values())
    torch.testing.assert_close(first, torch.ones_like(first))
    torch.testing.assert_close(second, torch.full_like(second, 2.0))
    # Everything discovery had to invent is fake, so inventing it allocated
    # nothing. A hyperparameter is the exception on purpose: it is the
    # caller's own scalar, and it has to be, because that is the tensor a
    # later step writes to change what the update does.
    fake = torch._subclasses.fake_tensor.FakeTensor
    for binding in captured.bindings:
        if binding.role is OptimizerTensorRole.HYPERPARAMETER:
            assert not isinstance(binding.tensor, fake)
            assert binding.tensor.device.type == "cpu"
        else:
            assert isinstance(binding.tensor, fake)


def test_state_the_first_step_would_create_is_refused() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.SGD([parameter], lr=1e-3, momentum=0.9)

    with pytest.raises(CaptureError, match=r"creates state \(optimizer\.weight"):
        capture_optimizer({"weight": parameter}, optimizer)

    assert optimizer.state == {}
    torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_optimizer_state_container_conversion_preserves_structure() -> None:
    source = torch.ones(2)
    converted = sandbox_module.map_optimizer_tensors(
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
    for item in restored:
        item.destination.copy_(item.source)

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


def test_optimizer_checkpoint_without_required_state_changes_nothing() -> None:
    parameter, optimizer = _initialized(torch.optim.AdamW)
    before = dict(optimizer.state[parameter])
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["state"] = {}
    checkpoint["param_groups"][0]["lr"] = 0.25

    with pytest.raises(RuntimeError, match="lacks planned state"):
        restore_optimizer_checkpoint_structure(
            {"weight": parameter},
            optimizer,
            checkpoint,
            required=("optimizer.weight.exp_avg",),
        )

    assert optimizer.state[parameter] == before
    assert optimizer.param_groups[0]["lr"] == 1e-2


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
    assert captured.update is not None
    assert "aten.add_.Tensor" in captured.update.operator_targets


def test_valid_data_dependent_optimizer_becomes_bounded_opaque_task() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    parameter.grad = torch.full_like(parameter, 2)
    first = capture_optimizer(
        {"parameter": parameter}, _DataDependentOptimizer([parameter])
    )
    second = capture_optimizer(
        {"parameter": parameter}, _DataDependentOptimizer([parameter])
    )

    assert first.update_is_opaque
    assert first.update is not None
    assert second.update is not None
    assert first.update.compatibility_digest == second.update.compatibility_digest
    assert "data-dependent" in (first.opaque_reason or "")
    assert torch.is_grad_enabled()
    torch.testing.assert_close(parameter, torch.ones_like(parameter))


def test_optimizer_copy_preserves_subclass_state_omitted_by_base_protocol() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    parameter.grad = torch.ones_like(parameter)
    captured = capture_optimizer(
        {"parameter": parameter}, _SubclassStateOptimizer([parameter])
    )
    assert captured.update is not None


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
    assert uncopyable.update_is_opaque
    assert "cannot be copied" in (uncopyable.opaque_reason or "")

    failing = capture_optimizer(
        {"parameter": parameter}, _FailingOptimizer([parameter])
    )
    assert failing.update_is_opaque
    assert "discovery step failed" in (failing.opaque_reason or "")

    optimizer = torch.optim.SGD([parameter], lr=0.1)

    def fail_export(_optimizer: torch.optim.Optimizer) -> torch.fx.GraphModule:
        raise RuntimeError("graph disabled")

    monkeypatch.setattr(trace_module, "_export_optimizer_graph", fail_export)
    opaque_graph = capture_optimizer({"parameter": parameter}, optimizer)
    assert opaque_graph.update_is_opaque
    assert opaque_graph.bindings[1].name == "gradient.parameter"
    assert opaque_graph.bindings[1].tensor.device.type == "meta"
    assert tuple(opaque_graph.bindings[1].tensor.shape) == (4,)
    assert parameter.grad is None
    assert "the optimizer update is opaque" in (opaque_graph.opaque_reason or "")

    def fail_fake(*arguments: object) -> object:
        del arguments
        raise RuntimeError("fake inventory disabled")

    monkeypatch.setattr(discovery_module, "fake_device_optimizer", fail_fake)
    failed_fake = capture_optimizer(
        {"parameter": parameter}, _FailingAfterStateOptimizer([parameter])
    )
    assert failed_fake.update is None
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


def test_an_update_capture_is_served_from_the_store(tmp_path: Path) -> None:
    from shadowspill.pytorch.optimizer.store import OptimizerCaptureStore

    records: list[dict[str, object]] = []

    def record(**kwargs: object) -> None:
        records.append(kwargs)

    parameter, optimizer = _initialized(torch.optim.AdamW)
    first = capture_optimizer(
        {"weight": parameter},
        optimizer,
        store=OptimizerCaptureStore(tmp_path, artifact_recorder=record),
    )
    assert [item["access"] for item in records] == ["write"]
    assert first.update is not None

    # a fresh optimizer over the same inventory reads the trace back
    parameter, optimizer = _initialized(torch.optim.AdamW)
    again = capture_optimizer(
        {"weight": parameter},
        optimizer,
        store=OptimizerCaptureStore(tmp_path, artifact_recorder=record),
    )
    assert [item["access"] for item in records] == ["write", "read"]
    assert again.update is not None
    assert again.update.compatibility_digest == first.update.compatibility_digest
    assert again.update.operator_targets == first.update.operator_targets
    assert [task.binding_names for task in again.update_tasks] == [
        task.binding_names for task in first.update_tasks
    ]
    assert [binding.name for binding in again.bindings] == [
        binding.name for binding in first.bindings
    ]
    assert again.mutation_names == first.mutation_names

    # a different optimizer over the same tensors is another entry
    parameter, optimizer = _initialized(torch.optim.SGD, momentum=0.9)
    capture_optimizer(
        {"weight": parameter},
        optimizer,
        store=OptimizerCaptureStore(tmp_path, artifact_recorder=record),
    )
    assert [item["access"] for item in records] == ["write", "read", "write"]


def _masters_of(
    model: torch.nn.Module, lr: float = 1e-2
) -> tuple[dict[str, torch.nn.Parameter], torch.optim.Optimizer]:
    """Master copies of a model's weights at fp32, and AdamW over them."""

    masters = {
        name: torch.nn.Parameter(parameter.detach().float())
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(masters.values(), lr=lr, foreach=False)
    for master in masters.values():
        optimizer.state[master] = {
            "step": torch.tensor(0.0),
            "exp_avg": torch.zeros_like(master),
            "exp_avg_sq": torch.zeros_like(master),
        }
    return masters, optimizer


def _run_update(
    captured: object,
    masters: dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    gradients: dict[str, torch.Tensor],
    copies: dict[str, torch.Tensor],
    steps: int,
) -> dict[str, torch.Tensor]:
    """Run a captured update on real tensors, in binding order, ``steps`` times."""

    values: dict[str, torch.Tensor] = {}
    for binding in captured.bindings:  # type: ignore[attr-defined]
        name = binding.name
        if binding.role is OptimizerTensorRole.PARAMETER:
            values[name] = masters[name].detach().clone()
        elif binding.role is OptimizerTensorRole.GRADIENT:
            values[name] = gradients[name.removeprefix("gradient.")].clone()
        elif binding.role is OptimizerTensorRole.STATE:
            owner, entry = name.removeprefix("optimizer.").rsplit(".", 1)
            values[name] = optimizer.state[masters[owner]][entry].detach().clone()
        else:
            values[name] = copies[name.removeprefix("compute.")].detach().clone()
    arguments = [values[binding.name] for binding in captured.bindings]  # type: ignore[attr-defined]
    with torch.no_grad():
        for _ in range(steps):
            captured.update.graph_module(*arguments)  # type: ignore[attr-defined]
    return values


def test_an_update_over_masters_steps_them_and_writes_their_copies() -> None:
    """A bf16 model's fp32 masters step exactly as fp32 weights would, given
    the gradients the model computes, and each copy is its master's cast."""

    torch.manual_seed(0)
    model = torch.nn.Linear(6, 3).to(torch.bfloat16)
    copies = dict(model.named_parameters())
    gradients = {name: torch.randn_like(value) for name, value in copies.items()}

    masters, optimizer = _masters_of(model)
    captured = capture_optimizer(masters, optimizer, compute_copies=copies)
    roles = {binding.name: binding for binding in captured.bindings}
    assert roles["weight"].tensor.dtype == torch.float32
    assert roles["gradient.weight"].tensor.dtype == torch.bfloat16
    assert roles["compute.weight"].role is OptimizerTensorRole.COMPUTE_COPY
    assert "compute.weight" in captured.mutation_names
    # One task per parameter, its copy with it.
    assert sorted(task.binding_names[-1] for task in captured.update_tasks) == [
        "compute.bias",
        "compute.weight",
    ]
    stepped = _run_update(captured, masters, optimizer, gradients, copies, steps=3)

    # The same optimizer over fp32 weights, handed the gradients cast up front.
    plain_masters, plain_optimizer = _masters_of(model)
    plain = capture_optimizer(plain_masters, plain_optimizer)
    reference = _run_update(
        plain,
        plain_masters,
        plain_optimizer,
        {name: value.float() for name, value in gradients.items()},
        {},
        steps=3,
    )
    for name in copies:
        assert torch.equal(stepped[name], reference[name])
        assert torch.equal(stepped[f"compute.{name}"], reference[name].bfloat16())
        assert not torch.equal(stepped[name], masters[name])  # it moved


def test_an_update_over_masters_is_served_from_the_store(tmp_path: Path) -> None:
    from shadowspill.pytorch.optimizer.store import OptimizerCaptureStore

    records: list[dict[str, object]] = []

    def record(**kwargs: object) -> None:
        records.append(kwargs)

    torch.manual_seed(0)
    model = torch.nn.Linear(6, 3).to(torch.bfloat16)
    copies = dict(model.named_parameters())
    first = capture_optimizer(
        *_masters_of(model),
        compute_copies=copies,
        store=OptimizerCaptureStore(tmp_path, artifact_recorder=record),
    )
    again = capture_optimizer(
        *_masters_of(model),
        compute_copies=copies,
        store=OptimizerCaptureStore(tmp_path, artifact_recorder=record),
    )
    # Without copies, the same optimizer is another entry.
    capture_optimizer(
        *_masters_of(model),
        store=OptimizerCaptureStore(tmp_path, artifact_recorder=record),
    )

    assert [item["access"] for item in records] == ["write", "read", "write"]
    assert [
        (binding.name, binding.role, binding.tensor.dtype) for binding in again.bindings
    ] == [
        (binding.name, binding.role, binding.tensor.dtype) for binding in first.bindings
    ]
    assert first.update is not None and again.update is not None
    assert again.update.compatibility_digest == first.update.compatibility_digest


def test_masters_need_an_update_that_can_be_traced() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    master = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))

    with pytest.raises(CaptureError, match="needs an update that can be traced"):
        capture_optimizer(
            {"parameter": master},
            _DataDependentOptimizer([master]),
            compute_copies={"parameter": parameter},
        )
