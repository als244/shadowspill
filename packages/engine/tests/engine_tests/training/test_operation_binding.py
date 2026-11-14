"""Prepared operation ABI, selective fields, modes, padding and real CPU capture."""

import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from engine_tests.fixtures.definition_operations import definitions, example, records
from shadowspill.parallelism import GroupRef, InputFieldPlacement, InputPlacement
from shadowspill.pytorch.capture.objective import capture_training_objective
from shadowspill.training.inputs.validation import combined_schema
from shadowspill.training.preparation.binding import control_tensors
from shadowspill.training.preparation.operation_build import build_operation
from shadowspill.training.preparation.operations import (
    OperationContract,
    OperationKey,
    resolve_operations,
)
from shadowspill_client import Operation
from shadowspill_client.protocol.controls import ControlSchema, ScalarControl


def test_operation_order_modes_defaults_and_optimizer_dependency():
    _, objective, _ = definitions()
    assert resolve_operations(objective, None, optimizer=True) == (
        OperationKey("step", "metrics"),
    )
    resolved = resolve_operations(
        objective,
        {
            "forward_backward": ["metrics"],
            "forward": ["predictions", Operation("metrics", mode="training")],
        },
        optimizer=True,
    )
    assert resolved == (
        OperationKey("forward_backward", "metrics"),
        OperationKey("forward", "predictions", "evaluation"),
        OperationKey("forward", "metrics", "training"),
        OperationKey("optim_step", None),
    )
    assert (
        len(
            resolve_operations(
                objective, {"forward_backward": ["metrics"]}, optimizer=False
            )
        )
        == 1
    )
    assert (
        len(resolve_operations(objective, {"forward": ["predictions"]}, optimizer=True))
        == 1
    )
    for bad in (
        {},
        {"unknown": ["metrics"]},
        {"forward": []},
        {"step": "metrics"},
        {"step": ["metrics", "metrics"]},
        {"forward": ["unknown"]},
        {"step": [Operation("metrics", mode="evaluation")]},
    ):
        with pytest.raises((ValueError, TypeError)):
            resolve_operations(objective, bad, optimizer=True)
    with pytest.raises(ValueError, match="requires an optimizer"):
        resolve_operations(objective, None, optimizer=False)


def test_external_capture_uses_runtime_coefficients_and_preserves_weighted_sum():
    resolved, objective, _ = definitions()
    model = resolved.create()
    with example(resolved, objective, "forward_backward") as (binding, adapted):
        args = binding.arguments(adapted)
        captured = capture_training_objective(model, binding.objective, args)
        for temperature in (0.5, 2.0):
            args = binding.arguments(adapted, {"temperature": temperature})
            loss, valid = captured.exported.exported_program.module()(*args)
            expected = objective.computation("metrics")(
                model,
                {"pixels": args[0]["pixels"][:3]},
                {k: v[:3] for k, v in args[1].items()},
                args[2],
            ).loss
            torch.testing.assert_close(loss, expected)
            assert valid.item() == 3  # Numeric zero weight still participates.
            actual_grad = torch.autograd.grad(
                loss, tuple(model.parameters()), retain_graph=True
            )
            expected_grad = torch.autograd.grad(expected, tuple(model.parameters()))
            torch.testing.assert_close(actual_grad, expected_grad)
        assert binding.arguments(adapted)[2]["temperature"].item() == 1.0
        with pytest.raises(ValueError, match="unknown controls"):
            binding.arguments(adapted, {"unused": 2.0})
        assert binding.signature()["objective"]["padding_safe"]


def test_forward_outputs_require_only_their_own_fields_and_controls():
    model, objective, _ = definitions()
    key = OperationKey("forward", "predictions", "evaluation")
    contract = OperationContract(key, objective)
    assert contract.required_paths == (("inputs", "pixels"),)
    assert not contract.loss_params()
    with example(model, objective, key, datums=records(targets=False)) as (
        binding,
        adapted,
    ):
        args = binding.arguments(adapted)
        loss, values = binding.forward(model.create().eval(), *args)
        assert loss is None and values["prediction"].shape == (4, 3)
        assert args[1] == {} and args[2] == {}
        with pytest.raises(ValueError, match="unknown controls"):
            binding.arguments(adapted, {"temperature": 1.0})


def test_loss_ownership_zeroes_only_nonleaders_without_averaging():
    resolved, objective, _ = definitions()
    schema = combined_schema(resolved.inputs, objective.inputs)
    placement = InputPlacement(
        GroupRef("cooperate", ((0, 1),)),
        fields=[InputFieldPlacement(tuple(f.path)) for f in schema.fields],
        loss="leader",
    )
    model = resolved.create()
    gradients = []
    for rank in (0, 1):
        with example(
            resolved, objective, "forward_backward", placement=placement, rank=rank
        ) as (binding, adapted):
            captured = capture_training_objective(
                model, binding.objective, binding.arguments(adapted)
            )
            loss, valid = captured.exported.exported_program.module()(
                *binding.arguments(adapted)
            )
            assert valid.item() == 3
            gradients.append(torch.autograd.grad(loss, tuple(model.parameters())))
            if rank == 1:
                assert loss.item() == 0.0
    assert any(torch.count_nonzero(g).item() for g in gradients[0])
    assert all(not torch.count_nonzero(g).item() for g in gradients[1])


def test_padding_claims_and_output_shapes_fail_before_preparation():
    model, objective, _ = definitions()
    unsafe = replace(
        objective, definition=replace(objective.definition, padding_safe=False)
    )
    with (
        pytest.raises(ValueError, match="declare padding safety"),
        example(model, unsafe, "step"),
    ):
        pass
    key = OperationKey("forward", "predictions", "evaluation")
    with example(model, objective, key) as (binding, adapted):
        args = binding.arguments(adapted)
        for bad in (
            {"prediction": torch.ones(4, 4)},
            {"prediction": torch.ones(4, 3, dtype=torch.float64)},
            {"unknown": torch.ones(4, 3)},
        ):
            binding.computation = lambda *args, result=bad: result
            with pytest.raises(ValueError, match="declared"):
                binding.forward(model.create(), *args)


def test_operation_builder_rejects_binding_overrides_before_gpu_access():
    model, objective, _ = definitions()
    with example(model, objective, "step") as (binding, adapted):
        with pytest.raises(ValueError, match="resolved optimizer"):
            build_operation(model.create(), binding, adapted)
        with pytest.raises(ValueError, match="override a prepared binding"):
            build_operation(
                model.create(),
                binding,
                adapted,
                options={"runtime_gradient_scale": False},
            )


def test_operation_inspection_imports_no_framework_or_catalog():
    script = """
import sys
for name in ('torch', 'triton', 'shadowspill_workloads'):
    sys.modules[name] = None
from engine_tests.fixtures.definition_operations import definitions
from shadowspill.training.preparation.operations import resolve_operations
from shadowspill.training.preparation.operation_build import build_operation
model, objective, optimizer = definitions()
assert resolve_operations(objective, None, optimizer=True)[0].outputs == 'metrics'
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


def test_scalar_controls_reject_overflow_or_rounding_outside_the_declared_domain():
    for field, value in (
        (ScalarControl(1.0), 1e100),
        (ScalarControl(1, dtype="int"), 1 << 64),
        (ScalarControl(1.0, minimum=0, minimum_inclusive=False), 1e-100),
    ):
        with pytest.raises(ValueError, match="captured scalar dtype"):
            control_tensors(ControlSchema({"value": field}), {"value": value})
