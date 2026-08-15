from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    TaskInputProvenance,
    TaskInputRole,
)
from shadowspill.pytorch.contracts import CaptureError, PlanningError, TensorSpec
from shadowspill.pytorch.materialization import representative_cpu_inputs
from shadowspill.pytorch.profiling.inputs import materialize_representative_inputs


class _Add(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_representative_inputs_preserve_exact_alias_views() -> None:
    owner = torch.arange(32, dtype=torch.float32)
    left = owner[2:18].view(4, 4)
    right = owner[4:20].view(4, 4)
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=(left, right),
        input_provenance=(
            TaskInputProvenance(
                TaskInputRole.PARAMETER,
                "left",
                representative_value=left,
            ),
            TaskInputProvenance(
                TaskInputRole.PARAMETER,
                "right",
                representative_value=right,
            ),
        ),
    )

    result = materialize_representative_inputs(artifact, device_ordinal=0)
    actual_left, actual_right = result.arguments
    assert isinstance(actual_left, torch.Tensor)
    assert isinstance(actual_right, torch.Tensor)
    assert actual_left.untyped_storage()._cdata == actual_right.untyped_storage()._cdata
    assert actual_left.storage_offset() == 2
    assert actual_right.storage_offset() == 4
    torch.testing.assert_close(actual_left.cpu(), left)
    torch.testing.assert_close(actual_right.cpu(), right)
    assert {item.value_policy for item in result.summaries} == {"authentic"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_anonymous_float_values_are_deterministic_standard_normal() -> None:
    values = (torch.empty(16_384), torch.empty(16_384))
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=values,
        input_provenance=(
            TaskInputProvenance(TaskInputRole.ACTIVATION, "activation.left"),
            TaskInputProvenance(TaskInputRole.RESIDUAL, "residual.right"),
        ),
    )

    first = materialize_representative_inputs(artifact, device_ordinal=0)
    second = materialize_representative_inputs(artifact, device_ordinal=0)
    for left, right in zip(first.arguments, second.arguments, strict=True):
        assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        host = left.cpu().float()
        assert abs(float(host.mean())) < 0.04
        assert abs(float(host.std()) - 1.0) < 0.04
        assert torch.count_nonzero(host) == host.numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_missing_authentic_value_fails_with_actionable_provenance() -> None:
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=(torch.empty(8), torch.empty(8)),
        input_provenance=(
            TaskInputProvenance(TaskInputRole.PARAMETER, "weight"),
            TaskInputProvenance(TaskInputRole.ACTIVATION, "activation"),
        ),
    )

    with pytest.raises(CaptureError, match=r"position=0, role=parameter"):
        materialize_representative_inputs(artifact, device_ordinal=0)


def test_missing_user_value_uses_explicit_deterministic_fallback() -> None:
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=(torch.empty(32), torch.empty(32)),
        input_provenance=(
            TaskInputProvenance(TaskInputRole.USER_INPUT, "user.left"),
            TaskInputProvenance(TaskInputRole.ACTIVATION, "activation.right"),
        ),
    )

    first = materialize_representative_inputs(artifact, device_ordinal=0)
    second = materialize_representative_inputs(artifact, device_ordinal=0)
    assert first.summaries[0].value_policy == "deterministic_normal_0_1"
    torch.testing.assert_close(first.arguments[0], second.arguments[0])


def test_integer_task_input_requires_authentic_value() -> None:
    integers = (torch.empty(8, dtype=torch.int64),) * 2
    missing = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=integers,
        input_provenance=(
            TaskInputProvenance(TaskInputRole.ACTIVATION, "producer.left"),
            TaskInputProvenance(TaskInputRole.ACTIVATION, "producer.right"),
        ),
    )
    with pytest.raises(CaptureError, match="producer-derived or caller-supplied"):
        materialize_representative_inputs(missing, device_ordinal=0)

    left = torch.tensor([0, 13, 32, 64, 0, 0, 0, 0], dtype=torch.int64)
    right = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
    authentic = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=(left, right),
        input_provenance=(
            TaskInputProvenance(
                TaskInputRole.CONTROL,
                "producer.left",
                representative_value=left,
            ),
            TaskInputProvenance(
                TaskInputRole.CONTROL,
                "caller.right",
                representative_value=right,
            ),
        ),
    )
    result = materialize_representative_inputs(authentic, device_ordinal=0)
    torch.testing.assert_close(result.arguments[0], left)
    torch.testing.assert_close(result.arguments[1], right)
    assert {item.value_policy for item in result.summaries} == {
        "authentic_control"
    }


def test_allocator_failure_is_checked_before_representative_population() -> None:
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=torch.fx.symbolic_trace(_Add()),
        example_inputs=(torch.empty(8), torch.empty(8)),
        input_provenance=(
            TaskInputProvenance(TaskInputRole.ACTIVATION, "activation.left"),
            TaskInputProvenance(TaskInputRole.ACTIVATION, "activation.right"),
        ),
    )
    operations: list[str] = []

    def reject(operation: str) -> None:
        operations.append(operation)
        raise RuntimeError("allocation rejected")

    with pytest.raises(RuntimeError, match="allocation rejected"):
        materialize_representative_inputs(
            artifact,
            device_ordinal=0,
            allocation_check=reject,
        )

    assert len(operations) == 1
    assert "alias group" in operations[0]


def test_tensor_spec_float_values_are_nonzero_and_deterministic() -> None:
    floating = TensorSpec((4096,), torch.float32)
    (first_float,) = representative_cpu_inputs((floating,))
    (second_float,) = representative_cpu_inputs((floating,))
    torch.testing.assert_close(first_float, second_float, rtol=0, atol=0)
    assert abs(float(first_float.mean())) < 0.06
    assert abs(float(first_float.std()) - 1.0) < 0.06


def test_integer_tensor_spec_requires_authentic_caller_value() -> None:
    with pytest.raises(PlanningError, match="caller-supplied tensor values"):
        representative_cpu_inputs((TensorSpec((32,), torch.int64),))

    authentic = torch.tensor([13, 19, 32], dtype=torch.int64)
    (observed,) = representative_cpu_inputs((authentic,))
    assert observed is authentic
