from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch.capture.artifacts import (
    GraphArtifact,
    TaskInputProvenance,
    TaskInputRole,
)
from shadowspill.pytorch.compilation.representative import (
    materialize_representative_inputs,
)
from shadowspill.pytorch.contracts import CaptureError, TensorSpec
from shadowspill.pytorch.materialization import representative_cpu_inputs


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


def test_tensor_spec_values_are_nonzero_deterministic_and_low_domain() -> None:
    floating = TensorSpec((4096,), torch.float32)
    integer = TensorSpec((32,), torch.int64)
    first_float, first_int = representative_cpu_inputs((floating, integer))
    second_float, second_int = representative_cpu_inputs((floating, integer))
    torch.testing.assert_close(first_float, second_float, rtol=0, atol=0)
    torch.testing.assert_close(first_int, second_int, rtol=0, atol=0)
    assert abs(float(first_float.mean())) < 0.06
    assert abs(float(first_float.std()) - 1.0) < 0.06
    assert set(first_int.tolist()) == {0, 1}
