from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch import CaptureError, ObjectiveError, ObjectiveResult
from shadowspill.pytorch import aot as aot_module
from shadowspill.pytorch.aot import (
    capture_forward,
    capture_training,
    capture_training_objective,
    inference_artifact,
)
from shadowspill.pytorch.capture import (
    GraphArtifact,
    TaskInputProvenance,
    TaskInputRole,
)
from shadowspill.pytorch.fake import fake_cuda_inputs, fake_cuda_model


class _Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(8, 32)
        self.second = nn.Linear(32, 8)

    def forward(self, inputs: torch.Tensor, scale: int) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.first(inputs))
        return self.second(hidden) * scale


def _objective(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, scale: int
) -> ObjectiveResult:
    prediction = model(inputs, scale)
    loss = torch.nn.functional.mse_loss(prediction, targets)
    return ObjectiveResult(loss, {"mean": prediction.mean(), "scale": scale})


def test_fake_export_and_aot_emit_save_and_recompute_graph_pairs() -> None:
    model = _Network()
    inputs = [torch.randn(4, 8), torch.randn(4, 8), 2]
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    fake_inputs = fake_cuda_inputs(inputs, mode)
    with mode:
        capture = capture_training(replica, _objective, fake_inputs)
    assert capture.objective_schema.tensor_metric_positions == (0,)
    assert capture.objective_schema.static_metric_leaves == ((1, 2),)
    assert capture.save_pair.forward.operator_targets
    assert capture.save_pair.backward.operator_targets
    assert capture.recompute_pair.forward.argument_count > 0
    assert capture.recompute_pair.backward.argument_count > 0
    assert capture.save_pair.forward.compatibility_digest
    for pair in (capture.save_pair, capture.recompute_pair):
        assert pair.specialized_unit_tangent_count == 1
        assert pair.backward.argument_count == pair.saved_value_count
        assert "aten.new_ones.default" in pair.backward.operator_targets


def test_objective_export_does_not_construct_a_whole_model_vjp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Network()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(4, 8), torch.randn(4, 8), 2], mode)
    monkeypatch.setattr(
        "shadowspill.pytorch.aot._capture_pair",
        lambda *arguments, **options: pytest.fail(
            f"unexpected whole-model AOT capture {arguments}, {options}"
        ),
    )
    with mode:
        capture = capture_training_objective(replica, _objective, inputs)
    assert capture.exported.user_output_indices
    assert capture.objective_schema.tensor_metric_positions == (0,)


def test_objective_schema_preserves_position_specific_static_metrics() -> None:
    model = _Network()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    first_inputs = fake_cuda_inputs([torch.randn(4, 8), torch.randn(4, 8), 2], mode)
    second_inputs = fake_cuda_inputs([torch.randn(4, 8), torch.randn(4, 8), 3], mode)
    with mode:
        first = capture_training_objective(replica, _objective, first_inputs)
        second = capture_training_objective(replica, _objective, second_inputs)

    assert first.objective_schema.static_metric_leaves == ((1, 2),)
    assert second.objective_schema.static_metric_leaves == ((1, 3),)


def test_forward_export_accepts_static_metadata_and_has_stable_identity() -> None:
    model = _Network().eval()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8), 3], mode)
    with mode, torch.no_grad():
        first = inference_artifact(capture_forward(replica, inputs))
        second = inference_artifact(capture_forward(replica, inputs))
    assert first.compatibility_digest == second.compatibility_digest
    assert "aten.linear.default" in first.operator_targets


def test_structural_identity_includes_input_storage_aliases_and_offsets() -> None:
    class _Pair(nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    graph = torch.fx.symbolic_trace(_Pair())
    owner = torch.arange(20, dtype=torch.float32)
    aliased = GraphArtifact.capture(
        kind="inference",
        graph_module=graph,
        example_inputs=(owner[1:9], owner[2:10]),
    )
    distinct = GraphArtifact.capture(
        kind="inference",
        graph_module=graph,
        example_inputs=(owner[1:9].clone(), owner[2:10].clone()),
    )
    assert aliased.tensor_argument_alias_groups == (0, 0)
    assert distinct.tensor_argument_alias_groups == (0, 1)
    assert aliased.tensor_inputs[0].storage_offset == 1
    assert aliased.tensor_inputs[1].storage_offset == 2
    assert aliased.compatibility_digest != distinct.compatibility_digest


def test_artifact_rebind_preserves_contract_and_rejects_abi_changes() -> None:
    class _Pair(nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    graph = torch.fx.symbolic_trace(_Pair())
    first_owner = torch.arange(20, dtype=torch.float32)
    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=graph,
        example_inputs=(first_owner[1:9], first_owner[2:10]),
    )
    second_owner = torch.arange(20, dtype=torch.float32)
    rebound = artifact.rebind_examples((second_owner[1:9], second_owner[2:10]))

    assert rebound.compatibility_digest == artifact.compatibility_digest
    assert rebound.storage_contract is artifact.storage_contract
    assert rebound.example_arguments[0].untyped_storage()._cdata != (
        artifact.example_arguments[0].untyped_storage()._cdata
    )
    with pytest.raises(CaptureError, match="geometry"):
        artifact.rebind_examples((second_owner[1:8], second_owner[2:9]))
    left_owner = torch.arange(20, dtype=torch.float32)
    right_owner = torch.arange(20, dtype=torch.float32)
    with pytest.raises(CaptureError, match="aliases"):
        artifact.rebind_examples((left_owner[1:9], right_owner[2:10]))


def test_saved_registered_state_preserves_its_aot_view_values() -> None:
    class _Transpose(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.t()

    value = torch.arange(30, dtype=torch.float32).view(3, 10)
    artifact = GraphArtifact.capture(
        kind="forward",
        graph_module=torch.fx.symbolic_trace(_Transpose()),
        example_inputs=(value,),
    )
    view = artifact.storage_contract.output_views[0]
    provenance = TaskInputProvenance(
        TaskInputRole.PARAMETER,
        "model.weight",
        representative_value=value,
    )

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    with mode:
        saved = aot_module._saved_input_view_provenance(provenance, view)

    assert saved.representative_value is not None
    torch.testing.assert_close(saved.representative_value, value.t())
    assert saved.representative_value.untyped_storage()._cdata == (
        value.untyped_storage()._cdata
    )


def test_input_compatibility_digest_is_offline_and_alias_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pair(nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return left + right

    graph = torch.fx.symbolic_trace(_Pair())
    owner = torch.arange(20, dtype=torch.float32)
    monkeypatch.setattr(
        graph,
        "forward",
        lambda *args: pytest.fail(f"unexpected graph execution: {args}"),
    )
    aliased = GraphArtifact.input_compatibility_digest(
        graph_module=graph,
        example_inputs=(owner[1:9], owner[2:10]),
    )
    distinct = GraphArtifact.input_compatibility_digest(
        graph_module=graph,
        example_inputs=(owner[1:9].clone(), owner[2:10].clone()),
    )
    assert aliased != distinct


@torch.library.custom_op("shadowspill_test::affine", mutates_args=())
def _affine(value: torch.Tensor, bias: float) -> torch.Tensor:
    return value + bias


@_affine.register_fake
def _affine_fake(value: torch.Tensor, bias: float) -> torch.Tensor:
    return torch.empty_like(value)


class _CustomOperationModule(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _affine(value, 1.25)


@torch.library.custom_op("shadowspill_test::saved_square", mutates_args=())
def _saved_square(value: torch.Tensor) -> torch.Tensor:
    return value.square()


@_saved_square.register_fake
def _saved_square_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


def _saved_square_setup(
    ctx: object, inputs: tuple[torch.Tensor], output: torch.Tensor
) -> None:
    del output
    ctx.save_for_backward(inputs[0])  # type: ignore[attr-defined]


def _saved_square_backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
    (value,) = ctx.saved_tensors  # type: ignore[attr-defined]
    return 2 * value * gradient


_saved_square.register_autograd(
    _saved_square_backward,
    setup_context=_saved_square_setup,
)


class _SavedOperationModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _saved_square(value * self.weight)


def test_unrelated_registered_custom_operation_exports_as_opaque() -> None:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(_CustomOperationModule(), mode)
    inputs = fake_cuda_inputs([torch.randn(2, 3)], mode)
    with mode, torch.no_grad():
        artifact = inference_artifact(capture_forward(model, inputs))
    assert any(
        "shadowspill_test.affine" in target for target in artifact.operator_targets
    )


def test_save_and_recompute_capture_isolates_custom_autograd_graphs() -> None:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(_SavedOperationModule(), mode)
    inputs = fake_cuda_inputs([torch.randn(2, 3)], mode)

    with mode:
        capture = capture_training(
            model, lambda module, value: module(value).sum(), inputs
        )

    assert capture.save_pair.backward.operator_targets
    assert capture.recompute_pair.backward.operator_targets
    assert capture.save_pair.specialized_unit_tangent_count == 1
    assert capture.recompute_pair.specialized_unit_tangent_count == 1


def test_objective_schema_reconstructs_metrics_and_rejects_count_change() -> None:
    model = _Network()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8), torch.randn(2, 8), 2], mode)
    with mode:
        capture = capture_training(replica, _objective, inputs)
        metric = torch.ones((), device="cuda")
        rebuilt = capture.objective_schema.rebuild_metrics((metric,))
    assert rebuilt["mean"] is metric
    assert rebuilt["scale"] == 2
    with pytest.raises(ObjectiveError, match="count"):
        capture.objective_schema.rebuild_metrics(())


@pytest.mark.parametrize(
    ("objective", "message"),
    [
        (lambda model, value: 3, "tensor"),
        (lambda model, value: model(value, 1), "scalar"),
        (lambda model, value: torch.ones((), dtype=torch.int64), "floating"),
        (lambda model, value: value.detach().sum(), "gradients"),
    ],
)
def test_invalid_objectives_fail_during_capture(
    objective: object, message: str
) -> None:
    model = _Network()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8)], mode)
    with mode, pytest.raises(ObjectiveError, match=message):
        capture_training(replica, objective, inputs)  # type: ignore[arg-type]
