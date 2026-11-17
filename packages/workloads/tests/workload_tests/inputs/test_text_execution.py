"""Text-policy axis selection distinguishes tokens from matching feature sizes."""

from dataclasses import replace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from engine_tests.pytorch.chunking.test_composition import executable
from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import GraphArtifact, TaskInputProvenance
from shadowspill.pytorch.capture.gradients import accumulate_gradient_outputs
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.task.inputs import TaskInputRole
from shadowspill_workloads.policies.text.execution import (
    TextExecutionBoundaries,
    text_execution_chunks,
)


@pytest.mark.parametrize("batch", [1, 2])
def test_forward_loss_joins_completed_token_reduction_once(batch):
    # Evaluation combines a complete head loss with auxiliary * valid tokens.
    # A shared loss inside the token loop would be counted once per chunk.
    targets = torch.arange(batch * 17).reshape(batch, 17)
    targets[:, 3::5] = -100
    inputs = (targets, torch.tensor(2.5), torch.tensor(7.0))

    def loss(targets, auxiliary, head):
        return (head + 0.001 * auxiliary * targets.ne(-100).sum()) / (batch * 17)

    artifact = GraphArtifact.capture(
        kind="inference",
        graph_module=make_fx(loss)(*inputs),
        example_inputs=inputs,
    )
    policy = text_execution_chunks(
        sequence_length=17,
        batch_size=batch,
        chunk_size=8,
        expert_chunk_size=16,
        rules=(),
        expert_targets=(),
    )
    result = region_artifact(artifact, policy, stage_id="loss", variant=None)
    assert any(child.chunking for child in result.regions.children)
    joins = [
        child
        for child in result.regions.children
        if "aten.add.Tensor" in child.operator_targets
    ]
    assert len(joins) == 1 and joins[0].chunking is None
    torch.testing.assert_close(executable(result)(*inputs), (loss(*inputs),))


def test_token_position_controls_are_sliced_but_rotary_tables_are_shared():
    from shadowspill.pytorch import ChunkAxis, StageInfo
    from shadowspill.pytorch.capture.artifacts import TensorGeometry

    # Positions belong to each token. Tables have the same extent but use
    # absolute positions, so slicing them would give wrong values or OOB reads.
    values = (
        torch.empty(2, 17, 4, 32),
        torch.arange(17).repeat(2),
        torch.empty(17, 8),
        torch.tensor([0, 17, 34]),
    )
    policy = text_execution_chunks(
        sequence_length=17,
        batch_size=2,
        chunk_size=8,
        expert_chunk_size=16,
        rules=(),
        expert_targets=(),
    )
    chunks = policy.chunking.configure_stage(
        StageInfo(
            "rope",
            (),
            tuple(TensorGeometry.from_tensor(value) for value in values),
            (
                TaskInputRole.ACTIVATION,
                TaskInputRole.CONTROL,
                TaskInputRole.BUFFER,
                TaskInputRole.CONTROL,
            ),
            ("mlops.partial_rope_builtin_triton_fwd.default",),
            False,
        )
    )
    assert chunks is not None
    assert dict(chunks.inputs) == {0: ChunkAxis(1), 1: ChunkAxis(0, before=2)}


@pytest.mark.parametrize("save", [False, True])
def test_parameter_extent_matching_tokens_keeps_shared_accumulation(save):
    torch.manual_seed(317)
    extent = 17
    inputs = (
        torch.randn(extent, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(extent, 3, dtype=torch.float64, requires_grad=True),
    )

    def layer(value, weight):
        return torch.sigmoid(value @ weight.T)

    expected = layer(*inputs)
    tangent = torch.randn_like(expected)
    gradients = torch.autograd.grad(expected, inputs, tangent)
    pair = capture_graph_pair(
        make_fx(layer)(*inputs),
        inputs,
        original_output=expected,
        memory_budget=float(save),
        input_provenance=(
            TaskInputProvenance(TaskInputRole.ACTIVATION),
            TaskInputProvenance(TaskInputRole.PARAMETER),
        ),
    )
    policy = text_execution_chunks(
        sequence_length=extent,
        batch_size=1,
        chunk_size=5,
        expert_chunk_size=5,
        rules=(),
        expert_targets=(),
        variants=("save", "recompute"),
    )
    with torch.no_grad():
        forward = region_artifact(
            pair.forward,
            policy,
            stage_id="projection",
            variant="save" if save else "recompute",
        )
        domains = forward.regions.output_domains
        creator = region_artifact(
            pair.backward,
            policy,
            stage_id="projection",
            variant="save" if save else "recompute",
            input_domains=(*domains[-pair.saved_value_count :], domains[0]),
            domain_rules=forward.regions.domain_rules,
            gradient_leaves=(1,),
        )
        backward = region_artifact(
            accumulate_gradient_outputs(pair.backward, (0, 1)),
            policy,
            stage_id="projection",
            variant="save" if save else "recompute",
            input_domains=(
                *domains[-pair.saved_value_count :],
                domains[0],
                *forward.regions.input_domains,
            ),
            domain_rules=forward.regions.domain_rules,
            gradient_leaves=(1,),
        )
        actual = executable(forward)(*inputs)
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(
            executable(creator)(*actual[-pair.saved_value_count :], tangent),
            gradients,
            rtol=1e-12,
            atol=1e-12,
        )
        priors = [torch.randn_like(value) for value in gradients]
        initial = [value.clone() for value in priors]
        runner = executable(backward)
        for count in (1, 2):
            result = runner(*actual[-pair.saved_value_count :], tangent, *priors)
            torch.testing.assert_close(
                result,
                tuple(
                    value + count * grad
                    for value, grad in zip(initial, gradients, strict=True)
                ),
                rtol=1e-12,
                atol=1e-12,
            )
            assert result[1].data_ptr() == priors[1].data_ptr()
            priors = list(result)


@torch.library.custom_op("text_execution_tests::projection_vjp", mutates_args=())
def projection_vjp(
    value: torch.Tensor, tangent: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return tangent @ weight.T, value.T @ tangent


@projection_vjp.register_fake
def projection_vjp_fake(value, tangent, weight):
    return torch.empty_like(value), torch.empty_like(weight)


@pytest.mark.parametrize("batch", [1, 2])
def test_opaque_parameter_gradient_matching_tokens_remains_shared(batch):
    torch.manual_seed(731)
    sequence = 17
    rows = batch * sequence
    value = torch.randn(rows, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(3, rows, dtype=torch.float64, requires_grad=True)
    tangent = torch.randn(rows, rows, dtype=torch.float64)
    expected = torch.autograd.grad(value.sin() @ weight, (value, weight), tangent)

    def backward(value, tangent, weight):
        dx, dw = projection_vjp(value.sin(), tangent, weight)
        return dx * value.cos(), dw.clone()

    arguments = (value.detach(), tangent, weight.detach())
    with FakeTensorMode() as mode:
        examples = tuple(mode.from_tensor(value) for value in arguments)
        base = GraphArtifact.capture(
            kind="backward",
            graph_module=make_fx(backward)(*examples),
            example_inputs=examples,
            input_provenance=(
                TaskInputProvenance(TaskInputRole.ACTIVATION),
                TaskInputProvenance(TaskInputRole.ACTIVATION),
                TaskInputProvenance(TaskInputRole.PARAMETER),
            ),
        )
    policy = text_execution_chunks(
        sequence_length=sequence,
        batch_size=batch,
        chunk_size=5,
        expert_chunk_size=5,
        rules=(),
        expert_targets=(),
    )
    target = "text_execution_tests.projection_vjp.default"

    class WholeProjection:
        def configure_stage(self, stage):
            if target in stage.operators:
                return None
            return policy.chunking.configure_stage(stage)

    regions = replace(
        policy,
        partition=TextExecutionBoundaries(sequence, (), (target, "aten.add_.Tensor")),
        chunking=WholeProjection(),
    )
    with torch.no_grad():
        creator = region_artifact(
            base,
            regions,
            stage_id="projection",
            variant="recompute",
            gradient_leaves=(1,),
        )
        torch.testing.assert_close(executable(creator)(*arguments), expected)
        # The opaque VJP returns both a token adjoint and a parameter adjoint.
        # Marking its entire output tuple as shared would hide a second bug.
        assert any(child.chunking for child in creator.regions.children)
        accumulating = region_artifact(
            accumulate_gradient_outputs(base, (1,)),
            regions,
            stage_id="projection",
            variant="recompute",
            gradient_leaves=(1,),
        )
        prior = torch.randn_like(weight)
        initial = prior.clone()
        runner = executable(accumulating)
        for count in (1, 2):
            result = runner(*arguments, prior)
            torch.testing.assert_close(result[0], expected[0])
            torch.testing.assert_close(result[1], initial + count * expected[1])
            assert result[1].data_ptr() == prior.data_ptr()
