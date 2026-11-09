"""Text-policy axis selection distinguishes tokens from matching feature sizes."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.artifacts import TaskInputProvenance
from shadowspill.pytorch.capture.gradients import accumulate_gradient_outputs
from shadowspill.pytorch.chunking.regions import region_artifact
from shadowspill.task.inputs import TaskInputRole
from tests.shadowspill.pytorch.chunking.test_composition import executable
from workloads.models.shared.execution import text_execution_chunks


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
