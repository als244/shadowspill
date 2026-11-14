"""The MoE router keeps physical storage and logical gradients through AOT."""

import pytest
import torch

from shadowspill.pytorch.capture.objective import capture_training
from shadowspill.pytorch.representations import tensor_components


def test_packed_router_save_and_recompute_preserve_gradients():
    router = pytest.importorskip("mlops.kernels.router")
    weights = pytest.importorskip("mlops.expert_parallel.parameters")

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                weights.BF16ComputeWeight(torch.randn(3, 16, dtype=torch.bfloat16))
            )

        def forward(self, x):
            return router.routed_logits(x, self.weight)

    torch.manual_seed(31)
    model = Network()
    x = torch.randn(7, 16, dtype=torch.bfloat16, requires_grad=True)
    expected = (x.float() @ model.weight._data.float().t()).square().sum()
    dy = 2 * (x.float() @ model.weight._data.float().t())
    expected_gradients = (
        dy.t() @ x.float(),
        (dy @ model.weight._data.float()).to(x.dtype),
    )
    capture = capture_training(model, lambda m, x: m(x).square().sum(), (x,))
    arguments = tuple(
        component
        for value in capture.exported.flat_inputs
        for _, component in tensor_components(value)
    )
    for pair in (capture.save_pair, capture.recompute_pair):
        outputs = pair.forward.graph_module(*arguments)
        torch.testing.assert_close(outputs[0], expected)
        gradients = pair.backward.graph_module(*outputs[-pair.saved_value_count :])
        for actual, reference in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, reference)
