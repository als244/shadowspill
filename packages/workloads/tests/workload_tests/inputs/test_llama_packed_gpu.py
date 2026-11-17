"""Small real-provider packing checks prepared before service qualification."""

import pytest
import torch

from shadowspill_workloads.models.llama3.config import Config
from shadowspill_workloads.models.llama3.implementations.mlops import Model


@pytest.mark.cuda
@pytest.mark.parametrize("capacity", [1, 4])
def test_single_token_with_empty_reserved_sequence_slots(capacity):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(17)
    model = Model(Config(1, 16, 2, 1, 24, 19, max_seq_len=capacity)).cuda().bfloat16()
    tokens = torch.tensor([[3]], device="cuda")
    target = torch.tensor([[4]], device="cuda")
    # The service reserves its max_sequences slots even for a one-token call.
    lengths = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device="cuda")
    actual = model.target_logprobs(tokens, target, lengths)
    expected = model.target_logprobs(tokens, target)
    torch.testing.assert_close(actual, expected)
    first = torch.autograd.grad(actual.sum(), tuple(model.parameters()))
    second = torch.autograd.grad(expected.sum(), tuple(model.parameters()))
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b)
