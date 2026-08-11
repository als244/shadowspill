from __future__ import annotations

import torch

from models.pytorch.llama3 import Llama3, Llama3Config


def test_llama3_parameter_catalog_and_gradients() -> None:
    numerical = Llama3Config.numerical()
    with torch.device("meta"):
        model = Llama3(numerical)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_179_699_200

    torch.manual_seed(3)
    tiny = Llama3(Llama3Config(2, 32, 4, 2, 64, 97, max_seq_len=16))
    tokens = torch.randint(0, 97, (2, 5))
    targets = torch.randint(0, 97, (2, 5))
    loss = tiny.loss(tokens, targets)
    loss.backward()
    assert loss.shape == ()
    assert all(parameter.grad is not None for parameter in tiny.parameters())


def test_llama3_packed_matches_independent_sequences() -> None:
    torch.manual_seed(5)
    model = Llama3(Llama3Config(1, 32, 4, 2, 64, 97, max_seq_len=16)).eval()
    first = torch.randint(0, 97, (1, 3))
    second = torch.randint(0, 97, (1, 5))
    packed = torch.cat((first, second), dim=1)
    expected = torch.cat((model(first), model(second)), dim=1)
    actual = model(packed, (3, 5))
    torch.testing.assert_close(actual, expected)


def test_llama3_config_rejects_incompatible_heads() -> None:
    try:
        Llama3Config(1, 30, 4, 2, 64, 97)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("invalid query head geometry was accepted")
