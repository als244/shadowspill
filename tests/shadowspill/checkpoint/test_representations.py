"""Quantized checkpoints contain physical tensors and use existing model types."""

from __future__ import annotations

import pytest
import torch

from shadowspill.checkpoint.representations import (
    decode_tensor_state,
    encode_tensor_state,
)
from tests.shadowspill.pytorch.representation_fixtures import ScaledWeight, model


def test_weights_only_checkpoint_preserves_nested_component_bytes(tmp_path):
    source = model()
    source.weight = torch.nn.Parameter(
        ScaledWeight(source.weight.detach(), torch.tensor(2.0))
    )
    source.tied = source.weight
    packed = encode_tensor_state(source.state_dict())
    path = tmp_path / "weights.pt"
    torch.save(packed, path)
    loaded = torch.load(path, weights_only=True, mmap=True)
    actual = decode_tensor_state(loaded, source.state_dict())
    assert isinstance(actual["weight"].payload, ScaledWeight)
    assert (
        actual["weight"].payload.payload.data_ptr()
        == actual["tied"].payload.payload.data_ptr()
    )
    torch.testing.assert_close(
        actual["weight"].dense(), source.weight.dense(), rtol=0, atol=0
    )
    assert (
        actual["weight"].payload.payload.data_ptr()
        != source.weight.payload.payload.data_ptr()
    )
    wrong = model()
    with pytest.raises(ValueError, match="representation differs"):
        decode_tensor_state(loaded, wrong.state_dict())
