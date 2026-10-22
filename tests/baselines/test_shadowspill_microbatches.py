"""Equal microbatches share capture contracts without sharing their data."""

import pytest
import torch

from baselines.engines.shadowspill.microbatches import split_microbatches
from shadowspill.pytorch.guards import capture_training_signatures


@pytest.mark.parametrize("size", [4096, 8192, 16384, 32768, 65536])
def test_microbatch_positions_have_one_signature_and_distinct_values(size):
    tokens = torch.arange(65536).reshape(1, -1)
    targets = tokens + 1
    batches = list(
        split_microbatches((tokens, targets), size=size, sequence_length=4096)
    )
    inputs = [item for item, _scale in batches]
    assert len(inputs) == 65536 // size
    assert len({item.digest for item in capture_training_signatures(inputs)}) == 1
    for position, ((actual, labels, lengths), scale) in enumerate(batches):
        start = position * size
        assert torch.equal(actual, tokens[:, start : start + size])
        assert torch.equal(labels, targets[:, start : start + size])
        assert actual.storage_offset() == labels.storage_offset() == 0
        assert actual.untyped_storage().nbytes() == size * tokens.element_size()
        assert lengths == (4096,) * (size // 4096)
        assert scale == 1.0
    inputs[0][0].zero_()
    assert tokens[0, 1].item() == 1
    if len(inputs) > 1:
        assert inputs[1][0][0, 0].item() == size
