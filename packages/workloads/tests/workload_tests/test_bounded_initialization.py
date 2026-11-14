"""Generated values survive loader shards, batching and storage dtype changes."""

from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from shadowspill.checkpoint.initialization import fill_generated
from shadowspill.execution.tensors import TensorChunk


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_random_values_are_identical_across_partial_rng_blocks_and_copy_batches(
    source_dtype,
):
    count = 257 * 35 + 17
    request = TensorChunk("weight", (count,), torch.bfloat16, 0, count)
    whole = torch.empty(count, dtype=request.dtype)
    shards = torch.empty_like(whole)
    visited = []

    def sample(name, shape, start, stop, generator):
        visited.append((start, stop))
        return torch.randn(stop - start, generator=generator).to(source_dtype)

    def fill(current, destination):
        fill_generated(
            current, destination, sample=sample, seed=123, block_elements=257
        )

    fill(request, whole)
    assert all(stop - start <= 257 for start, stop in visited)
    # Loader boundaries bisect RNG blocks and the sixteen-block copy batches.
    boundaries = [0, 113, 257 * 16 + 29, count - 23, count]
    for start, stop in pairwise(boundaries):
        fill(replace(request, offset=start), shards[start:stop])
    torch.testing.assert_close(whole, shards, rtol=0, atol=0)
    assert whole.std() > 0


def test_constants_skip_sampling_and_preserve_fp32_rounding_and_chunk_validation():
    request = TensorChunk("scale", (31,), torch.float64, 7, 31)
    destination = torch.empty(17, dtype=request.dtype)

    def sample(*args):
        pytest.fail("constant initialization must not generate RNG blocks")

    fill_generated(request, destination, sample=sample, seed=123, constant=0.02)
    expected = torch.full_like(destination, torch.tensor(0.02).item())
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="contiguous CPU tensor chunk"):
        fill_generated(
            replace(request, offset=30),
            destination,
            sample=sample,
            seed=123,
            constant=0,
        )
