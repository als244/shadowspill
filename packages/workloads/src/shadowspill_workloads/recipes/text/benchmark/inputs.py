"""Rank-local synthetic data, reused exactly across geometry candidates."""

import torch

from ..geometry import search_geometries


def candidates(manifest, *, size, minimum, maximum, seed, logical_rank):
    length, count = manifest.sequence_length, manifest.sequences_per_microbatch
    if size is not None and (size < 1 or count % size):
        raise ValueError("sequences_per_example must divide sequences_per_step")
    geometries, skipped = search_geometries(
        count,
        sequence_length=length,
        min_tokens_per_microbatch=minimum,
        max_tokens_per_microbatch=maximum,
    )
    if size is not None:
        geometries = ((size, count // size),)
    if not geometries:
        raise ValueError("no text geometry satisfies the token bounds")
    generator = torch.Generator().manual_seed(seed * 1_000_003 + logical_rank)
    tensors = [
        torch.randint(
            manifest.model_config.vocab_size, (length * count,), generator=generator
        )
        for _ in range(2)
    ]
    inputs = {}
    for sequences, _ in geometries:
        capacity = sequences * length
        inputs[str(sequences)] = {
            "args": [
                (
                    *(
                        tensor[start : start + capacity].clone().unsqueeze(0)
                        for tensor in tensors
                    ),
                    (length,) * sequences,
                )
                for start in range(0, length * count, capacity)
            ]
        }
    return inputs, skipped
