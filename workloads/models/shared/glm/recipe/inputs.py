"""Identical rank-local token streams across candidate example geometries."""

import torch


def candidates(
    config,
    *,
    sequence_length,
    sequences_per_step,
    sequences_per_example,
    logical_rank,
    seed,
):
    sizes = tuple(sequences_per_example)
    if sequence_length <= 0 or sequences_per_step <= 0 or not sizes:
        raise ValueError(
            "Sequence length, step size and candidate list must be positive"
        )
    if any(size <= 0 or sequences_per_step % size for size in sizes):
        raise ValueError("Each sequences_per_example must divide sequences_per_step")
    if len(set(sizes)) != len(sizes):
        raise ValueError("Duplicate example-size candidates")
    total = sequence_length * sequences_per_step
    generator = torch.Generator().manual_seed(seed + 1_000_003 * logical_rank)
    tokens = torch.randint(config.vocab_size, (total,), generator=generator)
    targets = torch.randint(config.vocab_size, (total,), generator=generator)
    result = {}
    for size in sizes:
        count = size * sequence_length
        boundaries = tuple(range(0, count + 1, sequence_length))
        cumulative = torch.tensor(boundaries, dtype=torch.int64)
        chunks = torch.tensor(
            [
                (sequence, chunk)
                for sequence in range(size)
                for chunk in range((sequence_length + 63) // 64)
            ],
            dtype=torch.int64,
        )
        result[str(size)] = {
            "args": tuple(
                (
                    tokens[start : start + count].clone(),
                    targets[start : start + count].clone(),
                    boundaries,
                    cumulative.clone(),
                    chunks.clone(),
                )
                for start in range(0, total, count)
            )
        }
    return result
