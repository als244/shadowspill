"""Optional token/sequence candidates for text recipes."""


def search_geometries(
    total_sequences_per_step: int,
    *,
    sequence_length: int,
    min_tokens_per_microbatch: int | None = None,
    max_tokens_per_microbatch: int | None = None,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, str], ...]]:
    """Split a step's sequence total into every microbatch/accumulation pair.

    Returns the admitted ``(sequences_per_microbatch, accumulation)`` pairs,
    largest microbatch first, and the pairs the optional token bounds
    skipped, each with its reason. Bounds are in tokens per microbatch, so
    they mean the same thing at every sequence length.
    """

    if total_sequences_per_step < 1:
        raise ValueError("total_sequences_per_step must be positive")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    admitted: list[tuple[int, int]] = []
    skipped: list[tuple[int, int, str]] = []
    for sequences in range(total_sequences_per_step, 0, -1):
        if total_sequences_per_step % sequences:
            continue
        accumulation = total_sequences_per_step // sequences
        tokens = sequences * sequence_length
        if min_tokens_per_microbatch is not None and tokens < min_tokens_per_microbatch:
            skipped.append(
                (
                    sequences,
                    accumulation,
                    f"{tokens} tokens per microbatch is below the minimum"
                    f" of {min_tokens_per_microbatch}",
                )
            )
            continue
        if max_tokens_per_microbatch is not None and tokens > max_tokens_per_microbatch:
            skipped.append(
                (
                    sequences,
                    accumulation,
                    f"{tokens} tokens per microbatch is above the maximum"
                    f" of {max_tokens_per_microbatch}",
                )
            )
            continue
        admitted.append((sequences, accumulation))
    return tuple(admitted), tuple(skipped)
