"""Which geometries a step can be split into, and the walks over each."""

from torch import OutOfMemoryError

from shadowspill.planner import (
    StepDataOrdering,
)

# a point the planner refuses, for whatever reason it gives, is recorded and
# the sweep goes on; ProblemPreparationError is one such RuntimeError
_REJECTED = (RuntimeError,)


def _device_exhausted(error: BaseException) -> bool:
    """Whether a build failed because the device ran out of memory.

    Profiling runs a task's real kernels, so the largest geometries can
    exhaust the device before any plan exists. The frontend wraps what a
    phase raised, chaining the original, so the exhaustion is found by
    walking the chain rather than by matching the outermost type.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OutOfMemoryError):
            return True
        current = current.__cause__ or current.__context__
    return False


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


def default_orderings(accumulation: int) -> tuple[StepDataOrdering, ...]:
    """Every ``depth x breadth`` factor pair of the accumulation count.

    Depth-first first, so the order every step ran in before there was a
    choice is the first program built and the bound the rest are searched
    against; then wider and wider passes, down to one pass over every
    microbatch. The flags stay at their defaults throughout: the search does
    not toggle them, because pairing the loss won every cell it was measured
    in and the reversed walk cost nothing.
    """
    return tuple(
        StepDataOrdering(accumulation // breadth, breadth)
        for breadth in range(1, accumulation + 1)
        if accumulation % breadth == 0
    )
