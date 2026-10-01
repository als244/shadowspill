"""Task orderings over any microbatch collection."""

from shadowspill.planner import StepDataOrdering


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
