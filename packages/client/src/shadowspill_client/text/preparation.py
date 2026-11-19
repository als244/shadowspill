"""Portable build candidates for the supplied packed text workloads."""

from collections.abc import Sequence
from itertools import product

from .._records import integer
from ..config.preparation import BuildCandidates, BuildCase, BuildOptions, Policy


def build_candidates(
    *,
    microbatch_tokens: Sequence[int],
    max_sequences_per_microbatch: int | None = None,
    projection_chunks: Sequence[int] = (0,),
    moe_chunks: Sequence[int] = (0,),
    retention: Sequence[str] = ("cost-input",),
) -> BuildCandidates:
    """Make the build candidates for packed text from sizes and chunk choices.

    Every combination of ``microbatch_tokens``, ``projection_chunks``,
    ``moe_chunks`` and ``retention`` becomes one `BuildCase`, for
    ``Preparation(candidates=...)``. Chunking never splits attention across
    sequence boundaries, and a chunked case regenerates chunk-local scratch
    inside each chunk. Attention kernels are bounded by the batch's
    ``max_sequence_length`` limit, not by these sizes. Building the candidates
    needs no service or PyTorch.

    Parameters
    ----------
    microbatch_tokens : sequence of int
        Token capacities to try for one microbatch of each input partition,
        which under `data_parallel` is one worker.
    max_sequences_per_microbatch : int, optional
        Most sequences packed into one microbatch; padding does not count
        against it. By default the text preset uses eight times the number of
        maximum-length sequences that fill the microbatch, capped by the
        batch's ``max_sequences`` limit.
    projection_chunks : sequence of int, default (0,)
        Chunk sizes, in tokens, to try for projections. Zero leaves them at
        the full microbatch.
    moe_chunks : sequence of int, default (0,)
        Chunk sizes, in tokens, to try for expert computations, independently
        of ``projection_chunks``. Zero leaves them at the full microbatch.
    retention : sequence of str, default ("cost-input",)
        Retention policies to try. Each names what a stage keeps for backward
        in its saving and its recomputing variant: ``"cost-input"`` (values
        chosen by cost; inputs only), ``"cost-attention"`` (values chosen by
        cost; attention outputs) or ``"attention-input"`` (attention outputs;
        inputs only).

    Returns
    -------
    BuildCandidates
        One case per combination, with portable policies.

    Raises
    ------
    ValueError
        ``microbatch_tokens`` is empty or holds a size below one or a repeated
        size, ``max_sequences_per_microbatch`` is below one, a chunk list is empty
        or holds a negative or repeated size, or ``retention`` is empty or
        holds an unknown or repeated name.
    """
    sizes = BuildCandidates(microbatch_sizes=microbatch_tokens).microbatch_sizes
    assert sizes is not None
    if max_sequences_per_microbatch is not None:
        integer(max_sequences_per_microbatch, "max_sequences_per_microbatch", minimum=1)
    for label, choices in (
        ("projection_chunks", projection_chunks),
        ("moe_chunks", moe_chunks),
    ):
        if not choices:
            raise ValueError(f"{label} must contain at least one size")
        for size in choices:
            integer(size, label)
        if len(set(choices)) != len(choices):
            raise ValueError(f"{label} must not contain duplicates")
    if (
        not retention
        or any(
            choice not in ("cost-input", "cost-attention", "attention-input")
            for choice in retention
        )
        or len(set(retention)) != len(retention)
    ):
        raise ValueError("retention requires distinct supported text policy names")
    return BuildCandidates(
        cases=[
            BuildCase(
                tokens,
                BuildOptions(
                    chunking=(
                        None
                        if projection == expert == 0
                        else Policy(
                            "shadowspill/text_chunks@1",
                            {
                                "token_capacity": tokens,
                                "chunk_size": projection or tokens,
                                "expert_chunk_size": expert or tokens,
                            },
                        )
                    ),
                    graph_pair_policy=Policy(
                        "shadowspill/text_retention@1",
                        {"name": choice, "chunked": projection != 0 or expert != 0},
                    ),
                ),
                max_datums=max_sequences_per_microbatch,
            )
            for tokens, projection, expert, choice in product(
                sizes, projection_chunks, moe_chunks, retention
            )
        ]
    )
