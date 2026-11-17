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
    """Try token capacities per input partition and independent chunk sizes.

    In data parallelism, each partition is one rank. Zero leaves the respective
    computation at full microbatch size. Chunking never splits sequence attention
    boundaries. Retention defaults to cost-input, with chunk-local scratch
    regenerated inside its region. Ordering and the bounded head keep their
    defaults. The result uses portable policies and requires no server or Torch.

    max_sequences_per_microbatch bounds the number of records packed into each
    input buffer. By default the text preset reserves eight times the number of
    maximum-length sequences needed to fill that buffer, capped by the global
    max_sequences limit. max_sequence_length is declared in BatchConfig and
    bounds attention kernels independently of microbatch_tokens. Padding has its
    own reserved metadata entries; it does not consume the record allowance.
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
