"""Bounded text policy/chunk candidate grids, independent of tensor frameworks."""


def build_variants(options, *, alignment=1):
    """Persist a bounded, deduplicated candidate product without importing Torch."""
    result = {}
    for policy in options.get("graph_pair_policies", ["cost-input"]):
        if policy not in {"cost-input", "cost-attention", "attention-input"}:
            raise ValueError("unknown graph-pair policy")
        for size in options.get("chunk_sizes", [0]):
            for requested in options.get("moe_chunk_sizes", [0]):
                if (
                    type(size) is not int
                    or type(requested) is not int
                    or min(size, requested) < 0
                ):
                    raise ValueError(
                        "execution chunk sizes must be nonnegative integers"
                    )
                expert = (requested or size) if size else 0
                if size and any(
                    value % alignment
                    for value in (size, expert, options["sequence_length"])
                ):
                    raise ValueError(
                        f"sequence and chunk sizes must be divisible by {alignment}"
                    )
                if size and expert < size:
                    raise ValueError(
                        "MoE chunk size must be at least the projection chunk size"
                    )
                name = f"{policy}-chunk-{size}-moe-{expert}"
                result[name] = dict(
                    name=name,
                    chunk_size=size,
                    moe_chunk_size=expert,
                    graph_pair_policy=policy,
                )
    if not result:
        raise ValueError("at least one build candidate is required")
    return list(result.values())
