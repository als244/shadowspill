"""Batch geometry and saved program paths, independent of tensor frameworks."""


def batch_sizes(options):
    count = options["sequences_per_step"]
    limit = min(count, options.get("max_sequences_per_example", count))
    return tuple(size for size in range(1, limit + 1) if count % size == 0)


def program_paths(options):
    kinds = (
        ("training", "evaluation") if options.get("eval_every", 2) else ("training",)
    )
    return tuple(
        f"{candidate['path']}/{kind}.ssp"
        for candidate in build_candidates(options)
        for kind in kinds
    )


def build_candidates(options):
    """Cross explicit build variants with batch geometry; old paths stay valid."""
    variants = options.get("build_variants")
    if variants is None:
        return tuple(
            dict(path=f"batch-{size}", sequences_per_example=size)
            for size in batch_sizes(options)
        )
    return tuple(
        {
            **variant,
            "path": f"{variant['name']}/batch-{size}",
            "sequences_per_example": size,
        }
        for variant in variants
        for size in batch_sizes(options)
    )
