"""Batch geometry and saved program paths, independent of tensor frameworks."""


def batch_sizes(options):
    """Validate explicit microbatch sizes and preserve their first-seen order."""
    count = options["sequences_per_step"]
    if type(count) is not int or count <= 0:
        raise ValueError("sequences_per_step must be a positive integer")
    sizes = options["microbatch_sizes"]
    if not isinstance(sizes, list | tuple) or not sizes:
        raise ValueError(
            "microbatch_sizes must be a nonempty list of positive integers"
        )
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("microbatch_sizes must contain positive integers")
    if any(count % size for size in sizes):
        raise ValueError("each microbatch size must divide sequences_per_step")
    return tuple(dict.fromkeys(sizes))


def check_batch_sizes(options, requested):
    """A saved run's geometry belongs to its built Programs, not a replan."""
    if requested is not None and batch_sizes(
        {**options, "microbatch_sizes": requested}
    ) != batch_sizes(options):
        raise ValueError(
            "changed microbatch sizes require a fresh run directory and build"
        )


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
    """Cross explicit build variants with the requested batch geometries."""
    sizes = batch_sizes(options)
    variants = options.get("build_variants")
    if variants is None:
        return tuple(
            dict(path=f"batch-{size}", sequences_per_example=size) for size in sizes
        )
    return tuple(
        {
            **variant,
            "path": f"{variant['name']}/batch-{size}",
            "sequences_per_example": size,
        }
        for variant in variants
        for size in sizes
    )
