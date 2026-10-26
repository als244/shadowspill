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
        f"batch-{size}/{kind}.ssp" for size in batch_sizes(options) for kind in kinds
    )
