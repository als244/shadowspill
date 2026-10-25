"""Candidate batch geometry, independent of models and tensor frameworks."""


def batch_sizes(options):
    count = options["sequences_per_step"]
    return tuple(size for size in range(1, count + 1) if count % size == 0)
