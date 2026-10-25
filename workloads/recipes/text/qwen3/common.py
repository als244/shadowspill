"""Saved model and candidate settings; imports require no tensor framework."""


def batch_sizes(options):
    count = options["sequences_per_step"]
    limit = min(count, options["max_sequences_per_example"])
    return tuple(size for size in range(1, limit + 1) if count % size == 0)


def model_config(options):
    from workloads.models.catalog import entry

    return entry(options["model"]).config(max_seq_len=options["sequence_length"])
