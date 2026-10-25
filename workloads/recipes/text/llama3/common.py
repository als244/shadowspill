"""Saved recipe settings; importing this module needs no tensor framework."""


def model_config(options):
    from workloads.models.catalog import entry

    return entry(options["model"]).config(max_seq_len=options["sequence_length"])
