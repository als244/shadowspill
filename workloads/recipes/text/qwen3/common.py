"""Saved model and candidate settings; imports require no tensor framework."""

from workloads.recipes.text.phases.candidates import batch_sizes as batch_sizes


def model_config(options):
    from workloads.models.catalog import entry

    return entry(options["model"]).config(max_seq_len=options["sequence_length"])
