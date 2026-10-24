"""Saved recipe settings; importing this module needs no tensor framework."""

import json
from pathlib import Path


def settings(directory):
    root = Path(directory)
    return root, json.loads((root / "config.json").read_text())


def model_config(options):
    from workloads.models.catalog import entry

    return entry(options["model"]).config(max_seq_len=options["sequence_length"])


def loss(model, tokens, targets, *, seq_lens, normalizer):
    return model.loss(tokens, targets, seq_lens=seq_lens, reduction="sum") * normalizer
