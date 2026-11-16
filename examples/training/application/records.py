"""Portable application settings and data identity, independent of model code."""

import json
from dataclasses import asdict


def record(settings):
    return json.loads(json.dumps(asdict(settings)))


def data_identity(source, validation):
    return {
        "train": source.identity,
        "validation": None if validation is None else validation.identity,
    }
