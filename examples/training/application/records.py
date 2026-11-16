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


def save_checkpoint(training, directory, *, step, application, settings, data):
    """Save resumable server state, then publish its portable client reference."""
    settings = record(settings)
    saved = training.save_checkpoint(
        name=f"step-{step:08d}",
        metadata={
            "application": application,
            "settings": settings,
            "data": data,
            "next_update": step,
        },
    ).result()
    (directory / f"step-{step:08d}.json").write_text(
        json.dumps(
            {
                "application": application,
                "settings": settings,
                "checkpoint": saved.reference.to_dict(),
                "plans": training.plans.reference.to_dict(),
            },
            indent=2,
        )
        + "\n"
    )
    return saved
