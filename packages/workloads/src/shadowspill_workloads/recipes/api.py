"""Readable Python front door to the model-independent saved-phase runner."""

from collections.abc import Mapping

from shadowspill.training.preparation import run

from .models import Model


def train(model: Model, config: Mapping[str, object], *, phase: str = "all"):
    """Build, measure, select and train using a named Python Model source.

    ``config`` contains ordinary data, search, memory and training choices.
    Shape/precision choices are frozen at build; a saved run may be replanned at
    another budget or resumed without reconstructing its Programs.
    """
    if not isinstance(model, Model):
        raise TypeError(
            "train expects a Model source, such as Model.from_checkpoint(...)"
        )
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    settings = dict(model._settings)
    duplicates = settings.keys() & config.keys()
    if duplicates:
        raise ValueError(
            f"model settings belong on the Model source: {sorted(duplicates)}"
        )
    return run(model._adapter, {**settings, **config}, phase=phase)
