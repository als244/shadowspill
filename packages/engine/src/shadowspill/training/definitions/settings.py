"""Structural options belong to installed definitions, never model-name branches."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from shadowspill_client import Model, Optimizer
from shadowspill_client._records import Value, mapping
from shadowspill_client.config.precision import parameter_dtypes


def constructor_config(
    settings: Model | Optimizer,
    configure: Callable[[Any], Mapping[str, Value]] | None,
) -> Mapping[str, Value]:
    """Resolve all structural settings into the ordinary factory configuration.

    A definition's optional CPU hook validates every supported option and must
    reject unsupported ones. Its result passes the same typed config factory as
    ordinary settings and must be stable when those defaults are supplied again.
    Initial weight loading stays separate from model construction/capture.
    """
    if isinstance(settings, Model) and "parameter_dtypes" in settings.precision:
        parameter_dtypes(settings.precision["parameter_dtypes"])
        settings = replace(
            settings,
            precision={
                key: value
                for key, value in settings.precision.items()
                if key != "parameter_dtypes"
            },
        )
    if configure is not None:
        return mapping(configure(settings), "configured definition settings")
    fields = (
        ("lora", "implementation", "precision")
        if isinstance(settings, Model)
        else ("groups", "precision")
    )
    unsupported = [
        name
        for name in fields
        if getattr(settings, name) is not None
        and (name not in ("precision", "groups") or getattr(settings, name))
    ]
    if unsupported:
        raise ValueError(
            f"definition {settings.definition} has no settings hook for {unsupported}"
        )
    return settings.config
