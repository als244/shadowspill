"""Build a run from a JSON config: its settings, and the objects they name.

A config is one JSON object of ``Trainer`` arguments. Three forms name Python
objects, so that a config can say which model, objective, optimizer and data a
run uses without this package knowing any of them:

- ``"@module:name"`` is the object ``name`` in ``module``;
- ``{"@call": "module:name", ...}`` is what calling it with the other keys as
  keyword arguments returns;
- ``{"@partial": "module:name", ...}`` is ``functools.partial`` of it with the
  other keys.

``name`` may be dotted (``Config.numerical``). Values are resolved depth first,
so any argument may itself be one of these forms. Overrides replace values by
dotted key before anything is resolved -- ``steps=100``,
``backend.execution_gib=12`` -- each value read as JSON, or taken as a string
when it is not JSON.
"""

from __future__ import annotations

import functools
import importlib
import json
from pathlib import Path
from typing import Any

CALL = "@call"
PARTIAL = "@partial"


def load(
    path: str | Path, overrides: list[str] | tuple[str, ...] = ()
) -> dict[str, Any]:
    """The config at ``path``, with ``key=value`` overrides applied, unresolved."""

    config = json.loads(Path(path).read_text())
    for override in overrides:
        key, separator, text = override.partition("=")
        if not separator:
            raise ValueError(f"an override is key=value, not {override!r}")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        *parents, name = key.split(".")
        target = config
        for parent in parents:
            target = target.setdefault(parent, {})
        target[name] = value
    return config


def resolve(value: Any) -> Any:
    """``value`` with every reference in it replaced by the object it names."""

    if isinstance(value, str) and value.startswith("@"):
        return reference(value[1:])
    if isinstance(value, list):
        return [resolve(item) for item in value]
    if not isinstance(value, dict):
        return value
    arguments = {key: resolve(item) for key, item in value.items() if key[0] != "@"}
    if CALL in value:
        return reference(value[CALL])(**arguments)
    if PARTIAL in value:
        return functools.partial(reference(value[PARTIAL]), **arguments)
    return arguments


def reference(text: str) -> Any:
    """The object ``module:name`` names; ``name`` may be dotted."""

    module, separator, name = text.partition(":")
    if not separator:
        raise ValueError(f"a reference is module:name, not {text!r}")
    found: Any = importlib.import_module(module)
    for part in name.split("."):
        found = getattr(found, part)
    return found
