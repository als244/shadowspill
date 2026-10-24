"""A training step's shape and provenance, with no framework present.

These are the two values that describe a *step* rather than a plan:
`StepDataOrdering`, the walk a step takes through its microbatches, and
`StepProgram`, the planning problem a captured step lowered to, with the
provenance that says what produced it.

They live beside the planner rather than inside it because a step is not a
planning concept -- the planner is handed a problem and does not care which
step shape produced it -- and beside the frontend rather than inside it
because nothing here needs a framework. That is load-bearing: a saved corpus of
`StepProgram` values is read, validated and planned with no the framework installed,
which is how a collection run and an evaluation run can be separate processes
on separate machines.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .ordering import StepDataOrdering


def __getattr__(name: str) -> Any:
    modules = {"StepArchive": "archive", "StepProgram": "program"}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f".{modules[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = ["StepArchive", "StepDataOrdering", "StepProgram"]
