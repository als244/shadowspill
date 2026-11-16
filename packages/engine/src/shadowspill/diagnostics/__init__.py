"""What one traced step turned out to cost, as immutable evidence.

`step` is that evidence: the summary, the two timelines and the clock each is
read against, what the allocator did, and what the runtime observed. Every
number in it is seconds or bytes, so reading a step back needs no framework --
the frontend measures, and this is what the measurement became.

Executor records optional device markers and runtime events. `collection/` joins
them with the selected simulation; it does not change task execution or costs.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .step import StepDiagnostics, StepTimingSummary, Timelines

__all__ = ["StepDiagnostics", "StepTimingSummary", "Timelines"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(".step", __name__), name)
    globals()[name] = value
    return value
