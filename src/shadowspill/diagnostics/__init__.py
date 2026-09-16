"""What one traced step turned out to cost, as immutable evidence.

`step` is that evidence: the summary, the two timelines and the clock each is
read against, what the allocator did, and what the runtime observed. Every
number in it is seconds or bytes, so reading a step back needs no framework --
the frontend measures, and this is what the measurement became.

Reading the device's timing events *is* the frontend's, in
the frontend's diagnostics, because the events are the framework's own
objects and only it can ask them what they recorded.
"""

from .step import StepDiagnostics, StepTimingSummary, Timelines

__all__ = ["StepDiagnostics", "StepTimingSummary", "Timelines"]
