"""PyTorch's bridge from one plan to the runtime it runs on.

What is left here is the part that needs PyTorch: the bridge that binds a plan's
objects to storages and crosses the task boundary, and the telemetry and trace
records read off a run. The runtime itself is
:mod:`shadowspill.runtime`.
"""

from .bridge import RuntimeBridge, actions_by_task

__all__ = ["RuntimeBridge", "actions_by_task"]
