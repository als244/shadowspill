"""The records a plan report is built from.

What a training plan says about itself is in ``training`` and a forward
plan in ``forward``; how each stage is named is in ``keys``, and one
captured graph with everything it allocates in ``graphs``.
"""

from .forward import (
    forward_stage_inventory,
)
from .training import (
    training_stage_inventory,
)

__all__ = ["forward_stage_inventory", "training_stage_inventory"]
