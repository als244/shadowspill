"""One capacity refinement: how much less PressureFit was given."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdmissionRefinement:
    """One monotonic reduction of logical object capacity after slab replay."""

    attempt: int
    previous_object_capacity_bytes: int
    required_additional_slack_bytes: int
    reserve_increment_bytes: int
    object_capacity_bytes: int
