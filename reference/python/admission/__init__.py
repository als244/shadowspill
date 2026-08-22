"""Readable admission oracles; production uses the compiled planner."""

from .placement import place_lifetimes
from .replay import replay_admission

__all__ = ["place_lifetimes", "replay_admission"]
