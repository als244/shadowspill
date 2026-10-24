"""Ordinary GPU events adapted to the shared timing-window policy."""

from collections.abc import Callable
from typing import Any

import torch


class EventTimer:
    def __init__(self, function: Callable[[], Any]):
        self.function = function
        self.start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        self.end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]

    def sample(self) -> int:
        self.start.record()
        self.function()
        self.end.record()
        self.end.synchronize()
        return round(float(self.start.elapsed_time(self.end)) * 1e6)
