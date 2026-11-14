"""In-memory read/write backend exercising the real bounded spill access path."""

import torch

from shadowspill.execution.spill import SpillStorage


class MemorySpill(SpillStorage):
    def __init__(self, values):
        self.values = values
        self.sizes = {name: value.nbytes for name, value in values.items()}
        self.addressable = False
        self.transfers = []

    def _transfer(self, alias, value, start, direction):
        self.transfers.append((alias, start, value.nbytes, direction))
        raw = self.values[alias].reshape(-1).view(torch.uint8)
        target = raw[start : start + value.nbytes]
        if direction == "read":
            value.copy_(target)
        else:
            target.copy_(value)
