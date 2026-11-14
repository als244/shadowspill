"""Deterministic rank-local examples with a resumable update cursor."""

import torch


class Inputs:
    def __init__(self, rank, seed=711):
        self.rank, self.seed, self.index = rank, seed, 0

    def __iter__(self):
        return self

    def __next__(self):
        value = self.example()
        self.index += 1
        return value

    def example(self):
        generator = torch.Generator().manual_seed(
            self.seed + self.rank * 1_000_003 + self.index
        )
        return {
            "args": [(torch.randn(8, 16, generator=generator),)],
            "kwargs": [{"targets": torch.randn(8, 8, generator=generator)}],
        }

    def state_dict(self):
        return {"rank": self.rank, "seed": self.seed, "index": self.index}

    def load_state_dict(self, state):
        if state["rank"] != self.rank or state["seed"] != self.seed:
            raise ValueError("checkpoint input source belongs to another rank/seed")
        index = state["index"]
        if type(index) is not int or index < 0:
            raise ValueError("invalid input cursor")
        self.index = index
